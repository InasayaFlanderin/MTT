import torch
import torch.nn as nn
import torch.nn.functional as func
from transformers import (
    AutoModel,
    AutoTokenizer,
    MT5ForConditionalGeneration,
    T5TokenizerFast,
)


class MTT(nn.Module):
    def __init__(self):
        super().__init__()

        self.targetLanguages = ["de", "fr", "vi", "en", "es"]
        specialTokens = [f"<2{lang}>" for lang in self.targetLanguages]

        # ── Source tokenizer: mmBERT's own Gemma 2-based tokenizer (~256k vocab) ──
        # mmBERT is built on ModernBERT and uses a Gemma 2 tokenizer, NOT WordPiece.
        # Its embedding table (98M of the 140M total params) was trained with this
        # vocabulary. Using any other tokenizer and then resizing would silently
        # remap the trained embedding weights to wrong token IDs, destroying the
        # encoder's pretrained knowledge.
        # The source (encoder) and target (decoder) tokenizers do NOT need to share
        # a vocabulary — they are connected only through cross-attention on projected
        # hidden states. We keep each tokenizer native to its model.
        # AutoTokenizer for mmBERT loads GemmaTokenizerFast, which is a Rust-backed
        # fast tokenizer and supports return_offsets_mapping=True (required by NER).
        self.srcTokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
        self.srcTokenizer.add_special_tokens({"additional_special_tokens": specialTokens})

        # ── Target tokenizer: MT5's SentencePiece tokenizer (~250k vocab) ────────
        # Used to encode target sequences for training and decode output token ids.
        # FIX (CRITICAL): Must be the Fast variant — the slow T5Tokenizer raises
        # NotImplementedError on return_offsets_mapping=True, causing NER to silently
        # never train (the exception was swallowed by try/except in train.py).
        self.tokenizer = T5TokenizerFast.from_pretrained("google/mt5-small")
        # Note: special lang tokens are only needed on the source side; the decoder
        # never sees them as input tokens. No resize needed for the target tokenizer.

        # ── Encoder: mmBERT (Modern Multilingual BERT) ────────────────────────
        encoderName = "jhu-clsp/mmBERT-small"
        self.encoder = AutoModel.from_pretrained(encoderName)

        # Only resize for the newly added special language tokens — preserves all
        # existing Gemma 2 embedding weights, appending new rows for <2de> etc.
        self.encoder.resize_token_embeddings(len(self.srcTokenizer))
        encoderHiddenSize = self.encoder.config.hidden_size

        # ── NER ───────────────────────────────────────────────────────────────
        self.nerVocab = {
            'O': 0, 'PERSON': 1, 'ORG': 2, 'LOC': 3, 'GPE': 4,
            'DATE': 5, 'MONEY': 6, 'PERCENT': 7, 'TIME': 8, 'QUANTITY': 9
        }
        self.nerEmbed = nn.Embedding(
            num_embeddings=len(self.nerVocab),
            embedding_dim=encoderHiddenSize,
        )
        self.nerClassifier = nn.Linear(
            encoderHiddenSize,
            len(self.nerVocab),
        )

        # ── Decoder: MT5 (native vocab, no resizing needed) ──────────────────
        mt5 = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        # No resize — MT5's lm_head and decoder embeddings stay aligned with
        # its own SentencePiece vocab, which self.tokenizer already matches.

        # FIX 2: Use `is not None` instead of truthiness check.
        # mt5.config.decoder_start_token_id == 0 for MT5, and `0 or x` evaluates
        # to x in Python (0 is falsy), so the original code always fell through
        # to pad_token_id.  Both happen to be 0 for MT5-small so no runtime
        # difference, but the logic was wrong and would break on other models.
        self.decoderStartTokenId = (
            mt5.config.decoder_start_token_id
            if mt5.config.decoder_start_token_id is not None
            else mt5.config.pad_token_id
        )

        self.decoder = mt5.decoder
        self.lmHead  = mt5.lm_head
        del mt5

        # Đảm bảo tokenizer có pad_token_id hợp lệ
        assert self.tokenizer.pad_token_id is not None, \
            "Tokenizer thiếu pad_token_id — kiểm tra lại tokenizer_name."

        # ── Projector: ánh xạ encoder dim → decoder dim ───────────────────────
        self.projector = nn.Linear(
            encoderHiddenSize,
            self.decoder.config.hidden_size,
        )

        # Tất cả params đều trainable
        for module in [self.encoder, self.decoder, self.projector, self.lmHead,
                       self.nerEmbed, self.nerClassifier]:
            for param in module.parameters():
                param.requires_grad = True

        # Uncertainty weighting (Kendall et al.)
        self.log_var_trans = nn.Parameter(torch.zeros(1))
        self.log_var_ner   = nn.Parameter(torch.zeros(1))

    # ══════════════════════════════════════════════════════════════════════════
    # LOAD CHECKPOINT
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "MTT":
        if not torch.cuda.is_available() and device == "cuda":
            print("[LOAD] CUDA không khả dụng, chuyển sang CPU.")
            device = "cpu"

        print("[LOAD] Đang khởi tạo model...")
        model = cls()

        print(f"[LOAD] Đang load checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        if "model" in ckpt:
            state_dict = ckpt["model"]
            print(f"[LOAD] Global step : {ckpt.get('global_step', '?')}")
            print(f"[LOAD] Cycle        : {ckpt.get('cycle', '?')}")
            print(f"[LOAD] LR cuối      : {ckpt.get('lr', '?')}")
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"[LOAD] Load thành công! Model sẵn sàng trên {device}.\n")
        return model

    # ══════════════════════════════════════════════════════════════════════════

    def paramsCalc(self):
        encoderPara  = sum(p.numel() for p in self.encoder.parameters())
        embedPara    = sum(p.numel() for p in self.nerEmbed.parameters())
        nerPara      = sum(p.numel() for p in self.nerClassifier.parameters())
        proPara      = sum(p.numel() for p in self.projector.parameters())
        decoderPara  = sum(p.numel() for p in self.decoder.parameters())
        lmPara       = sum(p.numel() for p in self.lmHead.parameters())
        total        = encoderPara + embedPara + nerPara + proPara + decoderPara + lmPara
        print(f"  Encoder     : {encoderPara:>12,}")
        print(f"  NER embed   : {embedPara:>12,}")
        print(f"  NER linear  : {nerPara:>12,}")
        print(f"  Projector   : {proPara:>12,}")
        print(f"  Decoder     : {decoderPara:>12,}")
        print(f"  LM Head     : {lmPara:>12,}")
        print(f"  ─────────────────────────")
        print(f"  Total       : {total:>12,}")

    def _shiftRight(self, inputIds: torch.Tensor) -> torch.Tensor:
        shifted = inputIds.new_zeros(inputIds.shape)
        shifted[:, 1:] = inputIds[:, :-1].clone()
        shifted[:, 0]  = self.decoderStartTokenId
        shifted[shifted == -100] = self.tokenizer.pad_token_id
        return shifted

    def forward(
        self,
        srcText,
        targetLang,
        targetText=None,
        nerTags=None,
        returnLoss=True,
        device="cpu",
    ):
        taggedSrc = [f"<2{targetLang}> {text}" for text in srcText]
        srcEncoded = self.srcTokenizer(          # source uses mmBERT's tokenizer
            taggedSrc,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        labels = None
        decoderInputIds = None

        if targetText is not None:
            targetEncoded = self.tokenizer(      # target uses MT5's tokenizer
                targetText,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            labels = targetEncoded.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            decoderInputIds = self._shiftRight(labels)

        encoderOut = self.encoder(
            input_ids=srcEncoded.input_ids,
            attention_mask=srcEncoded.attention_mask,
        ).last_hidden_state

        nerLogits = self.nerClassifier(encoderOut)

        nerLoss = None
        if nerTags is not None:
            nerTags = nerTags.to(device)
            nerLoss = func.cross_entropy(
                nerLogits.view(-1, len(self.nerVocab)),
                nerTags.view(-1),
                ignore_index=-100,
            )

        tagIndices = nerTags if nerTags is not None else nerLogits.argmax(dim=-1)
        valid_mask = (tagIndices >= 0).unsqueeze(-1).float()
        tagIndices = tagIndices.clamp(min=0)
        nerEmbeds  = self.nerEmbed(tagIndices) * valid_mask
        nerOut     = encoderOut + nerEmbeds

        projected = self.projector(nerOut)

        if decoderInputIds is None:
            decoderInputIds = torch.full(
                (projected.size(0), 1),
                self.decoderStartTokenId,
                dtype=torch.long,
                device=device,
            )

        decoderOut = self.decoder(
            input_ids=decoderInputIds,
            encoder_hidden_states=projected,
            encoder_attention_mask=srcEncoded.attention_mask,
            return_dict=True,
        )

        logits = self.lmHead(decoderOut.last_hidden_state)

        translationLoss = None
        if labels is not None:
            translationLoss = func.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        totalLoss = None
        if returnLoss and translationLoss is not None:
            precision_trans = torch.exp(-self.log_var_trans)
            precision_ner   = torch.exp(-self.log_var_ner)

            totalLoss = precision_trans * translationLoss + self.log_var_trans

            if nerLoss is not None:
                totalLoss = totalLoss + precision_ner * nerLoss + self.log_var_ner

        return {
            "loss":            totalLoss,
            "translationLoss": translationLoss,
            "nerLoss":         nerLoss,
            "logits":          logits,
            "nerLogits":       nerLogits,
            "weight_trans":    torch.exp(-self.log_var_trans).item(),
            "weight_ner":      torch.exp(-self.log_var_ner).item(),
        }

    def translate(
        self,
        srcText: list[str],
        targetLang: str,
        maxNewTokens: int = 128,
        numBeams: int = 4,
        device: str = "cpu",
    ) -> list[str]:

        assert targetLang in self.targetLanguages, \
            f"Unsupported target language '{targetLang}'. Choose from {self.targetLanguages}."

        self.eval()
        with torch.no_grad():
            taggedSrc = [f"<2{targetLang}> {text}" for text in srcText]
            srcEncoded = self.srcTokenizer(      # source uses mmBERT's tokenizer
                taggedSrc,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            encoderOut = self.encoder(
                input_ids=srcEncoded.input_ids,
                attention_mask=srcEncoded.attention_mask,
            ).last_hidden_state

            # FIX 3: Removed dead valid_mask — argmax always returns >= 0 so
            # the mask was always all-ones and zeroed nothing.  The mask is only
            # meaningful in forward() where nerTags can contain -100 (padding).
            nerLogits = self.nerClassifier(encoderOut)
            predTags  = nerLogits.argmax(dim=-1)          # always >= 0
            nerEmbeds = self.nerEmbed(predTags)            # no masking needed
            nerOut    = encoderOut + nerEmbeds
            projected = self.projector(nerOut)

            batchSize  = projected.size(0)
            bosTokenId = self.decoderStartTokenId
            eosTokenId = self.tokenizer.eos_token_id
            padTokenId = self.tokenizer.pad_token_id

            expanded = (
                projected.unsqueeze(1)
                .expand(-1, numBeams, -1, -1)
                .reshape(batchSize * numBeams, projected.size(1), projected.size(2))
            )
            expandedMask = (
                srcEncoded.attention_mask.unsqueeze(1)
                .expand(-1, numBeams, -1)
                .reshape(batchSize * numBeams, -1)
            )

            beamScores = torch.zeros(batchSize, numBeams, device=device)
            beamScores[:, 1:] = -1e9
            beamScores = beamScores.view(-1)

            inputIds = torch.full(
                (batchSize * numBeams, 1),
                bosTokenId,
                dtype=torch.long,
                device=device,
            )

            past_key_values = None
            done         = [False] * batchSize
            finishedSeqs = [[] for _ in range(batchSize)]

            for step_i in range(maxNewTokens):
                cur_input = inputIds if step_i == 0 else inputIds[:, -1:]

                decoderOut = self.decoder(
                    input_ids=cur_input,
                    encoder_hidden_states=expanded,
                    encoder_attention_mask=expandedMask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = decoderOut.past_key_values

                logits   = self.lmHead(decoderOut.last_hidden_state[:, -1, :])
                logProbs = torch.log_softmax(logits, dim=-1)

                vocabSize  = logProbs.size(-1)
                nextScores = beamScores.unsqueeze(-1) + logProbs
                nextScores = nextScores.view(batchSize, numBeams * vocabSize)

                topScores, topIndices = torch.topk(nextScores, 2 * numBeams, dim=-1)

                nextBeamScores  = []
                nextBeamTokens  = []
                nextBeamOrigins = []

                for b in range(batchSize):
                    if done[b]:
                        nextBeamScores.extend([0.0] * numBeams)
                        nextBeamTokens.extend([padTokenId] * numBeams)
                        nextBeamOrigins.extend(
                            list(range(b * numBeams, (b + 1) * numBeams))
                        )
                        continue

                    beamsCounted = 0
                    for score, idx in zip(
                        topScores[b].tolist(), topIndices[b].tolist()
                    ):
                        beamIdx    = idx // vocabSize
                        tokenIdx   = idx % vocabSize
                        globalBeam = b * numBeams + beamIdx

                        if tokenIdx == eosTokenId:
                            finishedSeqs[b].append(
                                (score, inputIds[globalBeam].tolist())
                            )
                        else:
                            nextBeamScores.append(score)
                            nextBeamTokens.append(tokenIdx)
                            nextBeamOrigins.append(globalBeam)
                            beamsCounted += 1

                        if beamsCounted == numBeams:
                            break

                    while beamsCounted < numBeams:
                        nextBeamScores.append(-1e9)
                        nextBeamTokens.append(padTokenId)
                        nextBeamOrigins.append(b * numBeams)
                        beamsCounted += 1

                    if len(finishedSeqs[b]) >= numBeams:
                        done[b] = True

                if all(done):
                    break

                beamScores = torch.tensor(nextBeamScores, device=device)
                nextTokens = torch.tensor(
                    nextBeamTokens, dtype=torch.long, device=device
                ).unsqueeze(-1)

                # FIX 4: Convert origins to a tensor once and reuse for both
                # inputIds reorder and past_key_values reorder — avoids implicit
                # list-to-tensor conversion happening twice with different semantics.
                origins_tensor = torch.tensor(
                    nextBeamOrigins, dtype=torch.long, device=device
                )
                inputIds = torch.cat(
                    [inputIds[origins_tensor], nextTokens], dim=-1
                )

                if past_key_values is not None:
                    past_key_values = tuple(
                        tuple(layer_cache[origins_tensor] for layer_cache in layer)
                        for layer in past_key_values
                    )

            translations = []
            for b in range(batchSize):
                if finishedSeqs[b]:
                    best     = max(finishedSeqs[b], key=lambda x: x[0])
                    tokenIds = best[1][1:]
                else:
                    tokenIds = inputIds[b * numBeams].tolist()[1:]

                text = self.tokenizer.decode(tokenIds, skip_special_tokens=True)
                translations.append(text)

            return translations