import torch
import torch.nn as nn
import torch.nn.functional as func
from transformers import (
    AutoModel,
    MT5ForConditionalGeneration,
    AutoTokenizer
)

class MTT(nn.Module):
    def __init__(self):
        super().__init__()

        self.targetLanguages = ["de", "fr", "vi", "en", "es"]
        specialTokens = [f"<2{lang}>" for lang in self.targetLanguages]
        
        #encoder
        encoderName = "jhu-clsp/mmBERT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(encoderName)
        self.tokenizer.add_special_tokens({"additional_special_tokens": specialTokens})
        self.encoder = AutoModel.from_pretrained(encoderName)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        encoderHiddenSize = self.encoder.config.hidden_size

        #NER
        self.nerVocab = {
            'O': 0, 'PERSON': 1, 'ORG': 2, 'LOC': 3, 'GPE': 4,
            'DATE': 5, 'MONEY': 6, 'PERCENT': 7, 'TIME': 8, 'QUANTITY': 9
        }
        self.nerEmbed = nn.Embedding(
            num_embeddings = len(self.nerVocab),
            embedding_dim = encoderHiddenSize
        )
        self.nerClassifier = nn.Linear(
            encoderHiddenSize,
            len(self.nerVocab)
        )

        #decoder
        mt5 = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        mt5.resize_token_embeddings(len(self.tokenizer))
        self.decoder = mt5.decoder
        self.lmHead = mt5.lm_head
        self.decoderStartTokenId = mt5.config.decoder_start_token_id or mt5.config.pad_token_id
        del mt5

        #projector
        self.projector = nn.Linear(
            encoderHiddenSize,
            self.decoder.config.hidden_size
        )

        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
        for param in self.projector.parameters():
            param.requires_grad = True
        for param in self.lmHead.parameters():
            param.requires_grad = True

    # ══════════════════════════════════════════════════════════════════════════
    # LOAD CHECKPOINT
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "MTT":
        """
        Tạo model mới rồi load weights từ file checkpoint của train.py.

        Ví dụ:
            model = MTT.load("mtt_checkpoint.pt", device="cpu")
            model.eval()

        Args:
            checkpoint_path : đường dẫn tới file .pt do train.py lưu
            device          : "cpu" hoặc "cuda"

        Returns:
            MTT instance đã load weights, ở chế độ eval
        """
        if not torch.cuda.is_available() and device == "cuda":
            print("[LOAD] CUDA không khả dụng, chuyển sang CPU.")
            device = "cpu"

        print(f"[LOAD] Đang khởi tạo model...")
        model = cls()

        print(f"[LOAD] Đang load checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Checkpoint từ train.py lưu key "model" chứa state_dict
        if "model" in ckpt:
            state_dict = ckpt["model"]
            print(f"[LOAD] Global step : {ckpt.get('global_step', '?')}")
            print(f"[LOAD] Cycle        : {ckpt.get('cycle', '?')}")
            print(f"[LOAD] LR cuối      : {ckpt.get('lr', '?')}")
        else:
            # Nếu file là raw state_dict (lưu thẳng không qua train.py)
            state_dict = ckpt

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"[LOAD] Load thành công! Model sẵn sàng trên {device}.\n")
        return model

    # ══════════════════════════════════════════════════════════════════════════

    def paramsCalc(self):
        encoderPara = sum(p.numel() for p in self.encoder.parameters())
        embedPara = sum(p.numel() for p in self.nerEmbed.parameters())
        nerPara = sum(p.numel() for p in self.nerClassifier.parameters())
        proPara = sum(p.numel() for p in self.projector.parameters())
        decoderPara = sum(p.numel() for p in self.decoder.parameters())
        lmPara = sum(p.numel() for p in self.lmHead.parameters())

        print("Total: ", encoderPara + embedPara + nerPara + proPara + decoderPara + lmPara)

    def _shiftRight(self, inputIds: torch.Tensor) -> torch.Tensor:
        """Shift input ids one position right, prepend decoder_start_token_id (teacher forcing)."""
        shifted = inputIds.new_zeros(inputIds.shape)
        shifted[:, 1:] = inputIds[:, :-1].clone()
        shifted[:, 0]  = self.decoderStartTokenId
        shifted[shifted == -100] = self.tokenizer.pad_token_id
        return shifted

    def forward(self, srcText, targetLang, targetText=None, nerTags=None, returnLoss=True, device="cpu"):
        taggedSrc = [f"<2{targetLang}> {text}" for text in srcText]
        srcEncoded = self.tokenizer(
            taggedSrc,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = 128
        ).to(device)
        labels = None
        decoderInputIds = None
    
        if targetText is not None:
            targetEncoded = self.tokenizer(
                targetText,
                return_tensors = "pt",
                padding = True,
                truncation = True,
                max_length = 128
            ).to(device)
            
            labels = targetEncoded.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            decoderInputIds = self._shiftRight(labels)

        encoderOut = self.encoder(
            input_ids = srcEncoded.input_ids,
            attention_mask = srcEncoded.attention_mask
        ).last_hidden_state
        
        nerLogits = self.nerClassifier(encoderOut)

        nerLoss = None
        if nerTags is not None:
            nerTags = nerTags.to(device)
            nerLoss = func.cross_entropy(
                nerLogits.view(-1, len(self.nerVocab)),
                nerTags.view(-1),
                ignore_index = -100
            )

        tagIndices = nerTags if nerTags is not None else nerLogits.argmax(dim=-1)
        tagIndices = tagIndices.clamp(min=0)
        nerEmbeds = self.nerEmbed(tagIndices)
        nerOut = encoderOut + nerEmbeds

        projected = self.projector(nerOut)

        if decoderInputIds is None:
            decoderInputIds = torch.full(
                (projected.size(0), 1),
                self.decoderStartTokenId,
                dtype=torch.long,
                device=device
            )

        decoderOut = self.decoder(
            input_ids              = decoderInputIds,
            encoder_hidden_states  = projected,
            encoder_attention_mask = srcEncoded.attention_mask,
            return_dict            = True
        )

        logits = self.lmHead(decoderOut.last_hidden_state)

        translationLoss = None
        if labels is not None:
            translationLoss = func.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index = -100
            )

        totalLoss = None
        if returnLoss and translationLoss is not None:
            totalLoss = translationLoss
            if nerLoss is not None:
                totalLoss = translationLoss + 0.2 * nerLoss

        return {
            "loss": totalLoss,
            "translationLoss": translationLoss,
            "nerLoss": nerLoss,
            "logits": logits,
            "nerLogits": nerLogits
        }

    def translate(
        self,
        srcText: list[str],
        targetLang: str,
        maxNewTokens: int = 128,
        numBeams: int = 4,
        device: str = "cpu"
    ) -> list[str]:

        assert targetLang in self.targetLanguages, \
            f"Unsupported target language '{targetLang}'. Choose from {self.targetLanguages}."

        self.eval()
        with torch.no_grad():
            taggedSrc = [f"<2{targetLang}> {text}" for text in srcText]
            srcEncoded = self.tokenizer(
                taggedSrc,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)

            encoderOut = self.encoder(
                input_ids=srcEncoded.input_ids,
                attention_mask=srcEncoded.attention_mask
            ).last_hidden_state

            nerLogits = self.nerClassifier(encoderOut)
            predTags  = nerLogits.argmax(dim=-1)
            nerEmbeds = self.nerEmbed(predTags)
            nerOut    = encoderOut + nerEmbeds

            projected = self.projector(nerOut)

            batchSize   = projected.size(0)
            bosTokenId  = self.decoderStartTokenId
            eosTokenId  = self.tokenizer.eos_token_id
            padTokenId  = self.tokenizer.pad_token_id

            expanded = projected.unsqueeze(1) \
                            .expand(-1, numBeams, -1, -1) \
                            .reshape(batchSize * numBeams, projected.size(1), projected.size(2))

            expandedMask = srcEncoded.attention_mask \
                                 .unsqueeze(1) \
                                 .expand(-1, numBeams, -1) \
                                 .reshape(batchSize * numBeams, -1)

            beamScores  = torch.zeros(batchSize, numBeams, device=device)
            beamScores[:, 1:] = -1e9
            beamScores  = beamScores.view(-1)

            inputIds = torch.full(
                (batchSize * numBeams, 1),
                bosTokenId,
                dtype=torch.long,
                device=device
            )

            done         = [False] * batchSize
            finishedSeqs = [[] for _ in range(batchSize)]

            for _ in range(maxNewTokens):
                decoderOut = self.decoder(
                    input_ids=inputIds,
                    encoder_hidden_states=expanded,
                    encoder_attention_mask=expandedMask,
                    return_dict=True
                )
                logits      = self.lmHead(decoderOut.last_hidden_state[:, -1, :])
                logProbs    = torch.log_softmax(logits, dim=-1)

                vocabSize   = logProbs.size(-1)
                nextScores  = beamScores.unsqueeze(-1) + logProbs
                nextScores  = nextScores.view(batchSize, numBeams * vocabSize)

                topScores, topIndices = torch.topk(nextScores, 2 * numBeams, dim=-1)

                nextBeamScores  = []
                nextBeamTokens  = []
                nextBeamOrigins = []

                for b in range(batchSize):
                    if done[b]:
                        nextBeamScores.extend([0.0] * numBeams)
                        nextBeamTokens.extend([padTokenId] * numBeams)
                        nextBeamOrigins.extend(list(range(b * numBeams, (b + 1) * numBeams)))
                        continue

                    beamsCounted = 0
                    for score, idx in zip(topScores[b].tolist(), topIndices[b].tolist()):
                        beamIdx    = idx // vocabSize
                        tokenIdx   = idx  % vocabSize
                        globalBeam = b * numBeams + beamIdx

                        if tokenIdx == eosTokenId:
                            finishedSeqs[b].append((score, inputIds[globalBeam].tolist()))
                        else:
                            nextBeamScores.append(score)
                            nextBeamTokens.append(tokenIdx)
                            nextBeamOrigins.append(globalBeam)
                            beamsCounted += 1

                        if beamsCounted == numBeams:
                            break

                    if len(finishedSeqs[b]) >= numBeams:
                        done[b] = True

                if all(done):
                    break

                beamScores = torch.tensor(nextBeamScores, device=device)
                nextTokens = torch.tensor(nextBeamTokens, dtype=torch.long, device=device).unsqueeze(-1)
                inputIds   = torch.cat([inputIds[nextBeamOrigins], nextTokens], dim=-1)

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