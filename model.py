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

    def paramsCalc(self):
        encoderPara = sum(p.numel() for p in self.encoder.parameters())
        embedPara = sum(p.numel() for p in self.nerEmbed.parameters())
        nerPara = sum(p.numel() for p in self.nerClassifier.parameters())
        proPara = sum(p.numel() for p in self.projector.parameters())
        decoderPara = sum(p.numel() for p in self.decoder.parameters())
        lmPara = sum(p.numel() for p in self.lmHead.parameters())

        print("Total: ", encoderPara + embedPara + nerPara + proPara + decoderPara + lmPara)

    def forward(self, srcText, targetLang, targetText = None, nerTags = None, returnLoss = True, device = "cpu"):
        taggedSrc = [f"<2{targetLang}> {text}" for text in srcText]
        srcEncoded = self.tokenizer(
            taggedSrc,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = 128
        ).to(device)
        labels = None
    
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

        tagIndices = nerTags if nerTags is not None else nerLogits.argmax(dim = -1)
        nerEmbeds = self.nerEmbed(tagIndices)
        nerOut = encoderOut + nerEmbeds

        projected = self.projector(nerOut)

        decoderOut = self.decoder(
            encoder_attention_mask = srcEncoded.attention_mask,
            encoder_hidden_states = projected,
            return_dict = True
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
            # --- Encode source ---
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
            ).last_hidden_state                                 # (B, S, encoder_hidden)

        # --- NER embedding (predicted tags, no ground truth at inference) ---
            nerLogits = self.nerClassifier(encoderOut)
            predTags  = nerLogits.argmax(dim=-1)               # (B, S)
            nerEmbeds = self.nerEmbed(predTags)                 # (B, S, encoder_hidden)
            nerOut    = encoderOut + nerEmbeds

            projected = self.projector(nerOut)                 # (B, S, decoder_hidden)

        # --- Autoregressive decoding with beam search ---
            batchSize   = projected.size(0)
            bosTokenId  = self.tokenizer.bos_token_id or self.tokenizer.pad_token_id
            eosTokenId  = self.tokenizer.eos_token_id
            padTokenId  = self.tokenizer.pad_token_id

        # Expand encoder outputs for beam search: (B*numBeams, S, H)
            expanded = projected.unsqueeze(1) \
                            .expand(-1, numBeams, -1, -1) \
                            .reshape(batchSize * numBeams, projected.size(1), projected.size(2))

            expandedMask = srcEncoded.attention_mask \
                                 .unsqueeze(1) \
                                 .expand(-1, numBeams, -1) \
                                 .reshape(batchSize * numBeams, -1)

        # Initialise beam hypotheses
            beamScores  = torch.zeros(batchSize, numBeams, device=device)
            beamScores[:, 1:] = -1e9                           # only first beam is live at t=0
            beamScores  = beamScores.view(-1)                  # (B*numBeams,)

        # decoder input: BOS token for every beam
            inputIds = torch.full(
                (batchSize * numBeams, 1),
                bosTokenId,
                dtype=torch.long,
                device=device
            )

            done        = [False] * batchSize
            finishedSeqs= [[] for _ in range(batchSize)]       # list of (score, ids) per batch item

            for _ in range(maxNewTokens):
                decoderOut = self.decoder(
                    input_ids=inputIds,
                    encoder_hidden_states=expanded,
                    encoder_attention_mask=expandedMask,
                    return_dict=True
                )
                logits      = self.lmHead(decoderOut.last_hidden_state[:, -1, :])  # (B*beams, vocab)
                logProbs    = torch.log_softmax(logits, dim=-1)                    # (B*beams, vocab)

                vocabSize   = logProbs.size(-1)
                nextScores  = beamScores.unsqueeze(-1) + logProbs                  # (B*beams, vocab)
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
                        beamIdx  = idx // vocabSize
                        tokenIdx = idx  % vocabSize
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

        # --- Pick best hypothesis per batch item ---
            translations = []
            for b in range(batchSize):
                if finishedSeqs[b]:
                # highest score among completed sequences
                    best = max(finishedSeqs[b], key=lambda x: x[0])
                    tokenIds = best[1][1:]                     # strip BOS
                else:
                # fallback: take the top beam's current sequence
                    tokenIds = inputIds[b * numBeams].tolist()[1:]

                text = self.tokenizer.decode(tokenIds, skip_special_tokens=True)
                translations.append(text)

            return translations
