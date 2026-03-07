"""
model.py – Kiến trúc dịch máy đa ngôn ngữ
Encoder : jhu-clsp/mmBERT-small  (ModernBERT-based, hidden=384, Gemma-2 tokenizer)
Decoder : google/mt5-small        (d_model=512)
Bridge  : EncoderProjection 384 → 512

Hỗ trợ 6 chiều dịch bằng language token:
    >>vi<<  >>en<<  >>fr<<  ghép vào đầu câu nguồn
    Ví dụ: ">>vi<< Hello world" → model biết cần dịch sang tiếng Việt
"""

import torch
import torch.nn as nn
from transformers import AutoModel, MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


# ──────────────────────────────────────────────────────────────
# 1. ENCODER – mmBERT-small
# ──────────────────────────────────────────────────────────────
class mmBERTEncoder(nn.Module):
    MMBERT_SMALL = "jhu-clsp/mmBERT-small"

    def __init__(self, model_name: str = MMBERT_SMALL, freeze: bool = False):
        super().__init__()
        try:
            self.bert = AutoModel.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            print("  ✅ Encoder: Flash Attention 2 được kích hoạt")
        except Exception:
            self.bert = AutoModel.from_pretrained(model_name)
            print("  ℹ️  Encoder: dùng Eager Attention")

        self.hidden_size = self.bert.config.hidden_size  # 384

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
            print("  🔒 Encoder bị đóng băng")

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, L, 384)


# ──────────────────────────────────────────────────────────────
# 2. PROJECTION – 384 → 512
# ──────────────────────────────────────────────────────────────
class EncoderProjection(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, dropout: float = 0.1):
        super().__init__()
        linear = nn.Linear(encoder_dim, decoder_dim)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        self.proj = nn.Sequential(
            linear,
            nn.LayerNorm(decoder_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x)  # (B, L, 512)


# ──────────────────────────────────────────────────────────────
# 3. MÔ HÌNH ĐẦY ĐỦ
# ──────────────────────────────────────────────────────────────
class TranslationModel(nn.Module):
    """
    Một model duy nhất hỗ trợ tất cả cặp ngôn ngữ.
    Chiều dịch được điều khiển bởi language token trong câu nguồn:

        ">>vi<< Hello world"  →  "Xin chào thế giới"
        ">>en<< Bonjour"      →  "Hello"
        ">>fr<< Xin chào"     →  "Bonjour"

    Các cặp hỗ trợ: en↔vi, en↔fr, vi↔fr  (6 chiều)
    """

    # Language tokens (ghép vào đầu câu nguồn)
    LANG_TOKENS = {
        "en": ">>en<<",
        "vi": ">>vi<<",
        "fr": ">>fr<<",
    }

    # 6 chiều dịch được hỗ trợ
    SUPPORTED_PAIRS = [
        ("en", "vi"), ("vi", "en"),
        ("en", "fr"), ("fr", "en"),
        ("vi", "fr"), ("fr", "vi"),
    ]

    DEFAULT_ENCODER = "jhu-clsp/mmBERT-small"
    DEFAULT_DECODER = "google/mt5-small"

    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        decoder_name: str = DEFAULT_DECODER,
        freeze_encoder: bool = False,
        proj_dropout: float = 0.1,
    ):
        super().__init__()
        print(f"\nKhởi tạo TranslationModel (đa ngôn ngữ):")
        print(f"  Encoder : {encoder_name}")
        print(f"  Decoder : {decoder_name}")
        print(f"  Hỗ trợ  : {len(self.SUPPORTED_PAIRS)} chiều dịch")

        self.encoder    = mmBERTEncoder(encoder_name, freeze=freeze_encoder)
        encoder_dim     = self.encoder.hidden_size        # 384

        self.mt5        = MT5ForConditionalGeneration.from_pretrained(decoder_name)
        decoder_dim     = self.mt5.config.d_model         # 512

        self.projection = EncoderProjection(encoder_dim, decoder_dim, proj_dropout)
        print(f"  Projection: {encoder_dim} → {decoder_dim}")

    @staticmethod
    def add_lang_token(text: str, tgt_lang: str) -> str:
        """Ghép language token vào đầu câu nguồn."""
        token = TranslationModel.LANG_TOKENS.get(tgt_lang, f">>{tgt_lang}<<")
        return f"{token} {text}"

    def forward(self, input_ids, attention_mask, labels=None,
                decoder_input_ids=None):
        enc_hidden      = self.encoder(input_ids, attention_mask)
        enc_projected   = self.projection(enc_hidden)
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_projected)
        return self.mt5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    @torch.no_grad()
    def translate(self, input_ids, attention_mask,
                  max_new_tokens=128, num_beams=4,
                  length_penalty=1.0, early_stopping=True, **kwargs):
        enc_hidden      = self.encoder(input_ids, attention_mask)
        enc_projected   = self.projection(enc_hidden)
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_projected)
        return self.mt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            **kwargs,
        )

    def count_parameters(self):
        fmt = lambda n: f"{n/1e6:.2f}M"
        enc  = sum(p.numel() for p in self.encoder.parameters())
        proj = sum(p.numel() for p in self.projection.parameters())
        dec  = sum(p.numel() for p in self.mt5.parameters())
        trn  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "mmBERT-small encoder" : fmt(enc),
            "projection (384→512)" : fmt(proj),
            "mT5-small decoder"    : fmt(dec),
            "total"                : fmt(enc + proj + dec),
            "trainable"            : fmt(trn),
        }


# ──────────────────────────────────────────────────────────────
# 4. SANITY CHECK
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = TranslationModel().to(device)
    for k, v in model.count_parameters().items():
        print(f"  {k:30s}: {v}")

    # Kiểm tra language token
    for src, tgt in TranslationModel.SUPPORTED_PAIRS:
        example = TranslationModel.add_lang_token("Hello world", tgt)
        print(f"  {src}→{tgt}: '{example}'")