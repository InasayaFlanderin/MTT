"""
model.py – Kiến trúc dịch máy
Encoder : jhu-clsp/mmBERT-small  (ModernBERT-based, hidden=384, Gemma-2 tokenizer)
Decoder : google/mt5-small        (d_model=512)
Bridge  : EncoderProjection 384 → 512

Thông số mmBERT-small (từ HF model card):
  Layers          : 22
  Hidden Size     : 384
  Intermediate    : 1152
  Attention Heads : 6
  Total Params    : 140M  (gồm embedding vocab 256k)
  Non-embed Params: 42M
  Max Seq Length  : 8192
  Tokenizer       : Gemma 2 (vocab 256,000)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


# ──────────────────────────────────────────────────────────────
# 1. ENCODER – mmBERT-small
# ──────────────────────────────────────────────────────────────
class mmBERTEncoder(nn.Module):
    """
    Bao lớp mmBERT-small làm encoder.

    Kiến trúc kế thừa từ ModernBERT:
      • RoPE thay vì sinusoidal positional encoding
      • Flash Attention 2  (kích hoạt tự động nếu cài flash-attn)
      • Unpadding          (bỏ padding khi compute attention)
      • Gemma-2 tokenizer  (vocab 256k, hỗ trợ 1800+ ngôn ngữ)
    """

    MMBERT_SMALL = "jhu-clsp/mmBERT-small"
    MMBERT_BASE  = "jhu-clsp/mmBERT-base"

    def __init__(self, model_name: str = MMBERT_SMALL, freeze: bool = False):
        super().__init__()

        # Thử kích hoạt Flash Attention 2 (cần: pip install flash-attn)
        try:
            self.bert = AutoModel.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            print("  ✅ Encoder: Flash Attention 2 được kích hoạt")
        except Exception:
            self.bert = AutoModel.from_pretrained(model_name)
            print("  ℹ️  Encoder: dùng Eager Attention (flash-attn chưa được cài)")

        self.hidden_size = self.bert.config.hidden_size  # 384 (small) / 768 (base)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
            print("  🔒 Encoder bị đóng băng (freeze_encoder=True)")

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, src_len)
        attention_mask: torch.Tensor,  # (B, src_len)
    ) -> torch.Tensor:
        """Trả về encoder hidden states: (B, src_len, 384)"""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


# ──────────────────────────────────────────────────────────────
# 2. PROJECTION – mmBERT 384 → mT5 512
# ──────────────────────────────────────────────────────────────
class EncoderProjection(nn.Module):
    """
    Ánh xạ encoder hidden states (384) sang không gian decoder (512).
        Linear(384→512) → LayerNorm(512) → Dropout
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # (B, L, 512)


# ──────────────────────────────────────────────────────────────
# 3. MÔ HÌNH ĐẦY ĐỦ
# ──────────────────────────────────────────────────────────────
class TranslationModel(nn.Module):
    """
    Pipeline hoàn chỉnh:

        Input (src)
            ↓  [Gemma-2 tokenizer – vocab 256k]
        mmBERT-small Encoder
            ↓  encoder hidden states (B, L, 384)
        EncoderProjection
            ↓  projected states (B, L, 512)
        Cross-Attention  ← [mT5 SentencePiece tokenizer]
        mT5-small Decoder
            ↓
        Translated sentence

    Decoder mT5 tích hợp sẵn:
      • Masked Self-Attention  : chỉ nhìn token trước (autoregressive)
      • Cross-Attention        : nhìn encoder projected states
      • Language Modeling head : sinh xác suất trên vocab
    """

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
        print(f"\nKhởi tạo TranslationModel:")
        print(f"  Encoder : {encoder_name}")
        print(f"  Decoder : {decoder_name}")

        # 3a. mmBERT encoder
        self.encoder  = mmBERTEncoder(encoder_name, freeze=freeze_encoder)
        encoder_dim   = self.encoder.hidden_size          # 384

        # 3b. mT5 decoder
        self.mt5      = MT5ForConditionalGeneration.from_pretrained(decoder_name)
        decoder_dim   = self.mt5.config.d_model           # 512

        # 3c. Projection bridge
        self.projection = EncoderProjection(encoder_dim, decoder_dim, proj_dropout)
        print(f"  Projection: {encoder_dim} → {decoder_dim}")

    # ── FORWARD (training) ───────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
    ):
        enc_hidden    = self.encoder(input_ids, attention_mask)  # (B,L,384)
        enc_projected = self.projection(enc_hidden)               # (B,L,512)
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_projected)

        return self.mt5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    # ── GENERATE (inference) ─────────────────────────────────
    @torch.no_grad()
    def translate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Beam Search mặc định (num_beams=4).
        Đổi num_beams=1 → Greedy Search.
        """
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

    # ── TIỆN ÍCH ─────────────────────────────────────────────
    def count_parameters(self) -> dict:
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
# 4. QUICK SANITY CHECK
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = TranslationModel().to(device)

    print("\nThông số mô hình:")
    for k, v in model.count_parameters().items():
        print(f"  {k:30s}: {v}")

    B, SL, TL = 2, 32, 24
    src  = torch.randint(0, 1000, (B, SL)).to(device)
    mask = torch.ones(B, SL, dtype=torch.long).to(device)
    lbl  = torch.randint(0, 1000, (B, TL)).to(device)

    out = model(input_ids=src, attention_mask=mask, labels=lbl)
    print(f"\nForward OK  → loss={out.loss.item():.4f}, logits={out.logits.shape}")

    gen = model.translate(src, mask, max_new_tokens=16, num_beams=2)
    print(f"Generate OK → output shape={gen.shape}")