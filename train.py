"""
train.py – Training Pipeline (tối ưu cho Google Colab)
Encoder: jhu-clsp/mmBERT-small  (Gemma-2 tokenizer, vocab 256k)
Decoder: google/mt5-small        (SentencePiece tokenizer)
Data   : opus-100 en→vi (thay đổi SRC_LANG/TGT_LANG nếu cần)

Cài đặt trước khi chạy trên Colab:
    !pip install transformers datasets sentencepiece sacrebleu accelerate -q
    # (Tuỳ chọn) Flash Attention 2 để tăng tốc mmBERT:
    !pip install flash-attn --no-build-isolation -q
"""

import os, math, time, json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import sacrebleu

from model import TranslationModel

# ──────────────────────────────────────────────────────────────
# 1. HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────
CFG = {
    # Model
    "encoder_name":   "jhu-clsp/mmBERT-small",
    "decoder_name":   "google/mt5-small",
    "freeze_encoder": False,   # True = chỉ train decoder + projection (tiết kiệm VRAM)

    # Data
    "src_lang":       "en",
    "tgt_lang":       "vi",
    # mmBERT hỗ trợ tới 8192 tokens, nhưng giữ 128 để tiết kiệm VRAM trên Colab
    "max_src_len":    128,
    "max_tgt_len":    128,
    "train_samples":  50_000,  # None = dùng toàn bộ dataset
    "val_samples":    2_000,

    # Training
    "batch_size":     16,      # Giảm xuống 8 nếu hết VRAM
    "grad_accum":     4,       # Effective batch = 16 × 4 = 64
    "epochs":         5,
    "lr":             3e-4,
    "warmup_ratio":   0.1,
    "weight_decay":   0.01,
    "max_grad_norm":  1.0,
    "fp16":           True,    # Mixed precision (cần GPU Colab)

    # Paths (Google Drive)
    "output_dir": "/content/drive/MyDrive/translation_model",
    "log_file":   "/content/drive/MyDrive/translation_model/train_log.json",
}

torch.manual_seed(42)


# ──────────────────────────────────────────────────────────────
# 2. DATASET
# ──────────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    """
    Tokenize cặp câu nguồn–đích bằng hai tokenizer khác nhau.

    Encoder tokenizer : Gemma-2 (AutoTokenizer từ jhu-clsp/mmBERT-small)
                        vocab 256k, hỗ trợ 1800+ ngôn ngữ
    Decoder tokenizer : SentencePiece (từ google/mt5-small)
    """

    def __init__(
        self,
        pairs: list,
        enc_tokenizer,
        dec_tokenizer,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
    ):
        self.pairs       = pairs
        self.enc_tok     = enc_tokenizer
        self.dec_tok     = dec_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src = self.pairs[idx]["src"]
        tgt = self.pairs[idx]["tgt"]

        # Tokenize nguồn bằng Gemma-2 tokenizer (mmBERT)
        enc = self.enc_tok(
            src,
            max_length=self.max_src_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize đích bằng SentencePiece tokenizer (mT5)
        dec = self.dec_tok(
            tgt,
            max_length=self.max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Padding token → -100 để bỏ qua khi tính Cross-Entropy Loss
        labels = dec["input_ids"].squeeze(0).clone()
        labels[labels == self.dec_tok.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


def load_opus_pairs(src_lang, tgt_lang, split, max_samples=None):
    """Tải opus-100 và trả về list dict {src, tgt}."""
    print(f"Đang tải opus-100 {src_lang}-{tgt_lang} [{split}]...")
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", f"{src_lang}-{tgt_lang}", split=split)
    except Exception:
        ds = load_dataset("Helsinki-NLP/opus-100", f"{tgt_lang}-{src_lang}", split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    pairs = []
    for item in ds:
        tr  = item["translation"]
        src = tr.get(src_lang) or tr.get(tgt_lang)
        tgt = tr.get(tgt_lang) or tr.get(src_lang)
        if src and tgt:
            pairs.append({"src": src, "tgt": tgt})

    print(f"  → {len(pairs)} cặp câu")
    return pairs


# ──────────────────────────────────────────────────────────────
# 3. METRICS
# ──────────────────────────────────────────────────────────────
def compute_bleu(predictions: list, references: list) -> float:
    refs = [[r] for r in references]
    bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))
    return bleu.score


# ──────────────────────────────────────────────────────────────
# 4. VALIDATION
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, dec_tokenizer, device, num_beams=4):
    model.eval()
    total_loss, steps = 0.0, 0
    preds, refs = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += out.loss.item()
        steps += 1

        # Tính BLEU trên 200 câu đầu (tiết kiệm thời gian)
        if len(preds) < 200:
            generated = model.translate(
                input_ids, attention_mask, max_new_tokens=64, num_beams=num_beams
            )
            decoded = dec_tokenizer.batch_decode(generated, skip_special_tokens=True)

            label_ids = labels.clone()
            label_ids[label_ids == -100] = dec_tokenizer.pad_token_id
            ref_decoded = dec_tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            preds.extend(decoded)
            refs.extend(ref_decoded)

    avg_loss = total_loss / steps if steps > 0 else 0
    bleu     = compute_bleu(preds, refs) if preds else 0.0
    model.train()
    return avg_loss, bleu


# ──────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ──────────────────────────────────────────────────────────────
def train():
    os.makedirs(CFG["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU  : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Tokenizers (2 tokenizer khác nhau) ─────────────────
    print("\nTải tokenizers...")
    # mmBERT-small dùng Gemma-2 tokenizer (vocab 256k)
    enc_tokenizer = AutoTokenizer.from_pretrained(CFG["encoder_name"])
    # mT5-small dùng SentencePiece tokenizer
    dec_tokenizer = AutoTokenizer.from_pretrained(CFG["decoder_name"])
    print(f"  Encoder tokenizer vocab: {enc_tokenizer.vocab_size:,}")
    print(f"  Decoder tokenizer vocab: {dec_tokenizer.vocab_size:,}")

    # ── Data ───────────────────────────────────────────────
    train_pairs = load_opus_pairs(CFG["src_lang"], CFG["tgt_lang"], "train",      CFG["train_samples"])
    val_pairs   = load_opus_pairs(CFG["src_lang"], CFG["tgt_lang"], "validation", CFG["val_samples"])

    train_ds = TranslationDataset(train_pairs, enc_tokenizer, dec_tokenizer, CFG["max_src_len"], CFG["max_tgt_len"])
    val_ds   = TranslationDataset(val_pairs,   enc_tokenizer, dec_tokenizer, CFG["max_src_len"], CFG["max_tgt_len"])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"] * 2, shuffle=False, num_workers=2, pin_memory=True)
    print(f"\nTrain: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ── Model ──────────────────────────────────────────────
    model = TranslationModel(
        encoder_name=CFG["encoder_name"],
        decoder_name=CFG["decoder_name"],
        freeze_encoder=CFG["freeze_encoder"],
    ).to(device)

    print("\nThông số mô hình:")
    for k, v in model.count_parameters().items():
        print(f"  {k:30s}: {v}")

    # ── Optimizer & Scheduler ──────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    total_steps  = (len(train_loader) // CFG["grad_accum"]) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Mixed Precision ────────────────────────────────────
    use_fp16 = CFG["fp16"] and device.type == "cuda"
    scaler   = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ── Resume checkpoint ──────────────────────────────────
    ckpt_path   = os.path.join(CFG["output_dir"], "checkpoint_latest.pt")
    start_epoch = 0
    best_bleu   = 0.0
    log_history = []

    if os.path.exists(ckpt_path):
        print(f"\nTìm thấy checkpoint, đang load: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_bleu   = ckpt.get("best_bleu", 0.0)
        log_history = ckpt.get("log_history", [])
        print(f"  → Tiếp tục từ epoch {start_epoch}")

    # ── Train ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training: {CFG['epochs']} epochs | {len(train_loader)} steps/epoch")
    print(f"Warmup: {warmup_steps} | Total: {total_steps} optimizer steps")
    print(f"{'='*60}")

    for epoch in range(start_epoch, CFG["epochs"]):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss / CFG["grad_accum"]

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * CFG["grad_accum"]

            if (step + 1) % CFG["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                elapsed = time.time() - t0
                lr_now  = scheduler.get_last_lr()[0]
                print(
                    f"  [E{epoch+1}] step {step+1}/{len(train_loader)} "
                    f"| loss={epoch_loss/(step+1):.4f} "
                    f"| lr={lr_now:.2e} | {elapsed:.0f}s"
                )

        # ── Validation ──────────────────────────────────
        avg_train = epoch_loss / len(train_loader)
        val_loss, bleu = validate(model, val_loader, dec_tokenizer, device)
        elapsed = time.time() - t0

        print(f"\n[Epoch {epoch+1}/{CFG['epochs']}]")
        print(f"  Train Loss : {avg_train:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}  (PPL={math.exp(val_loss):.2f})")
        print(f"  BLEU       : {bleu:.2f}")
        print(f"  Thời gian  : {elapsed/60:.1f} phút\n")

        entry = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train, 4),
            "val_loss":   round(val_loss, 4),
            "ppl":        round(math.exp(val_loss), 2),
            "bleu":       round(bleu, 2),
            "time_min":   round(elapsed / 60, 1),
        }
        log_history.append(entry)
        with open(CFG["log_file"], "w") as f:
            json.dump(log_history, f, indent=2, ensure_ascii=False)

        # ── Save checkpoint ──────────────────────────────
        ckpt_data = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "best_bleu":   best_bleu,
            "log_history": log_history,
            "cfg":         CFG,
        }
        torch.save(ckpt_data, ckpt_path)

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(ckpt_data, os.path.join(CFG["output_dir"], "best_model.pt"))
            print(f"  ✅ Best model saved! BLEU={bleu:.2f}")

    print(f"\n{'='*60}")
    print(f"Training hoàn tất! Best BLEU = {best_bleu:.2f}")
    print(f"Model lưu tại: {CFG['output_dir']}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Bỏ comment dòng dưới khi chạy trên Colab:
    # from google.colab import drive; drive.mount("/content/drive")
    train()