"""
train.py – Training đa ngôn ngữ (6 chiều dịch)
Hỗ trợ: en↔vi, en↔fr, vi↔fr

Cơ chế: Language Token ghép vào đầu câu nguồn
    ">>vi<< Hello"  →  model biết dịch sang tiếng Việt
    ">>en<< Bonjour" →  model biết dịch sang tiếng Anh

Dữ liệu: opus-100
    en-vi : ~1M cặp  → lấy train_samples_per_pair
    en-fr : ~1M cặp
    fr-vi : ít hơn   → zero-shot hoặc pivot qua en

Cài đặt:
    !pip install transformers datasets sentencepiece sacrebleu accelerate -q
"""

import os, math, time, json, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import sacrebleu

from model import TranslationModel

# ──────────────────────────────────────────────────────────────
# 1. HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────
CFG = {
    "encoder_name":   "jhu-clsp/mmBERT-small",
    "decoder_name":   "google/mt5-small",
    "freeze_encoder": False,

    # ── Các cặp ngôn ngữ cần train ──────────────────────────
    # Mỗi cặp (src, tgt) sẽ tự động tạo cả 2 chiều A→B và B→A
    # opus-100 có sẵn: en-vi, en-fr
    # fr-vi không có trực tiếp → dùng pivot (en làm cầu nối)
    "lang_pairs": [
        ("en", "vi"),   # → sinh en→vi và vi→en
        ("en", "fr"),   # → sinh en→fr và fr→en
        ("fr", "vi"), # uncomment nếu muốn thêm fr↔vi trực tiếp
    ],

    # Số mẫu mỗi chiều dịch (None = toàn bộ)
    "train_samples_per_pair": 30_000,   # × 4 chiều = 120k total
    "val_samples_per_pair":   500,

    "max_src_len":    128,
    "max_tgt_len":    128,

    "batch_size":     8,
    "grad_accum":     16,       # Effective batch = 128
    "epochs":         5,
    "lr":             5e-5,
    "warmup_ratio":   0.1,
    "weight_decay":   0.01,
    "max_grad_norm":  1.0,

    "save_every_minutes": 10,
    "output_dir": "/content/drive/MyDrive/translation_model",
    "log_file":   "/content/drive/MyDrive/translation_model/train_log.json",
}

torch.manual_seed(42)
random.seed(42)


# ──────────────────────────────────────────────────────────────
# 2. DATASET
# ──────────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    """
    Mỗi sample: câu nguồn có language token + câu đích.
    Ví dụ:
        input : ">>vi<< Hello world"      (en→vi)
        label : "Xin chào thế giới"
    """

    def __init__(self, pairs, enc_tokenizer, dec_tokenizer,
                 max_src_len=128, max_tgt_len=128):
        self.pairs       = pairs          # [{"src": ..., "tgt": ...}, ...]
        self.enc_tok     = enc_tokenizer
        self.dec_tok     = dec_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # src đã chứa language token: ">>vi<< ..."
        src = self.pairs[idx]["src"]
        tgt = self.pairs[idx]["tgt"]

        enc = self.enc_tok(src, max_length=self.max_src_len,
                           padding="max_length", truncation=True,
                           return_tensors="pt")
        dec = self.dec_tok(tgt, max_length=self.max_tgt_len,
                           padding="max_length", truncation=True,
                           return_tensors="pt")

        labels = dec["input_ids"].squeeze(0).clone()
        labels[labels == self.dec_tok.pad_token_id] = -100

        if (labels != -100).sum() == 0:
            labels[0] = dec["input_ids"].squeeze(0)[0]

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


def load_pairs_bidirectional(lang_a, lang_b, split, max_samples=None):
    """
    Tải opus-100 cho cặp (lang_a, lang_b) và tạo dữ liệu 2 chiều:
        A→B: src = ">>B<< câu_A",  tgt = câu_B
        B→A: src = ">>A<< câu_B",  tgt = câu_A
    """
    key = f"{lang_a}-{lang_b}"
    print(f"  Tải opus-100 [{key}] split={split}...")

    try:
        ds = load_dataset("Helsinki-NLP/opus-100", key, split=split)
    except Exception:
        try:
            ds = load_dataset("Helsinki-NLP/opus-100",
                              f"{lang_b}-{lang_a}", split=split)
        except Exception:
            print(f"  ⚠️  Không tìm thấy {key}, bỏ qua")
            return [], []

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    pairs_ab, pairs_ba = [], []
    lang_token_b = TranslationModel.LANG_TOKENS.get(lang_b, f">>{lang_b}<<")
    lang_token_a = TranslationModel.LANG_TOKENS.get(lang_a, f">>{lang_a}<<")

    for item in ds:
        tr   = item["translation"]
        text_a = tr.get(lang_a) or ""
        text_b = tr.get(lang_b) or ""
        if not text_a or not text_b:
            continue

        # Chiều A → B
        pairs_ab.append({
            "src": f"{lang_token_b} {text_a}",
            "tgt": text_b,
            "direction": f"{lang_a}→{lang_b}",
        })
        # Chiều B → A
        pairs_ba.append({
            "src": f"{lang_token_a} {text_b}",
            "tgt": text_a,
            "direction": f"{lang_b}→{lang_a}",
        })

    print(f"    → {lang_a}→{lang_b}: {len(pairs_ab):,} | "
          f"{lang_b}→{lang_a}: {len(pairs_ba):,}")
    return pairs_ab, pairs_ba


def build_all_pairs(lang_pairs, split, max_per_pair=None):
    """Tổng hợp tất cả cặp ngôn ngữ thành một list."""
    all_pairs = []
    stats = {}

    for lang_a, lang_b in lang_pairs:
        pairs_ab, pairs_ba = load_pairs_bidirectional(
            lang_a, lang_b, split, max_per_pair
        )
        all_pairs.extend(pairs_ab)
        all_pairs.extend(pairs_ba)
        stats[f"{lang_a}→{lang_b}"] = len(pairs_ab)
        stats[f"{lang_b}→{lang_a}"] = len(pairs_ba)

    # Shuffle để trộn đều các cặp
    random.shuffle(all_pairs)

    print(f"\n  Thống kê [{split}]:")
    for direction, count in stats.items():
        print(f"    {direction}: {count:,}")
    print(f"  Tổng: {len(all_pairs):,} cặp câu")
    return all_pairs


# ──────────────────────────────────────────────────────────────
# 3. METRICS
# ──────────────────────────────────────────────────────────────
def compute_bleu(predictions, references):
    return sacrebleu.corpus_bleu(predictions, [references]).score


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

        with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)

        total_loss += out.loss.item()
        steps += 1

        if len(preds) < 200:
            generated = model.translate(
                input_ids, attention_mask,
                max_new_tokens=64, num_beams=num_beams
            )
            decoded   = dec_tokenizer.batch_decode(generated,
                                                    skip_special_tokens=True)
            lbl_ids   = labels.clone()
            lbl_ids[lbl_ids == -100] = dec_tokenizer.pad_token_id
            ref_dec   = dec_tokenizer.batch_decode(lbl_ids,
                                                    skip_special_tokens=True)
            preds.extend(decoded)
            refs.extend(ref_dec)

    avg_loss = total_loss / steps if steps > 0 else 0.0
    bleu     = compute_bleu(preds, refs) if preds else 0.0
    model.train()
    return avg_loss, bleu


# ──────────────────────────────────────────────────────────────
# 5. SAVE CHECKPOINT
# ──────────────────────────────────────────────────────────────
def save_ckpt(path, epoch, step, epoch_loss_accum,
              model, optimizer, scheduler, best_bleu, log_history):
    torch.save({
        "epoch":            epoch,
        "step":             step,
        "epoch_loss_accum": epoch_loss_accum,
        "model":            model.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "scheduler":        scheduler.state_dict(),
        "best_bleu":        best_bleu,
        "log_history":      log_history,
        "cfg":              CFG,
    }, path)


# ──────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ──────────────────────────────────────────────────────────────
def train():
    os.makedirs(CFG["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU  : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Tokenizers ─────────────────────────────────────────
    print("\nTải tokenizers...")
    enc_tokenizer = AutoTokenizer.from_pretrained(CFG["encoder_name"])
    dec_tokenizer = AutoTokenizer.from_pretrained(CFG["decoder_name"])
    print(f"  Encoder vocab: {enc_tokenizer.vocab_size:,}  (Gemma-2)")
    print(f"  Decoder vocab: {dec_tokenizer.vocab_size:,}  (SentencePiece)")

    # ── Data: tất cả cặp ngôn ngữ ──────────────────────────
    print("\nTải dữ liệu đa ngôn ngữ...")
    print(f"Cặp ngôn ngữ: {CFG['lang_pairs']}")

    train_pairs = build_all_pairs(
        CFG["lang_pairs"], "train", CFG["train_samples_per_pair"]
    )
    val_pairs = build_all_pairs(
        CFG["lang_pairs"], "validation", CFG["val_samples_per_pair"]
    )

    train_ds = TranslationDataset(train_pairs, enc_tokenizer, dec_tokenizer,
                                   CFG["max_src_len"], CFG["max_tgt_len"])
    val_ds   = TranslationDataset(val_pairs,   enc_tokenizer, dec_tokenizer,
                                   CFG["max_src_len"], CFG["max_tgt_len"])

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=2, pin_memory=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"] * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )
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
    optimizer    = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG["lr"], weight_decay=CFG["weight_decay"],
    )
    total_steps  = (len(train_loader) // CFG["grad_accum"]) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # ── Resume checkpoint ──────────────────────────────────
    ckpt_path        = os.path.join(CFG["output_dir"], "checkpoint_latest.pt")
    start_epoch      = 0
    start_step       = 0
    best_bleu        = 0.0
    log_history      = []
    epoch_loss_accum = 0.0

    if os.path.exists(ckpt_path):
        print(f"\nTìm thấy checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch      = ckpt["epoch"]
        start_step       = ckpt.get("step", -1) + 1
        best_bleu        = ckpt.get("best_bleu", 0.0)
        log_history      = ckpt.get("log_history", [])
        epoch_loss_accum = ckpt.get("epoch_loss_accum", 0.0)
        print(f"  → Tiếp tục từ epoch {start_epoch + 1}, step {start_step}")
    else:
        print("  → Không tìm thấy checkpoint, train từ đầu")

    # ── Train ──────────────────────────────────────────────
    save_interval  = CFG["save_every_minutes"] * 60
    last_save_time = time.time()

    print(f"\n{'='*60}")
    print(f"Training đa ngôn ngữ ({len(CFG['lang_pairs'])*2} chiều dịch)")
    print(f"Epochs  : {CFG['epochs']} | Steps/epoch: {len(train_loader)}")
    print(f"Warmup  : {warmup_steps} → Total: {total_steps} steps")
    print(f"Save    : mỗi {CFG['save_every_minutes']} phút")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, CFG["epochs"]):
        model.train()
        optimizer.zero_grad()
        epoch_loss = epoch_loss_accum
        t0 = time.time()

        for step, batch in enumerate(train_loader):

            # Skip các step đã train khi resume
            if epoch == start_epoch and step < start_step:
                if step > 0 and step % 500 == 0:
                    print(f"  ⏩ Skip {step}/{start_step - 1}...")
                continue

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
                out  = model(input_ids=input_ids,
                             attention_mask=attention_mask, labels=labels)
                loss = out.loss / CFG["grad_accum"]

            if torch.isnan(loss):
                print(f"  ⚠️  NaN tại E{epoch+1} step {step+1}, bỏ qua")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * CFG["grad_accum"]

            if (step + 1) % CFG["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),
                                         CFG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Auto-save mỗi 10 phút
                if time.time() - last_save_time >= save_interval:
                    save_ckpt(
                        ckpt_path, epoch, step, epoch_loss,
                        model, optimizer, scheduler, best_bleu, log_history
                    )
                    elapsed_min = (time.time() - t0) / 60
                    print(f"  💾 Saved | E{epoch+1} step {step+1}/"
                          f"{len(train_loader)} | {elapsed_min:.1f} phút")
                    last_save_time = time.time()

            if (step + 1) % 100 == 0:
                steps_done = (step - start_step + 1
                              if epoch == start_epoch else step + 1)
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  [E{epoch+1}] {step+1}/{len(train_loader)} "
                    f"| loss={epoch_loss/max(steps_done,1):.4f} "
                    f"| lr={lr_now:.2e} | {time.time()-t0:.0f}s"
                )

        # ── Validation ─────────────────────────────────────
        avg_train      = epoch_loss / len(train_loader)
        val_loss, bleu = validate(model, val_loader, dec_tokenizer, device)
        elapsed        = time.time() - t0

        print(f"\n{'─'*50}")
        print(f"[Epoch {epoch+1}/{CFG['epochs']}]")
        print(f"  Train Loss : {avg_train:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}  (PPL={math.exp(val_loss):.2f})")
        print(f"  BLEU       : {bleu:.2f}  (avg tất cả chiều dịch)")
        print(f"  Thời gian  : {elapsed/60:.1f} phút")
        print(f"{'─'*50}\n")

        log_history.append({
            "epoch":      epoch + 1,
            "train_loss": round(avg_train, 4),
            "val_loss":   round(val_loss, 4),
            "ppl":        round(math.exp(val_loss), 2),
            "bleu":       round(bleu, 2),
            "time_min":   round(elapsed / 60, 1),
        })
        with open(CFG["log_file"], "w") as f:
            json.dump(log_history, f, indent=2, ensure_ascii=False)

        # Save cuối epoch
        save_ckpt(ckpt_path, epoch + 1, -1, 0.0,
                  model, optimizer, scheduler, best_bleu, log_history)
        last_save_time = time.time()
        start_step       = 0
        epoch_loss_accum = 0.0

        if bleu > best_bleu:
            best_bleu = bleu
            save_ckpt(
                os.path.join(CFG["output_dir"], "best_model.pt"),
                epoch + 1, -1, 0.0,
                model, optimizer, scheduler, best_bleu, log_history
            )
            print(f"  ✅ Best model saved! BLEU={bleu:.2f}")

    print(f"\n{'='*60}")
    print(f"Training hoàn tất! Best BLEU = {best_bleu:.2f}")
    print(f"Model lưu tại: {CFG['output_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # from google.colab import drive; drive.mount("/content/drive")
    train()