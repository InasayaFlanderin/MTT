"""
test.py – Inference & Đánh giá đa ngôn ngữ (6 chiều dịch)
Hỗ trợ: en↔vi, en↔fr, vi↔fr

Cách dùng language token:
    ">>vi<< Hello"   → dịch sang tiếng Việt
    ">>en<< Bonjour" → dịch sang tiếng Anh
    ">>fr<< Xin chào"→ dịch sang tiếng Pháp
"""

import os, json, torch
from transformers import AutoTokenizer
from datasets import load_dataset
import sacrebleu

from model import TranslationModel

# ──────────────────────────────────────────────────────────────
# 1. CẤU HÌNH
# ──────────────────────────────────────────────────────────────
CFG = {
    "model_path":     "/content/drive/MyDrive/translation_model/best_model.pt",
    "encoder_name":   "jhu-clsp/mmBERT-small",
    "decoder_name":   "google/mt5-small",
    "max_new_tokens": 128,
    "num_beams":      4,
    "length_penalty": 1.0,
    "test_samples":   300,   # mỗi chiều dịch
    "batch_size":     32,
}

# 6 chiều dịch cần đánh giá
EVAL_PAIRS = [
    ("en", "vi"),
    ("vi", "en"),
    ("en", "fr"),
    ("fr", "en"),
    ("vi", "fr"),
    ("fr", "vi"),
]


# ──────────────────────────────────────────────────────────────
# 2. LOAD MODEL
# ──────────────────────────────────────────────────────────────
def load_model(cfg, device):
    print(f"Load model từ: {cfg['model_path']}")
    ckpt  = torch.load(cfg["model_path"], map_location=device)
    model = TranslationModel(
        encoder_name=cfg["encoder_name"],
        decoder_name=cfg["decoder_name"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("  ✅ Load thành công!")
    if "log_history" in ckpt and ckpt["log_history"]:
        best = max(ckpt["log_history"], key=lambda x: x["bleu"])
        print(f"  → Best epoch {best['epoch']} | "
              f"BLEU={best['bleu']} | PPL={best.get('ppl','?')}")
    return model


# ──────────────────────────────────────────────────────────────
# 3. DỊCH MỘT CÂU
# ──────────────────────────────────────────────────────────────
def translate_sentence(text, src_lang, tgt_lang,
                       model, enc_tokenizer, dec_tokenizer, device,
                       num_beams=4, max_new_tokens=128, length_penalty=1.0):
    """
    Dịch một câu từ src_lang sang tgt_lang.
    Language token tự động được ghép vào đầu câu nguồn.
    """
    # Ghép language token: ">>tgt<< câu nguồn"
    src_with_token = TranslationModel.add_lang_token(text, tgt_lang)

    enc = enc_tokenizer(src_with_token, max_length=130,
                        padding="max_length", truncation=True,
                        return_tensors="pt")
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.translate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams, length_penalty=length_penalty,
        )
    return dec_tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────
# 4. ĐÁNH GIÁ BLEU TỪNG CHIỀU DỊCH
# ──────────────────────────────────────────────────────────────
def evaluate_direction(src_lang, tgt_lang,
                       model, enc_tokenizer, dec_tokenizer, device, cfg):
    """Đánh giá BLEU cho một chiều dịch cụ thể."""
    key = f"{src_lang}-{tgt_lang}"
    rev = f"{tgt_lang}-{src_lang}"
    print(f"\n  [{src_lang}→{tgt_lang}] Tải test set...")

    # Thử load theo cả 2 thứ tự
    ds = None
    for pair_key in [key, rev]:
        try:
            ds = load_dataset("Helsinki-NLP/opus-100",
                              pair_key, split="test")
            break
        except Exception:
            continue

    if ds is None:
        print(f"  ⚠️  Không tìm thấy test set cho {key}, bỏ qua")
        return None, None, None

    if cfg["test_samples"]:
        ds = ds.select(range(min(cfg["test_samples"], len(ds))))

    sources, references = [], []
    for item in ds:
        tr = item["translation"]
        s  = tr.get(src_lang, "")
        t  = tr.get(tgt_lang, "")
        if s and t:
            sources.append(s)
            references.append(t)

    predictions = []
    BS = cfg["batch_size"]
    model.eval()

    for i in range(0, len(sources), BS):
        batch_src = sources[i : i + BS]

        # Ghép language token cho cả batch
        batch_with_token = [
            TranslationModel.add_lang_token(s, tgt_lang)
            for s in batch_src
        ]

        enc = enc_tokenizer(batch_with_token, max_length=130,
                            padding="max_length", truncation=True,
                            return_tensors="pt")
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            output_ids = model.translate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=cfg["max_new_tokens"],
                num_beams=cfg["num_beams"],
                length_penalty=cfg["length_penalty"],
            )
        predictions.extend(
            dec_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        )

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    print(f"  [{src_lang}→{tgt_lang}] BLEU = {bleu:.2f}  ({len(sources)} câu)")
    return predictions, references, bleu


def evaluate_all_directions(model, enc_tokenizer, dec_tokenizer, device, cfg):
    """Đánh giá tất cả 6 chiều dịch và in bảng tổng hợp."""
    print("\n" + "="*60)
    print("ĐÁNH GIÁ BLEU – TẤT CẢ CÁC CHIỀU DỊCH")
    print("="*60)

    results = {}
    all_preds_refs = {}

    for src_lang, tgt_lang in EVAL_PAIRS:
        preds, refs, bleu = evaluate_direction(
            src_lang, tgt_lang,
            model, enc_tokenizer, dec_tokenizer, device, cfg
        )
        if bleu is not None:
            results[f"{src_lang}→{tgt_lang}"] = round(bleu, 2)
            all_preds_refs[f"{src_lang}→{tgt_lang}"] = {
                "predictions": preds[:20],
                "references":  refs[:20],
            }

    # Bảng tổng hợp
    print(f"\n{'='*40}")
    print(f"  {'Chiều dịch':15s} | {'BLEU':>6}")
    print(f"  {'-'*25}")
    for direction, bleu in results.items():
        print(f"  {direction:15s} | {bleu:>6.2f}")
    if results:
        avg = sum(results.values()) / len(results)
        print(f"  {'-'*25}")
        print(f"  {'Trung bình':15s} | {avg:>6.2f}")
    print(f"{'='*40}")

    return results, all_preds_refs


# ──────────────────────────────────────────────────────────────
# 5. DEMO NHANH – TẤT CẢ CHIỀU DỊCH
# ──────────────────────────────────────────────────────────────
def demo_all_directions(model, enc_tokenizer, dec_tokenizer, device):
    """Demo dịch một câu qua tất cả các chiều."""

    test_sentences = {
        "en": "Artificial intelligence is transforming the world.",
        "vi": "Trí tuệ nhân tạo đang thay đổi thế giới.",
        "fr": "L'intelligence artificielle transforme le monde.",
    }

    print("\n" + "="*60)
    print("DEMO – 6 CHIỀU DỊCH")
    print("="*60)

    for src_lang, tgt_lang in EVAL_PAIRS:
        src_text = test_sentences[src_lang]
        result   = translate_sentence(
            src_text, src_lang, tgt_lang,
            model, enc_tokenizer, dec_tokenizer, device,
        )
        lang_names = {"en": "English", "vi": "Tiếng Việt", "fr": "Français"}
        print(f"\n  {lang_names[src_lang]} → {lang_names[tgt_lang]}:")
        print(f"    IN : {src_text}")
        print(f"    OUT: {result}")


# ──────────────────────────────────────────────────────────────
# 6. INTERACTIVE MODE
# ──────────────────────────────────────────────────────────────
def interactive_translate(model, enc_tokenizer, dec_tokenizer, device):
    LANG_NAMES = {"en": "English", "vi": "Tiếng Việt", "fr": "Français"}
    LANGS      = list(LANG_NAMES.keys())

    print("\n" + "="*60)
    print("DỊCH TƯƠNG TÁC ĐA NGÔN NGỮ")
    print("="*60)
    print("Gõ 'quit' để thoát\n")

    while True:
        # Chọn ngôn ngữ nguồn
        print("Ngôn ngữ nguồn:", " | ".join(
            f"{i+1}.{l} ({LANG_NAMES[l]})" for i, l in enumerate(LANGS)
        ))
        src_choice = input("Chọn (1/2/3): ").strip()
        if src_choice.lower() == "quit": break
        try:
            src_lang = LANGS[int(src_choice) - 1]
        except Exception:
            print("Lựa chọn không hợp lệ"); continue

        # Chọn ngôn ngữ đích
        tgt_options = [l for l in LANGS if l != src_lang]
        print("Ngôn ngữ đích:", " | ".join(
            f"{i+1}.{l} ({LANG_NAMES[l]})" for i, l in enumerate(tgt_options)
        ))
        tgt_choice = input("Chọn (1/2): ").strip()
        if tgt_choice.lower() == "quit": break
        try:
            tgt_lang = tgt_options[int(tgt_choice) - 1]
        except Exception:
            print("Lựa chọn không hợp lệ"); continue

        # Nhập câu
        text = input(f"\n[{LANG_NAMES[src_lang]}] > ").strip()
        if not text: continue
        if text.lower() == "quit": break

        result = translate_sentence(
            text, src_lang, tgt_lang,
            model, enc_tokenizer, dec_tokenizer, device,
        )
        print(f"[{LANG_NAMES[tgt_lang]}] > {result}\n")


# ──────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # from google.colab import drive; drive.mount("/content/drive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    enc_tokenizer = AutoTokenizer.from_pretrained(CFG["encoder_name"])
    dec_tokenizer = AutoTokenizer.from_pretrained(CFG["decoder_name"])
    model         = load_model(CFG, device)

    # ── A. Demo 6 chiều dịch ─────────────────────────────
    demo_all_directions(model, enc_tokenizer, dec_tokenizer, device)

    # ── B. Đánh giá BLEU tất cả chiều ────────────────────
    results, samples = evaluate_all_directions(
        model, enc_tokenizer, dec_tokenizer, device, CFG
    )

    # Lưu kết quả
    out_path = os.path.join(
        os.path.dirname(CFG["model_path"]), "test_results.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "bleu_per_direction": results,
            "avg_bleu": round(sum(results.values())/len(results), 2) if results else 0,
            "samples":  samples,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nKết quả lưu: {out_path}")

    # ── C. Interactive (bỏ comment để dùng) ──────────────
    # interactive_translate(model, enc_tokenizer, dec_tokenizer, device)