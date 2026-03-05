"""
test.py – Inference & Đánh giá mô hình dịch máy
Chạy sau khi train.py hoàn tất.

Cài đặt:
    !pip install transformers sentencepiece sacrebleu -q
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
    "src_lang":       "en",
    "tgt_lang":       "vi",
    "max_new_tokens": 128,
    "num_beams":      4,       # 1 = greedy, >1 = beam search
    "length_penalty": 1.0,
    "test_samples":   500,     # None = toàn bộ test set
    "batch_size":     32,
}


# ──────────────────────────────────────────────────────────────
# 2. LOAD MODEL
# ──────────────────────────────────────────────────────────────
def load_model(cfg: dict, device: torch.device) -> TranslationModel:
    print(f"Load model từ: {cfg['model_path']}")
    ckpt = torch.load(cfg["model_path"], map_location=device)

    model = TranslationModel(
        encoder_name=cfg["encoder_name"],
        decoder_name=cfg["decoder_name"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print("  ✅ Load thành công!")

    if "log_history" in ckpt:
        best = max(ckpt["log_history"], key=lambda x: x["bleu"])
        print(f"  → Best epoch: {best['epoch']} | BLEU={best['bleu']} | PPL={best.get('ppl','?')}")

    return model


# ──────────────────────────────────────────────────────────────
# 3. DỊCH MỘT CÂU
# ──────────────────────────────────────────────────────────────
def translate_sentence(
    text: str,
    model: TranslationModel,
    enc_tokenizer,
    dec_tokenizer,
    device: torch.device,
    num_beams: int = 4,
    max_new_tokens: int = 128,
    length_penalty: float = 1.0,
) -> str:
    """Dịch một câu, trả về chuỗi kết quả."""
    # Gemma-2 tokenizer (mmBERT)
    enc = enc_tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.translate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

    # SentencePiece decode (mT5)
    return dec_tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────
# 4. DỊCH BATCH + TÍNH BLEU
# ──────────────────────────────────────────────────────────────
def evaluate_test_set(model, enc_tokenizer, dec_tokenizer, device, cfg):
    print(f"\nTải test set opus-100 {cfg['src_lang']}-{cfg['tgt_lang']}...")
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", f"{cfg['src_lang']}-{cfg['tgt_lang']}", split="test")
    except Exception:
        ds = load_dataset("Helsinki-NLP/opus-100", f"{cfg['tgt_lang']}-{cfg['src_lang']}", split="test")

    if cfg["test_samples"]:
        ds = ds.select(range(min(cfg["test_samples"], len(ds))))

    sources, references = [], []
    for item in ds:
        tr = item["translation"]
        sources.append(tr.get(cfg["src_lang"], ""))
        references.append(tr.get(cfg["tgt_lang"], ""))

    print(f"  → {len(sources)} câu test")

    predictions = []
    BS = cfg["batch_size"]
    model.eval()

    for i in range(0, len(sources), BS):
        batch_src = sources[i : i + BS]

        # Tokenize bằng Gemma-2 tokenizer (mmBERT)
        enc = enc_tokenizer(
            batch_src,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            output_ids = model.translate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg["max_new_tokens"],
                num_beams=cfg["num_beams"],
                length_penalty=cfg["length_penalty"],
            )

        decoded = dec_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(decoded)

        if (i // BS + 1) % 10 == 0:
            print(f"  Đã dịch {i + len(batch_src)}/{len(sources)}...")

    # BLEU score
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"\n{'='*50}")
    print(f"BLEU Score      : {bleu.score:.2f}")
    print(f"Brevity Penalty : {bleu.bp:.4f}")
    print(f"{'='*50}")

    return predictions, references, bleu.score


# ──────────────────────────────────────────────────────────────
# 5. SO SÁNH DECODING STRATEGIES
# ──────────────────────────────────────────────────────────────
def compare_decoding(sentences, model, enc_tokenizer, dec_tokenizer, device):
    print("\n" + "="*60)
    print("SO SÁNH CHIẾN LƯỢC DECODING")
    print("="*60)

    strategies = [
        ("Greedy Search   (beam=1)", 1),
        ("Beam Search     (beam=4)", 4),
        ("Beam Search     (beam=8)", 8),
    ]

    for text in sentences:
        print(f"\n📝 {text}")
        for name, beams in strategies:
            pred = translate_sentence(
                text, model, enc_tokenizer, dec_tokenizer, device,
                num_beams=beams
            )
            print(f"  [{name}]: {pred}")


# ──────────────────────────────────────────────────────────────
# 6. INTERACTIVE MODE
# ──────────────────────────────────────────────────────────────
def interactive_translate(model, enc_tokenizer, dec_tokenizer, device, cfg):
    print("\n" + "="*60)
    print(f"DỊCH TƯƠNG TÁC  ({cfg['src_lang'].upper()} → {cfg['tgt_lang'].upper()})")
    print("Lệnh: 'quit' thoát | 'greedy' Greedy | 'beam' Beam Search")
    print("="*60)

    beams = cfg["num_beams"]
    while True:
        text = input(f"\n[{cfg['src_lang'].upper()}] > ").strip()
        if not text:
            continue
        if text.lower() == "quit":
            break
        if text.lower() == "greedy":
            beams = 1
            print("  → Greedy Search")
            continue
        if text.lower() == "beam":
            beams = cfg["num_beams"]
            print(f"  → Beam Search (beams={beams})")
            continue

        result = translate_sentence(
            text, model, enc_tokenizer, dec_tokenizer, device, num_beams=beams
        )
        mode = "Greedy" if beams == 1 else f"Beam({beams})"
        print(f"[{cfg['tgt_lang'].upper()} – {mode}]: {result}")


# ──────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # from google.colab import drive; drive.mount("/content/drive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizers
    print("Tải tokenizers...")
    enc_tokenizer = AutoTokenizer.from_pretrained(CFG["encoder_name"])  # Gemma-2
    dec_tokenizer = AutoTokenizer.from_pretrained(CFG["decoder_name"])  # SentencePiece

    # Load model
    model = load_model(CFG, device)

    # ── A. Ví dụ dịch nhanh ──────────────────────────────
    examples = [
        "Hello, how are you today?",
        "Artificial intelligence is transforming the world.",
        "I love learning new languages every day.",
        "The weather in Hanoi is very hot in summer.",
        "Can you help me with this problem?",
    ]

    print("\n" + "="*60)
    print("VÍ DỤ DỊCH NHANH")
    print("="*60)
    for sent in examples:
        result = translate_sentence(sent, model, enc_tokenizer, dec_tokenizer, device)
        print(f"  EN: {sent}")
        print(f"  VI: {result}\n")

    # ── B. So sánh Greedy vs Beam Search ─────────────────
    compare_decoding(examples[:3], model, enc_tokenizer, dec_tokenizer, device)

    # ── C. Đánh giá BLEU trên test set ───────────────────
    predictions, references, bleu = evaluate_test_set(
        model, enc_tokenizer, dec_tokenizer, device, CFG
    )

    # Lưu kết quả
    out_path = os.path.join(os.path.dirname(CFG["model_path"]), "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "bleu": round(bleu, 2),
                "encoder": CFG["encoder_name"],
                "decoder": CFG["decoder_name"],
                "samples": [
                    {"reference": references[i], "prediction": predictions[i]}
                    for i in range(min(50, len(predictions)))
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nKết quả lưu: {out_path}")

    # ── D. Chế độ tương tác (bỏ comment để dùng) ─────────
    # interactive_translate(model, enc_tokenizer, dec_tokenizer, device, CFG)