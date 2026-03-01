"""
main.py  —  dịch câu bằng tay với MTT
  python main.py                                        # interactive
  python main.py --text "Hello, how are you?"
  python main.py --text "..." --tgt vi
  python main.py --text "..." --tgt de --beams 2
  python main.py --ckpt ./mtt_checkpoint.pt
"""
import argparse
import os
import torch
from model import MTT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           default="mtt_checkpoint.pt")
    parser.add_argument("--text",           default=None,   help="Câu cần dịch")
    parser.add_argument("--tgt",            default="vi",   help="Ngôn ngữ đích: de fr vi en es")
    parser.add_argument("--beams",          type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device",         default="auto")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ── Load model ────────────────────────────────────────────────────────────
    if os.path.exists(args.ckpt):
        model = MTT.load(args.ckpt, device=device)
    else:
        print(f"[WARN] Không tìm thấy '{args.ckpt}' — dùng random weights")
        model = MTT()
        model.to(device)
        model.eval()

    model.paramsCalc()
    print(f"Device : {device}")
    print(f"Target : {args.tgt}")
    print(f"Beams  : {args.beams}\n")

    # ── Helper ────────────────────────────────────────────────────────────────
    def translate(text: str) -> str:
        results = model.translate(
            srcText=[text],
            targetLang=args.tgt,
            maxNewTokens=args.max_new_tokens,
            numBeams=args.beams,
            device=device,
        )
        return results[0]

    # ── Single text mode ──────────────────────────────────────────────────────
    if args.text:
        output = translate(args.text)
        print(f"Input : {args.text}")
        print(f"Output: {output}")
        return

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("Interactive mode — Ctrl+C hoặc 'q' để thoát")
    print(f"(Đang dịch sang '{args.tgt}', đổi bằng --tgt)\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            print("Bye!")
            break

        output = translate(text)
        print(f"    → {output}\n")


if __name__ == "__main__":
    main()