"""
MTT Training Script
────────────────────────────────────────────────────────────────
One cycle = 800 train steps + 200 test steps = 1000 total
After each cycle: LR auto-adjusted based on avg test loss
Runs forever — press Ctrl+C to stop safely
Checkpoint saved every 30 minutes + at end of every cycle
All language pairs fetched automatically from OPUS-100

Install:
    pip install torch datasets sentencepiece transformers
"""

import os
import random
import signal
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import MTT


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    languages            = ["de", "fr", "vi", "es"],   # paired with "en" + each other
    max_samples_per_pair = 50_000,

    train_steps          = 800,    # train steps per cycle
    test_steps           = 200,    # test  steps per cycle (runs right after train)

    batch_size           = 8,
    lr                   = 5e-5,
    grad_clip            = 1.0,

    # ReduceLROnPlateau — fires once per cycle using avg test loss
    lr_factor            = 0.5,   # new_lr = old_lr * factor  on plateau
    lr_patience          = 2,     # cycles with no improvement before reducing
    lr_min               = 1e-7,  # hard floor for LR

    checkpoint_path      = "mtt_checkpoint.pt",
    checkpoint_minutes   = 30.0,

    device               = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers          = max(4, os.cpu_count() or 1),
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_pair(src: str, tgt: str, split: str, max_n: int) -> list:
    """Download one language pair from OPUS-100. Returns [(src_text, tgt_text, tgt_lang)]."""
    from datasets import load_dataset  # lazy — only imported once at startup

    for key in [f"{src}-{tgt}", f"{tgt}-{src}"]:
        try:
            ds  = load_dataset("Helsinki-NLP/opus-100", key,
                               split=split, trust_remote_code=True)
            out = []
            for row in ds:
                t = row["translation"]
                if src in t and tgt in t:
                    out.append((t[src], t[tgt], tgt))
                if len(out) >= max_n:
                    break
            print(f"  [DATA]  {src}→{tgt:<4}  {split:<12}  {len(out):>7,}  (key={key})")
            return out
        except Exception:
            continue

    print(f"  [DATA]  {src}→{tgt}  not found in OPUS-100, skipping.")
    return []


def fetch_all_pairs() -> tuple:
    """
    Build every unique language pair from {en, de, fr, vi, es}.
    Pairs missing from OPUS-100 are created via English pivot.
    Returns (train_samples, test_samples).
    """
    max_n     = CFG["max_samples_per_pair"]
    all_langs = CFG["languages"] + ["en"]

    # Every unique pair (order doesn't matter for the set, but we fetch both directions)
    unique_pairs = [(all_langs[i], all_langs[j])
                    for i in range(len(all_langs))
                    for j in range(i + 1, len(all_langs))]

    train_all, test_all = [], []

    for src, tgt in unique_pairs:
        rows = _fetch_pair(src, tgt, "train",      max_n)
        tst  = _fetch_pair(src, tgt, "validation", max_n // 5)

        if rows:
            train_all += rows
            test_all  += tst
        elif src != "en" and tgt != "en":
            # Pivot: src→en + en→tgt  →  src→tgt
            print(f"  [DATA]  Pivoting {src}→{tgt} through English...")
            en_src    = _fetch_pair("en", src, "train", max_n // 2)
            en_tgt    = _fetch_pair("en", tgt, "train", max_n // 2)
            en_to_src = {e: s for e, s, _ in en_src}
            pivoted   = [(en_to_src[e], t, tgt)
                         for e, t, _ in en_tgt if e in en_to_src][:max_n]
            train_all += pivoted
            print(f"  [DATA]  Pivoted  {src}→{tgt}  {len(pivoted):>7,}")

    random.shuffle(train_all)
    random.shuffle(test_all)
    print(f"\n  [DATA]  Train total : {len(train_all):,}")
    print(f"  [DATA]  Test  total : {len(test_all):,}\n")
    return train_all, test_all


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """Each sample: (src_text, tgt_text, tgt_lang, ner_tensor | None)"""

    def __init__(self, samples: list):
        self.samples = [s if len(s) == 4 else (*s, None) for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    srcs, tgts, langs, ners = zip(*batch)
    if ners[0] is not None:
        max_len = max(t.size(0) for t in ners)
        padded  = torch.full((len(ners), max_len), -100, dtype=torch.long)
        for i, t in enumerate(ners):
            padded[i, : t.size(0)] = t
        return list(srcs), list(tgts), list(langs), padded
    return list(srcs), list(tgts), list(langs), None


def infinite_loader(loader: DataLoader):
    """Yields batches forever, reshuffling each epoch."""
    while True:
        yield from loader


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scheduler, global_step: int, cycle: int):
    torch.save({
        "global_step": global_step,
        "cycle":       cycle,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "lr":          optimizer.param_groups[0]["lr"],
    }, CFG["checkpoint_path"])
    print(f"  [CKPT]  Saved → {CFG['checkpoint_path']}  "
          f"(global step {global_step}, cycle {cycle})")


def load_checkpoint(model, optimizer, scheduler, device: str) -> tuple:
    path = CFG["checkpoint_path"]
    if not os.path.exists(path):
        return 0, 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    for pg in optimizer.param_groups:
        pg["lr"] = ckpt["lr"]
    print(f"  [CKPT]  Resumed — global step {ckpt['global_step']}  "
          f"cycle {ckpt['cycle']}  lr={ckpt['lr']:.2e}")
    return ckpt["global_step"], ckpt["cycle"]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(model: MTT,
          train_dataset: TranslationDataset,
          test_dataset: TranslationDataset):

    device   = CFG["device"]
    ckpt_sec = CFG["checkpoint_minutes"] * 60

    model.to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG["lr"], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = CFG["lr_factor"],
        patience = CFG["lr_patience"],
        min_lr   = CFG["lr_min"],
    )

    global_step, cycle = load_checkpoint(model, optimizer, scheduler, device)

    use_pin = (device == "cuda")
    loader_kwargs = dict(
        batch_size         = CFG["batch_size"],
        collate_fn         = collate_fn,
        num_workers        = CFG["num_workers"],
        pin_memory         = use_pin,
        persistent_workers = CFG["num_workers"] > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)
    train_iter   = infinite_loader(train_loader)

    # Graceful Ctrl+C
    stop = False
    def _on_stop(sig, frame):
        nonlocal stop
        print("\n\n  [STOP]  Ctrl+C — finishing this step then saving...\n")
        stop = True
    signal.signal(signal.SIGINT, _on_stop)

    last_ckpt = time.time()

    print(f"\n{'═'*64}")
    print(f"  MTT Training  |  device={device}  |  lr={CFG['lr']:.2e}")
    print(f"  Cycle  =  800 train steps  +  200 test steps")
    print(f"  LR auto-adjusted via ReduceLROnPlateau after every cycle")
    print(f"  Checkpoint every {CFG['checkpoint_minutes']} min + end of each cycle")
    print(f"  Press Ctrl+C to stop safely at any time")
    print(f"{'═'*64}\n")

    # ── Infinite loop ─────────────────────────────────────────────────────────
    while not stop:
        cycle += 1

        print(f"\n{'─'*64}")
        print(f"  CYCLE {cycle}  |  global step: {global_step}")
        print(f"{'─'*64}\n")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1 — TRAIN  (800 steps)
        # ══════════════════════════════════════════════════════════════════════
        model.train()
        train_loss_sum = 0.0

        for train_step in range(1, CFG["train_steps"] + 1):
            if stop:
                break

            srcs, tgts, langs, ners = next(train_iter)

            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

            out = model(
                srcText    = srcs,
                targetLang = langs[0],
                targetText = tgts,
                nerTags    = ners,
                returnLoss = True,
                device     = device,
            )

            loss = out["loss"]

            # ── Backpropagation + gradient descent ────────────────────────────
            loss.backward()                                          # compute all gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(),      # clip to prevent explosion
                                           CFG["grad_clip"])
            optimizer.step()                                         # update every weight
            # ──────────────────────────────────────────────────────────────────

            global_step    += 1
            train_loss_sum += loss.item()

            trans = out["translationLoss"]
            ner   = out["nerLoss"]
            print(
                f"  [TRAIN]"
                f"  global={global_step:>7}"
                f"  cycle={cycle}  train_step={train_step:>3}/{CFG['train_steps']}"
                f"  lang={langs[0]}"
                f"  loss={loss.item():.4f}"
                + (f"  trans={trans.item():.4f}" if trans is not None else "")
                + (f"  ner={ner.item():.4f}"     if ner   is not None else "")
                + f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            # Timed checkpoint (every 30 min, mid-cycle)
            if time.time() - last_ckpt >= ckpt_sec:
                save_checkpoint(model, optimizer, scheduler, global_step, cycle)
                last_ckpt = time.time()

        avg_train = train_loss_sum / max(train_step, 1)
        print(f"\n  [CYCLE {cycle}]  ── Train done ──  avg loss: {avg_train:.4f}\n")

        if stop:
            break

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2 — TEST  (200 steps, no gradients)
        # ══════════════════════════════════════════════════════════════════════
        model.eval()
        test_loss_sum  = 0.0
        test_trans_sum = 0.0
        test_ner_sum   = 0.0

        with torch.no_grad():
            for test_step, (srcs, tgts, langs, ners) in enumerate(test_loader, 1):
                if test_step > CFG["test_steps"]:
                    break

                out = model(
                    srcText    = srcs,
                    targetLang = langs[0],
                    targetText = tgts,
                    nerTags    = ners,
                    returnLoss = True,
                    device     = device,
                )

                lv = out["loss"].item()
                test_loss_sum += lv
                if out["translationLoss"] is not None:
                    test_trans_sum += out["translationLoss"].item()
                if out["nerLoss"] is not None:
                    test_ner_sum += out["nerLoss"].item()

                print(
                    f"  [TEST ]"
                    f"  global={global_step:>7}"
                    f"  cycle={cycle}  test_step={test_step:>3}/{CFG['test_steps']}"
                    f"  lang={langs[0]}"
                    f"  loss={lv:.4f}"
                )

        n         = CFG["test_steps"]
        avg_test  = test_loss_sum  / n
        avg_trans = test_trans_sum / n
        avg_ner   = test_ner_sum   / n

        print(f"\n  [CYCLE {cycle}]  ── Test  done ──  "
              f"avg loss: {avg_test:.4f}  trans: {avg_trans:.4f}  ner: {avg_ner:.4f}")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 3 — AUTO ADJUST LR
        # ══════════════════════════════════════════════════════════════════════
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_test)                    # reduce if no improvement
        new_lr  = optimizer.param_groups[0]["lr"]

        if new_lr < prev_lr:
            print(f"  [LR  ]  Reduced  {prev_lr:.2e}  →  {new_lr:.2e}  (plateau)")
        else:
            print(f"  [LR  ]  Holding  {new_lr:.2e}  (still improving)")

        # End-of-cycle checkpoint
        print()
        save_checkpoint(model, optimizer, scheduler, global_step, cycle)
        last_ckpt = time.time()

    # ── Final save on Ctrl+C ──────────────────────────────────────────────────
    save_checkpoint(model, optimizer, scheduler, global_step, cycle)
    print(f"\n  Stopped at global step {global_step}, cycle {cycle}. Goodbye!")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("  Fetching data from HuggingFace OPUS-100...\n")
    train_samples, test_samples = fetch_all_pairs()

    train_ds = TranslationDataset(train_samples)
    test_ds  = TranslationDataset(test_samples)

    model = MTT()
    model.paramsCalc()

    train(model, train_ds, test_ds)
