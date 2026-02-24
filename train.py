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
from torch.amp import GradScaler, autocast

from model import MTT


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    languages            = ["de", "fr", "vi", "es"],
    max_samples_per_pair = 50_000,

    train_steps          = 800,
    test_steps           = 200,

    batch_size           = 1,
    accum_steps          = 4,
    lr                   = 5e-5,
    grad_clip            = 1.0,

    lr_factor            = 0.5,
    lr_patience          = 2,
    lr_min               = 1e-7,

    checkpoint_path      = "mtt_checkpoint.pt",
    checkpoint_minutes   = 30.0,

    device               = "cpu",
    num_workers          = 2,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_pair(src: str, tgt: str, split: str, max_n: int) -> list:
    from datasets import load_dataset

    for key in [f"{src}-{tgt}", f"{tgt}-{src}"]:
        try:
            ds  = load_dataset("Helsinki-NLP/opus-100", key, split=split)
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
    max_n     = CFG["max_samples_per_pair"]
    all_langs = CFG["languages"] + ["en"]

    unique_pairs = [(all_langs[i], all_langs[j])
                    for i in range(len(all_langs))
                    for j in range(i + 1, len(all_langs))]

    # Gom theo target language để balance sau
    train_by_lang: dict = {}
    test_by_lang:  dict = {}

    for src, tgt in unique_pairs:
        rows = _fetch_pair(src, tgt, "train",      max_n)
        tst  = _fetch_pair(src, tgt, "validation", max_n // 5)

        if rows:
            train_by_lang.setdefault(tgt, []).extend(rows)
            test_by_lang.setdefault(tgt,  []).extend(tst)
        elif src != "en" and tgt != "en":
            print(f"  [DATA]  Pivoting {src}→{tgt} through English...")
            en_src    = _fetch_pair("en", src, "train", max_n // 2)
            en_tgt    = _fetch_pair("en", tgt, "train", max_n // 2)
            en_to_src = {e: s for e, s, _ in en_src}
            pivoted   = [(en_to_src[e], t, tgt)
                         for e, t, _ in en_tgt if e in en_to_src][:max_n]
            train_by_lang.setdefault(tgt, []).extend(pivoted)
            print(f"  [DATA]  Pivoted  {src}→{tgt}  {len(pivoted):>7,}")

    # ── BUG FIX 2: Balance — cap mỗi target lang bằng nhau ───────────────────
    # Lấy số mẫu nhỏ nhất trong các ngôn ngữ non-English làm mức cap
    # → English không còn chiếm quá nhiều
    non_en_counts = [len(v) for k, v in train_by_lang.items() if k != "en"]
    cap = min(non_en_counts) if non_en_counts else max_n
    print(f"\n  [DATA]  Balancing — cap mỗi lang = {cap:,} samples")

    train_all, test_all = [], []
    for lang, samples in train_by_lang.items():
        random.shuffle(samples)
        taken = samples[:cap]
        train_all.extend(taken)
        print(f"  [DATA]  lang={lang:<4}  train={len(taken):,}")

    for lang, samples in test_by_lang.items():
        test_cap = max(1, cap // 5)
        random.shuffle(samples)
        test_all.extend(samples[:test_cap])

    random.shuffle(train_all)
    random.shuffle(test_all)
    print(f"\n  [DATA]  Train total : {len(train_all):,}")
    print(f"  [DATA]  Test  total : {len(test_all):,}\n")
    return train_all, test_all


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
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
    while True:
        yield from loader


# ── BUG FIX 1: Sampler đảm bảo mỗi batch chỉ có 1 target language ────────────
class SameLangSampler(torch.utils.data.Sampler):
    """
    Nhóm các index theo target language, shuffle trong từng nhóm,
    rồi interleave → batch_size bất kỳ cũng không bao giờ mix ngôn ngữ.
    """
    def __init__(self, dataset: TranslationDataset):
        from collections import defaultdict
        groups = defaultdict(list)
        for i, sample in enumerate(dataset.samples):
            groups[sample[2]].append(i)  # sample[2] = tgt_lang
        self.groups = list(groups.values())

    def __iter__(self):
        groups = [g[:] for g in self.groups]
        for g in groups:
            random.shuffle(g)
        random.shuffle(groups)
        iters = [iter(g) for g in groups]
        while iters:
            next_iters = []
            for it in iters:
                try:
                    yield next(it)
                    next_iters.append(it)
                except StopIteration:
                    pass
            iters = next_iters

    def __len__(self):
        return sum(len(g) for g in self.groups)


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scheduler, scaler, global_step: int, cycle: int):
    torch.save({
        "global_step": global_step,
        "cycle":       cycle,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "scaler":      scaler.state_dict(),
        "lr":          optimizer.param_groups[0]["lr"],
    }, CFG["checkpoint_path"])
    print(f"  [CKPT]  Saved → {CFG['checkpoint_path']}  "
          f"(global step {global_step}, cycle {cycle})")


def load_checkpoint(model, optimizer, scheduler, scaler, device: str) -> tuple:
    path = CFG["checkpoint_path"]
    if not os.path.exists(path):
        return 0, 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
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

    device      = CFG["device"]
    ckpt_sec    = CFG["checkpoint_minutes"] * 60
    accum_steps = CFG["accum_steps"]
    use_amp     = (device == "cuda")   # ✅ AMP chỉ dùng trên CUDA
    use_cuda    = (device == "cuda")

    model.to(device)

    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    if hasattr(model.decoder, "gradient_checkpointing_enable"):
        model.decoder.gradient_checkpointing_enable()

    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG["lr"], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = CFG["lr_factor"],
        patience = CFG["lr_patience"],
        min_lr   = CFG["lr_min"],
    )

    scaler = GradScaler(device, enabled=use_amp)

    global_step, cycle = load_checkpoint(model, optimizer, scheduler, scaler, device)

    loader_kwargs = dict(
        batch_size         = CFG["batch_size"],
        collate_fn         = collate_fn,
        num_workers        = CFG["num_workers"],
        pin_memory         = use_cuda,
        persistent_workers = CFG["num_workers"] > 0,
    )
    # BUG FIX 1: dùng SameLangSampler → mỗi batch chỉ có 1 target language
    train_loader = DataLoader(train_dataset,
                              sampler=SameLangSampler(train_dataset),
                              **loader_kwargs)
    test_loader  = DataLoader(test_dataset,
                              sampler=SameLangSampler(test_dataset),
                              **loader_kwargs)
    train_iter   = infinite_loader(train_loader)

    stop = False
    def _on_stop(sig, frame):
        nonlocal stop
        print("\n\n  [STOP]  Ctrl+C — finishing this step then saving...\n")
        stop = True
    signal.signal(signal.SIGINT, _on_stop)

    last_ckpt = time.time()
    eff_batch = CFG["batch_size"] * accum_steps

    print(f"\n{'═'*64}")
    print(f"  MTT Training  |  device={device}  |  lr={CFG['lr']:.2e}")
    print(f"  micro-batch={CFG['batch_size']}  accum={accum_steps}  effective batch={eff_batch}")
    print(f"  Mixed precision fp16: {use_amp}  |  Gradient checkpointing: ON")
    print(f"  Cycle  =  {CFG['train_steps']} train steps + {CFG['test_steps']} test steps")
    print(f"  LR auto-adjusted via ReduceLROnPlateau after every cycle")
    print(f"  Checkpoint every {CFG['checkpoint_minutes']} min + end of each cycle")
    print(f"  Press Ctrl+C to stop safely at any time")
    print(f"{'═'*64}\n")

    while not stop:
        cycle += 1

        print(f"\n{'─'*64}")
        print(f"  CYCLE {cycle}  |  global step: {global_step}")
        print(f"{'─'*64}\n")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1 — TRAIN
        # ══════════════════════════════════════════════════════════════════════
        model.train()
        train_loss_sum = 0.0
        accum_loss     = 0.0
        train_step     = 0

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(1, CFG["train_steps"] * accum_steps + 1):
            if stop:
                break

            srcs, tgts, langs, ners = next(train_iter)

            with autocast(device, enabled=use_amp):
                out  = model(
                    srcText    = srcs,
                    targetLang = langs[0],
                    targetText = tgts,
                    nerTags    = ners,
                    returnLoss = True,
                    device     = device,
                )
                loss = out["loss"] / accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if micro_step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step    += 1
                train_step     += 1
                train_loss_sum += accum_loss

                trans = out["translationLoss"]
                ner   = out["nerLoss"]
                print(
                    f"  [TRAIN]"
                    f"  global={global_step:>7}"
                    f"  cycle={cycle}  step={train_step:>3}/{CFG['train_steps']}"
                    f"  lang={langs[0]}"
                    f"  loss={accum_loss:.4f}"
                    + (f"  trans={trans.item()/accum_steps:.4f}" if trans is not None else "")
                    + (f"  ner={ner.item()/accum_steps:.4f}"     if ner   is not None else "")
                    + f"  lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                accum_loss = 0.0

                if time.time() - last_ckpt >= ckpt_sec:
                    save_checkpoint(model, optimizer, scheduler, scaler, global_step, cycle)
                    last_ckpt = time.time()

                # ✅ Chỉ gọi empty_cache khi đang dùng CUDA
                if use_cuda and global_step % 50 == 0:
                    torch.cuda.empty_cache()

        avg_train = train_loss_sum / max(train_step, 1)
        print(f"\n  [CYCLE {cycle}]  ── Train done ──  avg loss: {avg_train:.4f}\n")

        if stop:
            break

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2 — TEST
        # ══════════════════════════════════════════════════════════════════════
        model.eval()
        test_loss_sum  = 0.0
        test_trans_sum = 0.0
        test_ner_sum   = 0.0

        with torch.no_grad():
            for test_step, (srcs, tgts, langs, ners) in enumerate(test_loader, 1):
                if test_step > CFG["test_steps"]:
                    break

                with autocast(device, enabled=use_amp):
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

        # ✅ Chỉ gọi empty_cache khi đang dùng CUDA
        if use_cuda:
            torch.cuda.empty_cache()

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
        scheduler.step(avg_test)
        new_lr  = optimizer.param_groups[0]["lr"]

        if new_lr < prev_lr:
            print(f"  [LR  ]  Reduced  {prev_lr:.2e}  →  {new_lr:.2e}  (plateau)")
        else:
            print(f"  [LR  ]  Holding  {new_lr:.2e}  (still improving)")

        print()
        save_checkpoint(model, optimizer, scheduler, scaler, global_step, cycle)
        last_ckpt = time.time()

    save_checkpoint(model, optimizer, scheduler, scaler, global_step, cycle)
    print(f"\n  Stopped at global step {global_step}, cycle {cycle}. Goodbye!")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("  Fetching data from HuggingFace OPUS-100...\n")
    train_samples, test_samples = fetch_all_pairs()

    train_ds = TranslationDataset(train_samples)
    test_ds  = TranslationDataset(test_samples)

    model = MTT()
    model.paramsCalc()

    train(model, train_ds, test_ds)