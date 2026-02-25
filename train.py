"""
MTT Training Script
────────────────────────────────────────────────────────────────
One cycle = 800 train steps + 200 test steps = 1000 total
Each step = one monolingual batch (one language per batch, always)
After each cycle: LR auto-adjusted based on avg test loss
Runs forever — press Ctrl+C to stop safely
Checkpoint every 30 minutes + end of every cycle
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
    languages            = ["de", "fr", "vi", "es"],   # paired with "en" + each other
    max_samples_per_pair = 50_000,

    train_steps          = 800,    # train steps per cycle
    test_steps           = 200,    # test  steps per cycle (runs right after train)

    batch_size           = 4,
    accum_steps          = 4,      # gradient accumulation → effective batch = 4×4 = 16
    lr                   = 5e-5,
    grad_clip            = 1.0,

    # ReduceLROnPlateau — fires once per cycle using avg test loss
    lr_factor            = 0.5,
    lr_patience          = 2,
    lr_min               = 1e-7,

    checkpoint_path      = "mtt_checkpoint.pt",
    checkpoint_minutes   = 30.0,

    device               = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers          = max(0, (os.cpu_count() or 1) - 1),

)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_opus(lang: str, split: str, max_n: int) -> list:
    """
    Fetch en↔lang from OPUS-100. Returns [(en_text, lang_text)].
    OPUS-100 always stores pairs as "en-xx" (English on one side).
    """
    from datasets import load_dataset  # lazy import

    for key in [f"en-{lang}", f"{lang}-en"]:
        try:
            ds  = load_dataset("Helsinki-NLP/opus-100", key, split=split)
            out = []
            for row in ds:
                t = row["translation"]
                if "en" in t and lang in t:
                    out.append((t["en"], t[lang]))
                if len(out) >= max_n:
                    break
            print(f"  [DATA]  en↔{lang:<4}  {split:<12}  {len(out):>7,}  (key={key})")
            return out
        except Exception as e:
            print(f"  [DATA]  en↔{lang}  key={key} failed: {e}")
            continue

    print(f"  [DATA]  en↔{lang}  not found in OPUS-100, skipping.")
    return []


def fetch_all_pairs() -> tuple:
    """
    Fetch en↔X for every language X.
    Stores data in BOTH directions:
      - under tgt=X : (en_src → X_tgt)   model learns to translate INTO X
      - under tgt=en: (X_src → en_tgt)   model learns to translate INTO en
    Also builds cross-language pairs via English pivot and stores them
    under the correct target language with proper test splits.

    Returns:
        train_by_lang : { tgt_lang -> [(src_text, tgt_text), ...] }
        test_by_lang  : { tgt_lang -> [(src_text, tgt_text), ...] }
    """
    max_n     = CFG["max_samples_per_pair"]
    langs     = CFG["languages"]   # ["de", "fr", "vi", "es"]

    train_by_lang: dict[str, list] = {}
    test_by_lang:  dict[str, list] = {}

    # Cache en↔X data so we don't re-download for pivot
    cache_train: dict[str, list] = {}   # lang -> [(en, lang_text)]
    cache_test:  dict[str, list] = {}

    # ── Step 1: fetch en↔X for every X ────────────────────────────────────────
    for lang in langs:
        train_rows = _fetch_opus(lang, "train",      max_n)
        test_rows  = _fetch_opus(lang, "validation", max_n // 5)

        cache_train[lang] = train_rows
        cache_test[lang]  = test_rows

        if not train_rows:
            print(f"  [DATA]  WARNING: no data for {lang}, skipping.")
            continue

        # en → lang  (model learns to produce lang)
        train_by_lang.setdefault(lang, []).extend(
            [(en, lx) for en, lx in train_rows]
        )
        test_by_lang.setdefault(lang, []).extend(
            [(en, lx) for en, lx in test_rows]
        )

        # lang → en  (model learns to produce en)
        train_by_lang.setdefault("en", []).extend(
            [(lx, en) for en, lx in train_rows]
        )
        test_by_lang.setdefault("en", []).extend(
            [(lx, en) for en, lx in test_rows]
        )

    # ── Step 2: cross-language pairs via English pivot ─────────────────────────
    # For every pair (l1, l2), build l1→l2 by matching on the English sentence.
    # Both train AND test are pivoted so test sets are never empty.
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            l1, l2 = langs[i], langs[j]
            if not cache_train.get(l1) or not cache_train.get(l2):
                continue

            print(f"  [DATA]  Pivoting {l1}→{l2} and {l2}→{l1} through English...")

            # train pivot
            en_to_l1 = {en: lx for en, lx in cache_train[l1]}
            en_to_l2 = {en: lx for en, lx in cache_train[l2]}
            common   = set(en_to_l1) & set(en_to_l2)

            pivot_l1_l2 = [(en_to_l1[e], en_to_l2[e]) for e in common][:max_n]
            pivot_l2_l1 = [(en_to_l2[e], en_to_l1[e]) for e in common][:max_n]
            random.shuffle(pivot_l1_l2)
            random.shuffle(pivot_l2_l1)

            # test pivot
            en_to_l1_t = {en: lx for en, lx in cache_test.get(l1, [])}
            en_to_l2_t = {en: lx for en, lx in cache_test.get(l2, [])}
            common_t   = set(en_to_l1_t) & set(en_to_l2_t)

            pivot_l1_l2_t = [(en_to_l1_t[e], en_to_l2_t[e]) for e in common_t][:max_n // 5]
            pivot_l2_l1_t = [(en_to_l2_t[e], en_to_l1_t[e]) for e in common_t][:max_n // 5]

            train_by_lang.setdefault(l2, []).extend(pivot_l1_l2)
            train_by_lang.setdefault(l1, []).extend(pivot_l2_l1)
            test_by_lang.setdefault(l2,  []).extend(pivot_l1_l2_t)
            test_by_lang.setdefault(l1,  []).extend(pivot_l2_l1_t)

            print(f"  [DATA]  Pivoted  {l1}→{l2}  train={len(pivot_l1_l2):,}  test={len(pivot_l1_l2_t):,}")
            print(f"  [DATA]  Pivoted  {l2}→{l1}  train={len(pivot_l2_l1):,}  test={len(pivot_l2_l1_t):,}")

    for lang in train_by_lang:
        random.shuffle(train_by_lang[lang])
    for lang in test_by_lang:
        random.shuffle(test_by_lang[lang])

    print(f"\n  [DATA]  Languages loaded: {sorted(train_by_lang.keys())}")
    for lang in sorted(train_by_lang):
        print(f"  [DATA]    {lang}  "
              f"train={len(train_by_lang[lang]):,}  "
              f"test={len(test_by_lang.get(lang, [])):,}")
    print()
    return train_by_lang, test_by_lang


# ══════════════════════════════════════════════════════════════════════════════
# DATASET  — one dataset per language, all samples share the same target lang
# ══════════════════════════════════════════════════════════════════════════════

class MonolingualDataset(Dataset):
    """
    Every sample in this dataset has the same target language.
    One batch = one language = one clean forward pass, no grouping needed.
    """
    def __init__(self, samples: list):
        self.samples = samples  # [(src_text, tgt_text), ...]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # (src_text, tgt_text)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    return list(srcs), list(tgts)


def _infinite(loader: DataLoader):
    """Yield batches forever, reshuffling each epoch."""
    while True:
        yield from loader


def make_lang_loaders(by_lang: dict, batch_size: int,
                      num_workers: int, pin_memory: bool,
                      shuffle: bool) -> dict:
    """Build one infinite iterator per language."""
    return {
        lang: _infinite(DataLoader(
            MonolingualDataset(samples),
            batch_size         = batch_size,
            shuffle            = shuffle,
            collate_fn         = collate_fn,
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            persistent_workers = num_workers > 0,
            drop_last          = True,
        ))
        for lang, samples in by_lang.items()
    }


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
          train_by_lang: dict,
          test_by_lang:  dict):

    device      = CFG["device"]
    ckpt_sec    = CFG["checkpoint_minutes"] * 60
    accum_steps = CFG["accum_steps"]
    use_amp     = (device == "cuda")
    use_pin     = (device == "cuda")

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

    # One infinite iterator per language for train and test
    train_iters = make_lang_loaders(train_by_lang, CFG["batch_size"],
                                    CFG["num_workers"], use_pin, shuffle=True)
    test_iters  = make_lang_loaders(test_by_lang,  CFG["batch_size"],
                                    CFG["num_workers"], use_pin, shuffle=False)

    # Round-robin order so all languages are visited equally each cycle
    train_langs = sorted(train_iters.keys())
    test_langs  = sorted(test_iters.keys())

    # Graceful Ctrl+C
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
    print(f"  Mixed precision fp16 : {use_amp}  |  Gradient checkpointing : ON")
    print(f"  Cycle = {CFG['train_steps']} train steps + {CFG['test_steps']} test steps")
    print(f"  Each step = one monolingual batch (one language, always correct)")
    print(f"  Languages : {train_langs}")
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
        # PHASE 1 — TRAIN  (800 steps)
        # Each step: pick next language round-robin → pull one monolingual batch
        # ══════════════════════════════════════════════════════════════════════
        model.train()
        train_loss_sum = 0.0
        accum_loss     = 0.0
        train_step     = 0
        optimizer.zero_grad(set_to_none=True)

        # Shuffled language schedule — one language per micro-step.
        # Each block contains every language once, shuffled randomly.
        total_micro = CFG["train_steps"] * accum_steps
        schedule    = []
        while len(schedule) < total_micro:
            block = train_langs.copy()
            random.shuffle(block)
            schedule.extend(block)
        schedule = schedule[:total_micro]

        langs_in_step = []   # collect langs across micro-batches for logging

        for micro_step, lang in enumerate(schedule, 1):
            if stop:
                break
            langs_in_step.append(lang)
            srcs, tgts = next(train_iters[lang])

            with autocast(device, enabled=use_amp):
                out  = model(
                    srcText    = srcs,
                    targetLang = lang,      # entire batch is this language
                    targetText = tgts,
                    nerTags    = None,
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

                trans      = out["translationLoss"]
                ner        = out["nerLoss"]
                langs_str  = "+".join(langs_in_step)   # e.g. "de+fr+vi+es"
                print(
                    f"  [TRAIN]"
                    f"  global={global_step:>7}"
                    f"  cycle={cycle}  step={train_step:>3}/{CFG['train_steps']}"
                    f"  langs=[{langs_str}]"
                    f"  loss={accum_loss:.4f}"
                    + (f"  trans={trans.item()/accum_steps:.4f}" if trans is not None else "")
                    + (f"  ner={ner.item()/accum_steps:.4f}"     if ner   is not None else "")
                    + f"  lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                accum_loss    = 0.0
                langs_in_step = []   # reset for next step

                if time.time() - last_ckpt >= ckpt_sec:
                    save_checkpoint(model, optimizer, scheduler, scaler, global_step, cycle)
                    last_ckpt = time.time()

                if global_step % 50 == 0 and device == "cuda":
                    torch.cuda.empty_cache()

        avg_train = train_loss_sum / max(train_step, 1)
        print(f"\n  [CYCLE {cycle}]  ── Train done ──  avg loss: {avg_train:.4f}\n")

        if stop:
            break

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2 — TEST  (200 steps, no gradients)
        # Same round-robin: each step is one monolingual batch
        # ══════════════════════════════════════════════════════════════════════
        model.eval()
        test_loss_sum  = 0.0
        test_trans_sum = 0.0
        test_ner_sum   = 0.0

        test_schedule = []
        while len(test_schedule) < CFG["test_steps"]:
            block = test_langs.copy()
            random.shuffle(block)
            test_schedule.extend(block)
        test_schedule = test_schedule[:CFG["test_steps"]]

        with torch.no_grad():
            for test_step, lang in enumerate(test_schedule, 1):
                srcs, tgts = next(test_iters[lang])

                with autocast(device, enabled=use_amp):
                    out = model(
                        srcText    = srcs,
                        targetLang = lang,
                        targetText = tgts,
                        nerTags    = None,
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
                    f"  lang={lang}"
                    f"  loss={lv:.4f}"
                )

        if device == "cuda":
            torch.cuda.empty_cache()

        n         = CFG["test_steps"]
        avg_test  = test_loss_sum  / n
        avg_trans = test_trans_sum / n
        avg_ner   = test_ner_sum   / n
        print(f"\n  [CYCLE {cycle}]  ── Test done ──  "
              f"avg loss: {avg_test:.4f}  trans: {avg_trans:.4f}  ner: {avg_ner:.4f}")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 3 — AUTO ADJUST LR  (based on avg test loss)
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
    train_by_lang, test_by_lang = fetch_all_pairs()

    model = MTT()
    model.paramsCalc()

    train(model, train_by_lang, test_by_lang)
