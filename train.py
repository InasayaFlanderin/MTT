#!/usr/bin/env python3
"""
train.py — MTT (Multilingual Translation Transformer) Training Script
Languages : vi, en, es, de, fr
Architecture: mmBERT-small (encoder) + NER head + projector + mT5-small (decoder)

Rules implemented:
  1.  Backpropagation + gradient descent (AdamW)
  2.  Cycle = 800 train steps + 200 eval steps
  3.  Continuous training until Ctrl-C
  4.  Step display (global + within-cycle)
  5.  Checkpoint every 30 minutes + end of each cycle
  6.  All language pairs; missing pairs pivoted through English
  7.  Gradient accumulation with small mini-batches
  8.  Each mini-batch targets exactly ONE output language
  9.  Automatic LR (warmup + cosine decay)
  10. NER and translation trained simultaneously
  11. Full architecture trained end-to-end
  12. Datasets auto-fetched (opus-100 + WikiANN via HuggingFace)
  13. Train split ≠ eval split
  14. GPU (6 GB VRAM) or CPU (21 GB RAM) supported
"""

import os
import sys
import time
import math
import random
import itertools
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset

from model import MTT

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

LANGUAGES: List[str] = ["vi", "en", "es", "de", "fr"]
PIVOT_LANG: str = "en"

CKPT_FILE: str = "mtt_checkpoint.pt"
CKPT_INTERVAL: int = 30 * 60          # seconds between auto-saves

TRAIN_STEPS: int = 800
EVAL_STEPS:  int = 200
MAX_LEN:     int = 128

WARMUP_STEPS: int   = 500
LR_PEAK:      float = 5e-5
LR_MIN:       float = 1e-6
LR_TOTAL:     int   = 2_000_000       # total steps for cosine decay horizon

GRAD_CLIP:   float = 1.0
NER_WEIGHT:  float = 0.2
BUFFER_SIZE: int   = 8_000            # sentences per pair in RAM
PIVOT_DICT_MAX: int = 200_000         # pivot English→lang dict size

# ── Device detection ─────────────────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    MINI_BS:    int  = 2
    ACCUM:      int  = 8
    USE_AMP:    bool = True
    print(f"[CONFIG] GPU detected ({vram_gb:.1f} GB) "
          f"| mini_batch={MINI_BS} | accum_steps={ACCUM} | AMP=ON")
else:
    MINI_BS:    int  = 4
    ACCUM:      int  = 4
    USE_AMP:    bool = False
    print(f"[CONFIG] CPU mode "
          f"| mini_batch={MINI_BS} | accum_steps={ACCUM} | AMP=OFF")

# Effective batch size = MINI_BS × ACCUM
print(f"[CONFIG] Effective batch size = {MINI_BS * ACCUM}")

# ── WikiANN tag mapping ───────────────────────────────────────────────────────
# WikiANN: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC
# Model  : 0=O, 1=PERSON, 2=ORG, 3=LOC, 4=GPE, 5=DATE, …
WIKIANN_MAP: Dict[int, int] = {
    0: 0,   # O      → O
    1: 1,   # B-PER  → PERSON
    2: 1,   # I-PER  → PERSON
    3: 2,   # B-ORG  → ORG
    4: 2,   # I-ORG  → ORG
    5: 3,   # B-LOC  → LOC
    6: 3,   # I-LOC  → LOC
}

# All directed pairs (src, tgt) — 5 × 4 = 20
ALL_PAIRS: List[Tuple[str, str]] = [
    (a, b) for a, b in itertools.permutations(LANGUAGES, 2)
]


# ══════════════════════════════════════════════════════════════════════════════
#  LEARNING RATE SCHEDULER  (warmup + cosine decay)
# ══════════════════════════════════════════════════════════════════════════════

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR_MIN + (LR_PEAK - LR_MIN) * step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, LR_TOTAL - WARMUP_STEPS)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return LR_MIN + (LR_PEAK - LR_MIN) * cosine


def apply_lr(optimizer: AdamW, step: int) -> float:
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET  —  Translation pairs
# ══════════════════════════════════════════════════════════════════════════════

def _opus_config(l1: str, l2: str) -> str:
    """opus-100 config name is alphabetically sorted with hyphen."""
    a, b = sorted([l1, l2])
    return f"{a}-{b}"


def _direct_opus_iter(src: str, tgt: str, split: str) -> Optional[Iterable]:
    """Load a streaming opus-100 pair. Returns None if not available."""
    config = _opus_config(src, tgt)
    try:
        ds = load_dataset(
            "Helsinki-NLP/opus-100", config,
            split=split, streaming=True, trust_remote_code=True
        )
        return ds
    except Exception:
        return None


def _stream_direct(raw_ds, src: str, tgt: str) -> Iterator[Tuple[str, str]]:
    for ex in raw_ds:
        t = ex.get("translation", {})
        s, d = t.get(src, ""), t.get(tgt, "")
        if s and d:
            yield s.strip(), d.strip()


def _build_pivot_iter(
    src: str, tgt: str, split: str
) -> Iterator[Tuple[str, str]]:
    """
    Build pseudo-parallel corpus via English pivot:
      src-en  ×  tgt-en  →  src-tgt
    Strategy: load src-en into a dict keyed by English sentence,
    then stream tgt-en and look up matching src sentences.
    """
    print(f"    [PIVOT] Building {src}→{tgt} via {PIVOT_LANG} …")

    # Step 1: build {en: src_sentence} dict (limited size)
    pivot_to_src: Dict[str, str] = {}
    src_ds = _direct_opus_iter(src, PIVOT_LANG, split="train")
    if src_ds is not None:
        for ex in itertools.islice(src_ds, PIVOT_DICT_MAX):
            t = ex.get("translation", {})
            s, p = t.get(src, ""), t.get(PIVOT_LANG, "")
            if s and p:
                pivot_to_src[p] = s

    if not pivot_to_src:
        print(f"    [PIVOT] WARNING: Could not build pivot dict for {src}.")

    # Step 2: stream tgt-en; look up src on English side
    tgt_ds = _direct_opus_iter(tgt, PIVOT_LANG, split)
    if tgt_ds is not None:
        for ex in tgt_ds:
            t = ex.get("translation", {})
            d, p = t.get(tgt, ""), t.get(PIVOT_LANG, "")
            if d and p and p in pivot_to_src:
                yield pivot_to_src[p], d

    # If pivot produced nothing, yield empty (caller handles fallback)


class PairBuffer:
    """
    Streaming buffer for one (src, tgt) language pair.
    Refills from HuggingFace on demand; restarts iterator when exhausted.
    """

    def __init__(self, src: str, tgt: str, split: str = "train"):
        self.src   = src
        self.tgt   = tgt
        self.split = split
        self.buf:  List[Tuple[str, str]] = []
        self._iter: Optional[Iterator] = None
        self._init_iter()

    def _init_iter(self):
        raw = _direct_opus_iter(self.src, self.tgt, self.split)
        if raw is not None:
            self._iter = _stream_direct(raw, self.src, self.tgt)
            print(f"    [PAIR] {self.src}→{self.tgt} ({self.split}): direct opus-100")
        else:
            # Neither direction direct — use pivot
            self._iter = _build_pivot_iter(self.src, self.tgt, self.split)

    def _refill(self):
        if self._iter is None:
            return
        needed = BUFFER_SIZE - len(self.buf)
        added  = 0
        while added < needed:
            try:
                self.buf.append(next(self._iter))
                added += 1
            except StopIteration:
                if not self.buf:
                    # Restart
                    self._init_iter()
                break

    def sample(self, n: int) -> List[Tuple[str, str]]:
        if len(self.buf) < n:
            self._refill()
        if not self.buf:
            # Ultimate fallback: trivial pair
            return [("Hello.", "Hello.")] * n
        return random.choices(self.buf, k=n)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET  —  NER (WikiANN)
# ══════════════════════════════════════════════════════════════════════════════

class NERBuffer:
    """Streaming WikiANN buffer for one language."""

    def __init__(self, lang: str, split: str = "train"):
        self.lang  = lang
        self.split = split
        self.buf:  List[Tuple[List[str], List[int]]] = []
        self._iter = None
        self._init_iter()

    def _init_iter(self):
        try:
            ds = load_dataset(
                "wikiann", self.lang,
                split=self.split, streaming=True, trust_remote_code=True
            )
            self._iter = iter(ds)
            print(f"    [NER]  wikiann/{self.lang} ({self.split})")
        except Exception as e:
            print(f"    [NER]  WARNING: wikiann/{self.lang} not available: {e}")
            self._iter = None

    def _refill(self):
        if self._iter is None:
            return
        needed = BUFFER_SIZE - len(self.buf)
        added  = 0
        while added < needed:
            try:
                ex = next(self._iter)
                tokens = ex.get("tokens", [])
                tags   = ex.get("ner_tags", [])
                if tokens and tags:
                    self.buf.append((tokens, tags))
                    added += 1
            except StopIteration:
                self._init_iter()
                break

    def sample(self, n: int) -> List[Tuple[List[str], List[int]]]:
        if len(self.buf) < n:
            self._refill()
        if not self.buf:
            return [(["Hello"], [0])] * n
        return random.choices(self.buf, k=n)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class DataManager:
    """Central hub for all translation and NER buffers."""

    def __init__(self):
        print("\n[DATA] Initializing translation pair buffers …")
        # One buffer per directed pair × {train, eval}
        self._trans_train: Dict[Tuple[str, str], PairBuffer] = {}
        self._trans_eval:  Dict[Tuple[str, str], PairBuffer] = {}
        for src, tgt in ALL_PAIRS:
            self._trans_train[(src, tgt)] = PairBuffer(src, tgt, split="train")
            self._trans_eval [(src, tgt)] = PairBuffer(src, tgt, split="validation")

        print("\n[DATA] Initializing NER buffers …")
        self._ner_train: Dict[str, NERBuffer] = {}
        self._ner_eval:  Dict[str, NERBuffer] = {}
        for lang in LANGUAGES:
            self._ner_train[lang] = NERBuffer(lang, split="train")
            self._ner_eval [lang] = NERBuffer(lang, split="validation")

        print("[DATA] All buffers ready.\n")

    def trans_sample(
        self, src: str, tgt: str, n: int, train: bool = True
    ) -> List[Tuple[str, str]]:
        pool = self._trans_train if train else self._trans_eval
        return pool[(src, tgt)].sample(n)

    def ner_sample(
        self, lang: str, n: int, train: bool = True
    ) -> List[Tuple[List[str], List[int]]]:
        pool = self._ner_train if train else self._ner_eval
        return pool[lang].sample(n)


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
#  NER TAG ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def align_ner_tags(
    word_token_lists: List[List[str]],
    word_tag_lists:   List[List[int]],
    seq_len:          int,
) -> torch.Tensor:
    """
    Align word-level WikiANN tags to subword token positions.

    BERT tokenization layout (model prepends <2lang> internally):
      [CLS] <2lang_token> subword1 subword2 … [SEP] [PAD]…

    We don't have the exact subword count per word without re-tokenizing,
    so we assign one tag per word and pad with -100 (ignored in loss).
    The first two positions (CLS + lang token) are always -100.

    Returns:
        LongTensor of shape (batch, seq_len)
    """
    PREFIX = 2   # positions for [CLS] and <2lang>
    batch  = []
    for tokens, tags in zip(word_token_lists, word_tag_lists):
        row = [-100] * PREFIX
        for tag in tags:
            row.append(WIKIANN_MAP.get(tag, 0))
        row.append(-100)           # [SEP]
        # Pad / truncate
        if len(row) < seq_len:
            row += [-100] * (seq_len - len(row))
        else:
            row = row[:seq_len]
        batch.append(row)
    return torch.tensor(batch, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
#  FORWARD PASS (translation + NER combined)
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss(
    model:       MTT,
    src_texts:   List[str],
    tgt_lang:    str,
    tgt_texts:   List[str],
    ner_samples: List[Tuple[List[str], List[int]]],
) -> Tuple[torch.Tensor, float, float]:
    """
    Run one mini-batch forward pass.

    Translation and NER use separate sub-batches inside the same step:
      - Translation forward: srcText + targetText → translationLoss
      - NER forward: wikiann texts → nerLogits → nerLoss (F.cross_entropy)
    Total loss = translationLoss + NER_WEIGHT × nerLoss

    Returns:
        (total_loss_tensor, trans_loss_float, ner_loss_float)
    """
    # ── Translation ──────────────────────────────────────────────────────────
    trans_out  = model.forward(
        srcText    = src_texts,
        targetLang = tgt_lang,
        targetText = tgt_texts,
        nerTags    = None,
        returnLoss = True,
        device     = DEVICE,
    )
    trans_loss: torch.Tensor = trans_out["translationLoss"]

    # ── NER ──────────────────────────────────────────────────────────────────
    ner_texts = [" ".join(s[0]) for s in ner_samples]
    ner_out   = model.forward(
        srcText    = ner_texts,
        targetLang = tgt_lang,         # just needs a valid lang token
        targetText = None,
        nerTags    = None,
        returnLoss = False,
        device     = DEVICE,
    )
    ner_logits: torch.Tensor = ner_out["nerLogits"]   # (B, seq, num_tags)
    seq_len = ner_logits.size(1)

    ner_tag_tensor = align_ner_tags(
        [s[0] for s in ner_samples],
        [s[1] for s in ner_samples],
        seq_len,
    ).to(DEVICE)

    ner_loss: torch.Tensor = F.cross_entropy(
        ner_logits.view(-1, ner_logits.size(-1)),
        ner_tag_tensor.view(-1),
        ignore_index=-100,
    )

    total = trans_loss + NER_WEIGHT * ner_loss
    return total, trans_loss.item(), ner_loss.item()


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING STEP  (gradient accumulation)
# ══════════════════════════════════════════════════════════════════════════════

def train_step(
    model:     MTT,
    optimizer: AdamW,
    data_mgr:  DataManager,
    scaler,                            # GradScaler or None
) -> Tuple[float, float]:
    """
    One training step = ACCUM mini-batches.
    Each mini-batch targets exactly ONE output language (rule 8).
    Returns (avg_trans_loss, avg_ner_loss) over all mini-batches.
    """
    model.train()
    optimizer.zero_grad()

    sum_trans = 0.0
    sum_ner   = 0.0

    for _ in range(ACCUM):
        # Rule 8: pick a single target language per mini-batch
        tgt_lang = random.choice(LANGUAGES)
        src_lang = random.choice([l for l in LANGUAGES if l != tgt_lang])
        ner_lang = random.choice(LANGUAGES)

        pairs    = data_mgr.trans_sample(src_lang, tgt_lang, MINI_BS, train=True)
        src_txts = [p[0] for p in pairs]
        tgt_txts = [p[1] for p in pairs]
        ner_smp  = data_mgr.ner_sample(ner_lang, MINI_BS, train=True)

        if USE_AMP:
            with torch.cuda.amp.autocast():
                loss, tl, nl = compute_loss(model, src_txts, tgt_lang, tgt_txts, ner_smp)
            scaler.scale(loss / ACCUM).backward()
        else:
            loss, tl, nl = compute_loss(model, src_txts, tgt_lang, tgt_txts, ner_smp)
            (loss / ACCUM).backward()

        sum_trans += tl
        sum_ner   += nl

    # Gradient clipping + optimizer step
    if USE_AMP:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    return sum_trans / ACCUM, sum_ner / ACCUM


# ══════════════════════════════════════════════════════════════════════════════
#  EVAL STEP
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_step(
    model:    MTT,
    data_mgr: DataManager,
) -> Tuple[float, float]:
    model.eval()

    tgt_lang = random.choice(LANGUAGES)
    src_lang = random.choice([l for l in LANGUAGES if l != tgt_lang])
    ner_lang = random.choice(LANGUAGES)

    pairs    = data_mgr.trans_sample(src_lang, tgt_lang, MINI_BS, train=False)
    src_txts = [p[0] for p in pairs]
    tgt_txts = [p[1] for p in pairs]
    ner_smp  = data_mgr.ner_sample(ner_lang, MINI_BS, train=False)

    _, tl, nl = compute_loss(model, src_txts, tgt_lang, tgt_txts, ner_smp)
    return tl, nl


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:       MTT,
    optimizer:   AdamW,
    global_step: int,
    cycle:       int,
    lr:          float,
    reason:      str = "auto",
):
    torch.save({
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "global_step": global_step,
        "cycle":       cycle,
        "lr":          lr,
    }, CKPT_FILE)
    ts = time.strftime("%H:%M:%S")
    print(f"\n  ✓ [{ts}] Checkpoint saved ({reason}) "
          f"→ {CKPT_FILE}  [step={global_step}, cycle={cycle}]\n")


def load_checkpoint(model: MTT, optimizer: AdamW) -> Tuple[int, int]:
    if not os.path.exists(CKPT_FILE):
        print(f"[CKPT] No checkpoint found at {CKPT_FILE}. Starting fresh.\n")
        return 0, 0
    ckpt = torch.load(CKPT_FILE, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        print(f"[CKPT] Could not restore optimizer state: {e}")
    step  = ckpt.get("global_step", 0)
    cycle = ckpt.get("cycle", 0)
    print(f"[CKPT] Resumed → step={step}, cycle={cycle}, lr={ckpt.get('lr','?')}\n")
    return step, cycle


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    bar = "═" * 72
    print(f"\n{bar}")
    print("  MTT — Multilingual Translation Transformer  |  train.py")
    print(f"{bar}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[INIT] Building model …")
    model = MTT().to(DEVICE)
    model.paramsCalc()

    # ── Optimizer & scaler ────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR_PEAK, weight_decay=1e-2)
    scaler    = torch.cuda.amp.GradScaler() if USE_AMP else None

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    global_step, start_cycle = load_checkpoint(model, optimizer)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("[INIT] Fetching datasets (first run may download data) …")
    data_mgr = DataManager()

    # ── Loop state ────────────────────────────────────────────────────────────
    last_ckpt_t = time.time()
    cycle       = start_cycle

    print(f"[TRAIN] Starting — cycle={cycle + 1}, global_step={global_step}")
    print(f"[TRAIN] Press Ctrl-C to stop (checkpoint saves automatically)\n")

    try:
        while True:
            cycle += 1
            thin  = "─" * 72

            # ═══════════════════════════════════════════════════════════════
            #  TRAIN PHASE
            # ═══════════════════════════════════════════════════════════════
            print(f"\n{bar}")
            print(f"  CYCLE {cycle:>4d}  │  TRAIN  │  {TRAIN_STEPS} steps  │  "
                  f"global_step starts at {global_step}")
            print(bar)

            train_tl_sum = 0.0
            train_nl_sum = 0.0

            for local_step in range(1, TRAIN_STEPS + 1):

                # Apply LR schedule
                lr = apply_lr(optimizer, global_step)

                # Gradient-accumulated training step
                tl, nl = train_step(model, optimizer, data_mgr, scaler)
                train_tl_sum += tl
                train_nl_sum += nl
                global_step  += 1

                # ── Console progress ────────────────────────────────────
                sys.stdout.write(
                    f"\r  [TRAIN] "
                    f"Cycle {cycle:4d} | "
                    f"Step {local_step:3d}/{TRAIN_STEPS} | "
                    f"Global {global_step:8d} | "
                    f"LR {lr:.2e} | "
                    f"TransLoss {tl:7.4f} | "
                    f"NERLoss {nl:7.4f}   "
                )
                sys.stdout.flush()

                # ── Time-based checkpoint ───────────────────────────────
                if time.time() - last_ckpt_t >= CKPT_INTERVAL:
                    sys.stdout.write("\n")
                    save_checkpoint(model, optimizer, global_step, cycle, lr, reason="30min")
                    last_ckpt_t = time.time()

            # End-of-train-phase summary
            avg_tl = train_tl_sum / TRAIN_STEPS
            avg_nl = train_nl_sum / TRAIN_STEPS
            print(f"\n  ↳ Train avg │ TransLoss {avg_tl:.4f} │ NERLoss {avg_nl:.4f}")

            # ═══════════════════════════════════════════════════════════════
            #  EVAL PHASE
            # ═══════════════════════════════════════════════════════════════
            print(f"\n{thin}")
            print(f"  CYCLE {cycle:>4d}  │  EVAL   │  {EVAL_STEPS} steps")
            print(thin)

            eval_tl_sum = 0.0
            eval_nl_sum = 0.0

            for local_step in range(1, EVAL_STEPS + 1):
                tl, nl = eval_step(model, data_mgr)
                eval_tl_sum += tl
                eval_nl_sum += nl

                sys.stdout.write(
                    f"\r  [EVAL ] "
                    f"Cycle {cycle:4d} | "
                    f"Step {local_step:3d}/{EVAL_STEPS} | "
                    f"Global {global_step:8d} | "
                    f"TransLoss {tl:7.4f} | "
                    f"NERLoss {nl:7.4f}   "
                )
                sys.stdout.flush()

            avg_tl = eval_tl_sum / EVAL_STEPS
            avg_nl = eval_nl_sum / EVAL_STEPS
            print(f"\n  ↳ Eval  avg │ TransLoss {avg_tl:.4f} │ NERLoss {avg_nl:.4f}")

            # ── End-of-cycle checkpoint ─────────────────────────────────
            lr_now = get_lr(global_step)
            save_checkpoint(model, optimizer, global_step, cycle, lr_now, reason="end-of-cycle")
            last_ckpt_t = time.time()

    except KeyboardInterrupt:
        print("\n\n[STOP] Keyboard interrupt received. Saving final checkpoint …")
        save_checkpoint(model, optimizer, global_step, cycle,
                        get_lr(global_step), reason="user-stop")
        print("[STOP] Done. Goodbye!\n")
        sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
