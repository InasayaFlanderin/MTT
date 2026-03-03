"""
MTT Training Script
────────────────────────────────────────────────────────────────
One cycle = train_steps train steps + test_steps test steps
Each step = one monolingual batch (one language per batch, always)
After each cycle: LR auto-adjusted based on avg test loss
Runs forever — press Ctrl+C to stop safely
Checkpoint every 30 minutes + end of every cycle
All language pairs fetched automatically from OPUS-100

Install:
    pip install torch datasets sentencepiece transformers spacy
    python -m spacy download xx_ent_wiki_sm
"""

import gc
import math
import os
import random
import signal
import time

import spacy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

from model import MTT


# ── spaCy label → nerVocab key ────────────────────────────────────────────────
_SPACY_TO_NER = {
    "PERSON":   "PERSON",
    "PER":      "PERSON",
    "ORG":      "ORG",
    "LOC":      "LOC",
    "GPE":      "GPE",
    "DATE":     "DATE",
    "MONEY":    "MONEY",
    "PERCENT":  "PERCENT",
    "TIME":     "TIME",
    "QUANTITY": "QUANTITY",
    "CARDINAL": "QUANTITY",
}


def _load_spacy(model_name: str = "xx_ent_wiki_sm"):
    try:
        nlp = spacy.load(
            model_name, disable=["tagger", "parser", "senter", "lemmatizer"]
        )
        print(f"  [NER ]  spaCy '{model_name}' loaded.")
        return nlp
    except OSError:
        print(f"  [NER ]  Model '{model_name}' not found. Run:")
        print(f"          python -m spacy download {model_name}")
        raise


def _extract_ner_tags(
    src_texts: list[str],
    target_lang: str,
    nlp,
    tokenizer,
    ner_vocab: dict,
    max_length: int = 128,
    device: str = "cpu",
) -> torch.Tensor:
    prefix = f"<2{target_lang}> "
    prefix_len = len(prefix)
    tagged_src = [prefix + t for t in src_texts]

    char_tag_maps = []
    for doc in nlp.pipe(src_texts, batch_size=32):
        char_tags: dict[int, int] = {}
        for ent in doc.ents:
            tag_id = ner_vocab.get(_SPACY_TO_NER.get(ent.label_, ""), 0)
            if tag_id:
                for c in range(ent.start_char, ent.end_char):
                    char_tags[c] = tag_id
        char_tag_maps.append(char_tags)

    # NOTE: requires a fast tokenizer (T5TokenizerFast).
    # return_offsets_mapping raises NotImplementedError on slow tokenizers.
    enc = tokenizer(
        tagged_src,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    B, seq_len = enc.input_ids.shape
    tag_tensor = torch.full((B, seq_len), -100, dtype=torch.long)

    for i in range(B):
        for j in range(seq_len):
            start, end = enc.offset_mapping[i, j].tolist()
            if start == 0 and end == 0:
                continue
            if end <= prefix_len:
                continue
            src_start = start - prefix_len
            src_end   = end   - prefix_len
            tag = 0
            for c in range(max(0, src_start), src_end):
                if c in char_tag_maps[i]:
                    tag = char_tag_maps[i][c]
                    break
            tag_tensor[i, j] = tag

    return tag_tensor.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# AMP DTYPE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _pick_amp_dtype(device: str):
    if device != "cuda":
        return False, torch.float32

    if torch.cuda.is_bf16_supported():
        print("  [AMP ]  bf16 được hỗ trợ → dùng bf16 (nhanh + ổn định)")
        return True, torch.bfloat16

    gpu_name = torch.cuda.get_device_name(0)
    print(f"  [AMP ]  GPU '{gpu_name}' không hỗ trợ bf16 → tắt AMP, dùng fp32")
    print("  [AMP ]  (fp16 bị bỏ vì dễ NaN; fp32 chậm hơn ~20% nhưng luôn ổn định)")
    return False, torch.float32


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    languages              = ["de", "fr", "vi", "es"],
    max_samples_per_pair   = 50_000,
    train_steps            = 800,
    test_steps             = 200,
    batch_size             = 8,
    accum_steps            = 16,
    lr                     = 5e-5,
    grad_clip              = 1.0,
    warmup_steps           = 400,
    lr_min                 = 1e-7,
    checkpoint_path        = "mtt_checkpoint.pt",
    checkpoint_minutes     = 30.0,
    device                 = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers            = max(0, (os.cpu_count() or 1) - 1),
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════


def _fetch_opus(lang: str, split: str, max_n: int) -> list:
    from datasets import load_dataset

    for key in [f"en-{lang}", f"{lang}-en"]:
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", key, split=split)
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
    max_n = CFG["max_samples_per_pair"]
    langs = CFG["languages"]

    train_by_lang: dict[str, list] = {}
    test_by_lang:  dict[str, list] = {}
    cache_train:   dict[str, list] = {}
    cache_test:    dict[str, list] = {}

    for lang in langs:
        train_rows = _fetch_opus(lang, "train",      max_n)
        test_rows  = _fetch_opus(lang, "validation", max_n // 5)

        cache_train[lang] = train_rows
        cache_test[lang]  = test_rows

        if not train_rows:
            print(f"  [DATA]  WARNING: no data for {lang}, skipping.")
            continue

        train_keys = {s + "|||" + t for s, t in train_rows}
        test_rows  = [(s, t) for s, t in test_rows if s + "|||" + t not in train_keys]

        train_by_lang.setdefault(lang, []).extend([(en, lx) for en, lx in train_rows])
        test_by_lang.setdefault(lang,  []).extend([(en, lx) for en, lx in test_rows])
        train_by_lang.setdefault("en", []).extend([(lx, en) for en, lx in train_rows])
        test_by_lang.setdefault("en",  []).extend([(lx, en) for en, lx in test_rows])

    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            l1, l2 = langs[i], langs[j]
            if not cache_train.get(l1) or not cache_train.get(l2):
                continue

            print(f"  [DATA]  Pivoting {l1}→{l2} and {l2}→{l1} through English...")

            en_to_l1 = {en: lx for en, lx in cache_train[l1]}
            en_to_l2 = {en: lx for en, lx in cache_train[l2]}
            common   = set(en_to_l1) & set(en_to_l2)

            pivot_l1_l2 = [(en_to_l1[e], en_to_l2[e]) for e in common][:max_n]
            pivot_l2_l1 = [(en_to_l2[e], en_to_l1[e]) for e in common][:max_n]
            random.shuffle(pivot_l1_l2)
            random.shuffle(pivot_l2_l1)

            en_to_l1_t  = {en: lx for en, lx in cache_test.get(l1, [])}
            en_to_l2_t  = {en: lx for en, lx in cache_test.get(l2, [])}
            common_t    = set(en_to_l1_t) & set(en_to_l2_t)

            pivot_l1_l2_t = [(en_to_l1_t[e], en_to_l2_t[e]) for e in common_t][: max_n // 5]
            pivot_l2_l1_t = [(en_to_l2_t[e], en_to_l1_t[e]) for e in common_t][: max_n // 5]

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
        print(
            f"  [DATA]    {lang}  "
            f"train={len(train_by_lang[lang]):,}  "
            f"test={len(test_by_lang.get(lang, [])):,}"
        )
    print()
    return train_by_lang, test_by_lang


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════


class MonolingualDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    return list(srcs), list(tgts)


def _infinite(loader: DataLoader):
    while True:
        yield from loader


def make_lang_loaders(
    by_lang: dict, batch_size: int, num_workers: int, pin_memory: bool, shuffle: bool
) -> dict:
    return {
        lang: _infinite(
            DataLoader(
                MonolingualDataset(samples),
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
                drop_last=True,
            )
        )
        for lang, samples in by_lang.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════


def save_checkpoint(model, optimizer, scheduler, scaler, global_step: int, cycle: int):
    torch.save(
        {
            "global_step": global_step,
            "cycle":       cycle,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "scaler":      scaler.state_dict(),
            "lr":          optimizer.param_groups[0]["lr"],
        },
        CFG["checkpoint_path"],
    )
    print(
        f"  [CKPT]  Saved → {CFG['checkpoint_path']}  "
        f"(global step {global_step}, cycle {cycle})"
    )


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
    print(
        f"  [CKPT]  Resumed — global step {ckpt['global_step']}  "
        f"cycle {ckpt['cycle']}  lr={ckpt['lr']:.2e}"
    )
    return ckpt["global_step"], ckpt["cycle"]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════


def train(model: MTT, train_by_lang: dict, test_by_lang: dict, nlp=None):

    device    = CFG["device"]
    ckpt_sec  = CFG["checkpoint_minutes"] * 60
    accum_steps = CFG["accum_steps"]

    use_amp, amp_dtype = _pick_amp_dtype(device)
    use_pin = device == "cuda"

    model.to(device)

    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    if hasattr(model.decoder, "gradient_checkpointing_enable"):
        model.decoder.gradient_checkpointing_enable()

    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-2)

    _warmup = CFG["warmup_steps"]
    _cycle  = CFG["train_steps"]
    _min_r  = CFG["lr_min"] / CFG["lr"]

    def _lr_lambda(step: int) -> float:
        if step < _warmup:
            return step / max(1, _warmup)
        pos = (step - _warmup) % _cycle
        return _min_r + (1 - _min_r) * 0.5 * (1 + math.cos(math.pi * pos / _cycle))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler    = GradScaler(enabled=use_amp)

    global_step, cycle = load_checkpoint(model, optimizer, scheduler, scaler, device)

    train_iters = make_lang_loaders(
        train_by_lang, CFG["batch_size"], CFG["num_workers"], use_pin, shuffle=True
    )
    test_iters = make_lang_loaders(
        test_by_lang, CFG["batch_size"], CFG["num_workers"], use_pin, shuffle=False
    )

    train_langs = sorted(train_iters.keys())
    test_langs  = sorted(test_iters.keys())

    stop = False

    def _on_stop(sig, frame):
        nonlocal stop
        print("\n\n  [STOP]  Ctrl+C — finishing this step then saving...\n")
        stop = True

    signal.signal(signal.SIGINT, _on_stop)

    last_ckpt  = time.time()
    eff_batch  = CFG["batch_size"] * accum_steps
    amp_label  = "bf16" if (use_amp and amp_dtype == torch.bfloat16) else "fp32 (AMP off)"

    print(f"\n{'═'*64}")
    print(f"  MTT Training  |  device={device}  |  lr={CFG['lr']:.2e}")
    print(f"  micro-batch={CFG['batch_size']}  accum={accum_steps}  effective batch={eff_batch}")
    print(f"  Precision : {amp_label}  |  Gradient checkpointing : ON")
    print(f"  Cycle = {CFG['train_steps']} train steps + {CFG['test_steps']} test steps")
    print(f"  Each step = one monolingual batch (one language, always correct)")
    print(f"  Languages : {train_langs}")
    print(f"  LR: warmup {CFG['warmup_steps']} steps → cosine/cycle → min={CFG['lr_min']:.1e}")
    print(f"  NER: spaCy → offset_mapping align → forward() nerTags")
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
        train_step     = 0
        optimizer.zero_grad(set_to_none=True)

        # FIX 5: Build the schedule by cycling through a shuffled language list
        # repeatedly until we have exactly total_micro entries.  The previous
        # approach sliced a concatenated list which could under-represent languages
        # at the tail of the last shuffle block.
        total_micro = CFG["train_steps"] * accum_steps
        schedule: list[str] = []
        while len(schedule) < total_micro:
            block = train_langs.copy()
            random.shuffle(block)
            schedule.extend(block)
        schedule = schedule[:total_micro]

        accum_loss      = 0.0
        accum_trans     = 0.0
        accum_ner_count = 0
        accum_ner_sum   = 0.0
        langs_in_step   = []

        for micro_step, lang in enumerate(schedule, 1):
            if stop:
                break
            langs_in_step.append(lang)
            srcs, tgts = next(train_iters[lang])

            ner_tags = None
            if nlp is not None:
                try:
                    ner_tags = _extract_ner_tags(
                        src_texts=srcs,
                        target_lang=lang,
                        nlp=nlp,
                        tokenizer=model.srcTokenizer,  # NER aligns with source tokens
                        ner_vocab=model.nerVocab,
                        max_length=128,
                        device=device,
                    )
                except Exception as e:
                    print(f"  [NER ]  WARNING: {e} — skipping nerTags this step")

            with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                out  = model(
                    srcText=srcs,
                    targetLang=lang,
                    targetText=tgts,
                    nerTags=ner_tags,
                    returnLoss=True,
                    device=device,
                )
                loss = out["loss"] / accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            # FIX 6: Accumulate trans/ner losses across micro-steps so the log
            # line reports the average over the whole effective batch, not just
            # the last micro-step.
            if out["translationLoss"] is not None:
                accum_trans += out["translationLoss"].item()
            if out["nerLoss"] is not None:
                accum_ner_sum   += out["nerLoss"].item()
                accum_ner_count += 1

            if micro_step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step    += 1
                train_step     += 1
                train_loss_sum += accum_loss

                avg_trans = accum_trans / accum_steps
                avg_ner   = (accum_ner_sum / accum_ner_count) if accum_ner_count else None
                langs_str = "+".join(langs_in_step)

                print(
                    f"  [TRAIN]"
                    f"  global={global_step:>7}"
                    f"  cycle={cycle}  step={train_step:>3}/{CFG['train_steps']}"
                    f"  langs=[{langs_str}]"
                    f"  loss={accum_loss:.4f}"
                    f"  trans={avg_trans:.4f}"
                    + (f"  ner={avg_ner:.4f}" if avg_ner is not None else "")
                    + f"  w_trans={out['weight_trans']:.3f}"
                    + f"  w_ner={out['weight_ner']:.3f}"
                    + f"  lr={optimizer.param_groups[0]['lr']:.2e}"
                )

                accum_loss      = 0.0
                accum_trans     = 0.0
                accum_ner_sum   = 0.0
                accum_ner_count = 0
                langs_in_step   = []

                if time.time() - last_ckpt >= ckpt_sec:
                    save_checkpoint(model, optimizer, scheduler, scaler, global_step, cycle)
                    last_ckpt = time.time()

                if global_step % 50 == 0 and device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

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

        test_schedule = []
        while len(test_schedule) < CFG["test_steps"]:
            block = test_langs.copy()
            random.shuffle(block)
            test_schedule.extend(block)
        test_schedule = test_schedule[: CFG["test_steps"]]

        with torch.no_grad():
            for test_step, lang in enumerate(test_schedule, 1):
                srcs, tgts = next(test_iters[lang])

                ner_tags = None
                if nlp is not None:
                    try:
                        ner_tags = _extract_ner_tags(
                            src_texts=srcs,
                            target_lang=lang,
                            nlp=nlp,
                            tokenizer=model.srcTokenizer,  # NER aligns with source tokens
                            ner_vocab=model.nerVocab,
                            max_length=128,
                            device=device,
                        )
                    except Exception:
                        pass

                with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    out = model(
                        srcText=srcs,
                        targetLang=lang,
                        targetText=tgts,
                        nerTags=ner_tags,
                        returnLoss=True,
                        device=device,
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
            gc.collect()

        n         = CFG["test_steps"]
        avg_test  = test_loss_sum  / n
        avg_trans = test_trans_sum / n
        avg_ner   = test_ner_sum   / n
        print(
            f"\n  [CYCLE {cycle}]  ── Test done ──  "
            f"avg loss: {avg_test:.4f}  trans: {avg_trans:.4f}  ner: {avg_ner:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

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

    nlp = _load_spacy("xx_ent_wiki_sm")

    print("  Fetching data from HuggingFace OPUS-100...\n")
    train_by_lang, test_by_lang = fetch_all_pairs()

    model = MTT()
    model.paramsCalc()

    train(model, train_by_lang, test_by_lang, nlp)