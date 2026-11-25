#!/usr/bin/env python3
"""Fine-tune a SMALL model as the MEMO MEMORY model (LoRA SFT on reflection QA).

Demo-scale replica of MEMO (arXiv 2605.15156):
  paper: MEMORY = Qwen2.5-14B-Instruct, SFT 3 epochs on ~1.3M reflection QA pairs, 8xH100.
  demo : MEMORY = Qwen2.5-0.5B-Instruct, LoRA, ~300-500 reflection QA pairs, CPU.

Honest limits: this machine has no GPU. Training runs on CPU with a strict
wall-clock time-box (--max-seconds). Use --time-test first to measure seconds/step
and extrapolate before committing to a full run.

Usage:
  .venv/bin/python memo/train_memory.py --time-test --time-test-steps 3
  .venv/bin/python memo/train_memory.py --max-examples 150 --epochs 1 --max-seconds 2700
"""
import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="memo/reflection_qa.jsonl")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", default="memo/memory_model")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-examples", type=int, default=0, help="0 = all")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seconds", type=float, default=2700,
                   help="wall-clock cap; training stops and saves when exceeded")
    p.add_argument("--time-test", action="store_true",
                   help="run N steps, report s/step + ETA, do not save")
    p.add_argument("--time-test-steps", type=int, default=3)
    p.add_argument("--threads", type=int, default=0, help="0 = all cores")
    return p.parse_args()


class QADataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        skipped = 0
        for r in rows:
            prompt = (f"<|im_start|>user\n{r['question']}<|im_end|>\n"
                      f"<|im_start|>assistant\n")
            full = prompt + r["answer"] + "<|im_end|>"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(f_ids) > max_len:
                skipped += 1
                continue
            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
            self.items.append((f_ids, labels))
        print(f"dataset: {len(self.items)} examples, {skipped} skipped (> {max_len} tokens)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    t_start = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(l) for l in open(args.data)]
    random.shuffle(rows)
    if args.max_examples:
        rows = rows[: args.max_examples]
    print(f"training on {len(rows)} examples from {args.data}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"loading {args.model} (bf16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    ds = QADataset(rows, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    total_opt_steps = math.ceil(len(ds) / (args.batch_size * args.grad_accum))
    total_opt_steps = max(1, int(total_opt_steps * args.epochs))
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(
        opt, int(0.1 * total_opt_steps), total_opt_steps)

    model.train()
    step = 0
    micro = 0
    losses = []
    log = {"model": args.model, "examples": len(ds), "epochs": args.epochs,
           "lora_r": args.lora_r, "lr": args.lr, "max_len": args.max_len,
           "batch_size": args.batch_size, "grad_accum": args.grad_accum,
           "optimizer_steps": 0, "micro_steps": 0, "loss_mean": None,
           "seconds": None, "stopped_by_timebox": False}
    loss_acc = 0.0
    stop = False
    step_times = []

    def save_ckpt(tag):
        d = os.path.join(args.out + "_ckpt", f"step_{tag}")
        os.makedirs(d, exist_ok=True)
        model.save_pretrained(d)
        tok.save_pretrained(d)
        print(f"checkpoint -> {d}", flush=True)
    while not stop:
        for batch in dl:
            if stop:
                break
            t0 = time.time()
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            loss_acc += out.loss.item()
            micro += 1
            step_times.append(time.time() - t0)
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                losses.append(loss_acc / args.grad_accum)
                loss_acc = 0.0
                if step % 5 == 0 or step == 1:
                    el = time.time() - t_start
                    print(f"opt step {step}/{total_opt_steps} "
                          f"loss {losses[-1]:.4f} elapsed {el:.0f}s", flush=True)
                if step % 25 == 0:
                    save_ckpt(step)
                if args.time_test and step >= args.time_test_steps:
                    s = sum(step_times) / len(step_times)
                    print(f"\nTIME TEST: {s:.1f} s/micro-step "
                          f"(batch {args.batch_size} x seq<={args.max_len})")
                    print(f"ETA full run ({len(ds)} ex, {args.epochs} ep): "
                          f"~{s * len(ds) * args.epochs / 60:.0f} min "
                          f"(micro-steps: {len(ds) * args.epochs:.0f} x {s:.1f}s)")
                    return
                if time.time() - t_start > args.max_seconds:
                    print(f"TIMEBOX hit ({args.max_seconds:.0f}s) at opt step {step}; saving.")
                    log["stopped_by_timebox"] = True
                    stop = True
                    break
                if step >= total_opt_steps:
                    stop = True
                    break
        if not args.time_test and not stop:
            # next epoch: reshuffle via new DataLoader order is handled by shuffle=True
            pass
        if args.time_test:
            break
    log["optimizer_steps"] = step
    log["micro_steps"] = micro
    log["loss_mean"] = sum(losses) / len(losses) if losses else None
    log["seconds"] = time.time() - t_start
    log["loss_last10_mean"] = (sum(losses[-10:]) / len(losses[-10:])
                               if losses else None)
    if not args.time_test:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        with open(os.path.join(args.out, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"\nsaved adapter -> {args.out}")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
