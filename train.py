"""Train the character Transformer. Importing this module does not start training."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

if __package__:
    from .model import Config, SelfAttention, Transformer
    from . import runtime
else:
    from model import Config, SelfAttention, Transformer
    import runtime


ROOT = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    max_iters: int = 2000
    batch_size: int = 12
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    decay_lr: bool = True
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 20
    log_interval: int = 100
    seed: int = 1337
    device: str = "auto"
    data_dir: str | Path | None = None
    out_dir: str | Path | None = None
    num_threads: int | None = None
    init_from: str = "scratch"
    resume_checkpoint: str | Path | None = None

    def __post_init__(self):
        for name in ("max_iters", "batch_size", "block_size", "n_layer", "n_head", "n_embd",
                     "eval_interval", "eval_iters", "log_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_head must evenly divide n_embd")
        if self.warmup_iters < 0 or self.grad_clip < 0 or self.weight_decay < 0:
            raise ValueError("warmup_iters, grad_clip, and weight_decay must be nonnegative")
        if not 0 <= self.dropout < 1 or not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Invalid dropout or AdamW beta values")
        if not 0 < self.min_lr <= self.learning_rate or not math.isfinite(self.learning_rate):
            raise ValueError("Require finite 0 < min_lr <= learning_rate")
        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError("num_threads must be positive")
        if self.init_from not in ("scratch", "resume"):
            raise ValueError("init_from must be 'scratch' or 'resume'")


def _learning_rate(update_index, config):
    if not config.decay_lr:
        return config.learning_rate
    # update_index is zero based. The first actual update uses a nonzero learning rate.
    warmup = min(config.warmup_iters, max(0, config.max_iters - 1))
    if update_index < warmup:
        return config.learning_rate * (update_index + 1) / warmup
    denominator = max(1, config.max_iters - 1 - warmup)
    ratio = min(1.0, max(0.0, (update_index - warmup) / denominator))
    return config.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (config.learning_rate - config.min_lr)


def _serializable_config(config):
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def train(config=None):
    """Perform exactly config.max_iters total updates and return artifact paths/history.

    When resuming, max_iters is the desired final update count. ckpt.pt stores the
    lowest validation loss, while last.pt always stores the final completed update.
    Evaluation occurs before any update (step 0), at each interval, and at the end.
    """
    config = TrainConfig() if config is None else config
    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    environment = runtime.validate_execution_environment(config.device)
    runtime.require_complete_attention(SelfAttention)
    device = environment["device"]
    threads = config.num_threads or environment.get("cpu_threads", 1)
    if environment.get("is_cs_cluster") and threads > environment.get("cpu_threads", 1):
        raise ValueError("num_threads exceeds the CPUs allowed by the Slurm allocation")
    torch.set_num_threads(threads)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    data_dir, metadata = runtime.load_metadata(config.data_dir)
    data = {}
    for split in ("train", "val"):
        path = data_dir / f"{split}.bin"
        if path.stat().st_size % np.dtype(np.uint16).itemsize:
            raise ValueError(f"{path} is not a valid uint16 token file")
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        if len(tokens) <= config.block_size:
            raise ValueError(f"{split}.bin needs more than block_size={config.block_size} tokens")
        if int(tokens.max()) >= metadata["vocab_size"]:
            raise ValueError(f"{split}.bin contains a token outside the metadata vocabulary")
        data[split] = tokens

    batch_rng = torch.Generator().manual_seed(config.seed + 1)
    eval_rng = torch.Generator().manual_seed(config.seed + 2)

    def get_batch(split, generator):
        tokens = data[split]
        starts = torch.randint(len(tokens) - config.block_size, (config.batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(np.array(tokens[i:i + config.block_size], dtype=np.int64)) for i in starts.tolist()])
        y = torch.stack([torch.from_numpy(np.array(tokens[i + 1:i + config.block_size + 1], dtype=np.int64)) for i in starts.tolist()])
        return x.to(device), y.to(device)

    model_args = dict(
        n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
        block_size=config.block_size, vocab_size=metadata["vocab_size"],
        dropout=config.dropout, bias=config.bias,
    )
    model = Transformer(Config(**model_args)).to(device)
    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device)
    out_dir = Path(config.out_dir).expanduser().resolve() if config.out_dir is not None else ROOT / "checkpoints"
    history = []
    best_val_loss = float("inf")
    start_step = 0
    if config.init_from == "resume":
        resume_path = Path(config.resume_checkpoint) if config.resume_checkpoint is not None else out_dir / "last.pt"
        saved = torch.load(resume_path, map_location=device, weights_only=True)
        if saved["model_args"] != model_args:
            raise ValueError("Resume checkpoint model_args must match this TrainConfig and vocabulary")
        state = {name.removeprefix("_orig_mod."): value for name, value in saved["model"].items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"] if "step" in saved else saved["iter_num"])
        if start_step >= config.max_iters:
            raise ValueError("Resume max_iters must exceed the number of updates already completed")
        history = list(saved.get("history", []))
        best_val_loss = float(saved["best_val_loss"])
        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"].cpu())
        if "batch_rng_state" in saved:
            batch_rng.set_state(saved["batch_rng_state"].cpu())
        if "eval_rng_state" in saved:
            eval_rng.set_state(saved["eval_rng_state"].cpu())
        if device == "cuda" and "cuda_rng_state" in saved:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
        # Evaluation RNG restoration is for future evaluations; reuse the existing history point.

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out_dir / "ckpt.pt", out_dir / "last.pt"
    started = time.monotonic()

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}
        for split in ("train", "val"):
            values = []
            for _ in range(config.eval_iters):
                x, y = get_batch(split, eval_rng)
                _, loss = model(x, y)
                values.append(float(loss.item()))
            losses[split] = sum(values) / len(values)
        model.train()
        if not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError("Evaluation loss is non-finite")
        return losses

    def save_checkpoint(path, step):
        saved = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_args": model_args, "step": step, "iter_num": step,
            "best_val_loss": float(best_val_loss), "history": list(history),
            "config": _serializable_config(config),
            "torch_rng_state": torch.get_rng_state(),
            "batch_rng_state": batch_rng.get_state(), "eval_rng_state": eval_rng.get_state(),
        }
        if device == "cuda":
            saved["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(saved, path)

    def evaluate(step):
        nonlocal best_val_loss
        losses = estimate_loss()
        row = {"step": step, "train_loss": losses["train"], "val_loss": losses["val"]}
        if history and history[-1]["step"] == step:
            history[-1] = row
        else:
            history.append(row)
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            save_checkpoint(best_path, step)

    print(f"Training on {environment['hostname']} ({device}); {config.max_iters - start_step} updates", flush=True)
    if start_step == 0:
        evaluate(0)
    elif not best_path.exists():
        # A resumed run can use a new output directory; initialize its best artifact.
        # The checkpoint's stored model is available, but the earlier best model may not be.
        best_val_loss = float("inf")
        evaluate(start_step)
    for update_index in range(start_step, config.max_iters):
        lr = _learning_rate(update_index, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch("train", batch_rng)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss before update {update_index + 1}")
        loss.backward()
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        step = update_index + 1
        if step % config.log_interval == 0:
            print(f"update {step}/{config.max_iters}: loss {loss.item():.4f}, lr {lr:.6g}", flush=True)
        if step % config.eval_interval == 0 or step == config.max_iters:
            evaluate(step)
    save_checkpoint(last_path, config.max_iters)

    csv_path, json_path, curve_path = out_dir / "history.csv", out_dir / "history.json", out_dir / "loss_curve.png"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "train_loss", "val_loss"))
        writer.writeheader()
        writer.writerows(history)
    json_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["step"] for row in history], [row["train_loss"] for row in history], marker="o", label="Training")
    axis.plot([row["step"] for row in history], [row["val_loss"] for row in history], marker="o", label="Validation")
    axis.set(xlabel="Optimizer updates", ylabel="Cross-entropy loss", title="Character Transformer loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(curve_path, dpi=150)
    plt.close(figure)
    return {
        "history": history, "checkpoint": str(best_path), "last_checkpoint": str(last_path),
        "loss_curve": str(curve_path), "history_csv": str(csv_path), "history_json": str(json_path),
        "model_args": model_args, "environment": environment,
        "num_updates": config.max_iters - start_step,
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("max_iters", 2000), ("batch_size", 12), ("block_size", 64),
                          ("n_layer", 4), ("n_head", 4), ("n_embd", 128), ("eval_interval", 250),
                          ("eval_iters", 20), ("log_interval", 100), ("warmup_iters", 100), ("seed", 1337)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in (("learning_rate", 1e-3), ("min_lr", 1e-4), ("weight_decay", 0.1),
                          ("dropout", 0.0), ("grad_clip", 1.0), ("beta1", 0.9), ("beta2", 0.99)):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decay-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-from", choices=("scratch", "resume"), default="scratch")
    parser.add_argument("--resume-checkpoint", type=Path)
    result = train(TrainConfig(**vars(parser.parse_args())))
    print(f"Best checkpoint: {result['checkpoint']}\nLast checkpoint: {result['last_checkpoint']}\nLoss curve: {result['loss_curve']}")


if __name__ == "__main__":
    main()
