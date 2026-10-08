"""Generate new characters from a fresh model or an explicit trained checkpoint."""

import argparse
from pathlib import Path
import sys

import torch

if __package__:
    from .model import Config, SelfAttention, Transformer
    from . import runtime
else:
    from model import Config, SelfAttention, Transformer
    import runtime


ROOT = Path(__file__).resolve().parent


def sample(*, init_from="scratch", checkpoint=None, start="\n", max_new_tokens=500,
           device="auto", seed=1337, temperature=1.0, top_k=None, data_dir=None,
           model_args=None):
    """Return prompt, continuation, and combined text without printing the prompt.

    scratch deliberately uses no checkpoint, including for the unfinished student's
    part (a). resume requires completed attention and uses ckpt.pt unless a path is given.
    A character is one token, so continuation contains exactly max_new_tokens characters.
    """
    if init_from not in ("scratch", "resume"):
        raise ValueError("init_from must be 'scratch' or 'resume'")
    if checkpoint is not None and init_from != "resume":
        raise ValueError("To use a checkpoint, set init_from='resume' explicitly")
    if not isinstance(start, str) or not start:
        raise ValueError("start must be a nonempty string")
    environment = runtime.validate_execution_environment(device)
    device = environment["device"]
    _, metadata = runtime.load_metadata(data_dir)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    if start.startswith("FILE:"):
        start = Path(start[5:]).read_text(encoding="utf-8")
        if not start:
            raise ValueError("The prompt file must be nonempty")
    unknown = sorted(set(start) - metadata["stoi"].keys())
    if unknown:
        raise ValueError(f"Prompt contains characters outside this vocabulary: {unknown!r}")

    checkpoint_path = None
    if init_from == "scratch":
        arguments = dict(n_layer=4, n_head=4, n_embd=128, block_size=64, dropout=0.0, bias=False)
        arguments.update(model_args or {})
        if "vocab_size" in arguments and arguments["vocab_size"] != metadata["vocab_size"]:
            raise ValueError("model_args.vocab_size must match the data metadata")
        arguments["vocab_size"] = metadata["vocab_size"]
        model = Transformer(Config(**arguments))
    else:
        runtime.require_complete_attention(SelfAttention)
        if model_args is not None:
            raise ValueError("resume uses the model_args saved in its checkpoint")
        checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint is not None else ROOT / "checkpoints" / "ckpt.pt"
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        arguments = dict(saved["model_args"])
        if arguments["vocab_size"] != metadata["vocab_size"]:
            raise ValueError("Checkpoint vocabulary size does not match the dataset")
        model = Transformer(Config(**arguments))
        state = {name.removeprefix("_orig_mod."): value for name, value in saved["model"].items()}
        model.load_state_dict(state)
    model = model.to(device).eval()
    prompt_ids = [metadata["stoi"][character] for character in start]
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    continuation = "".join(metadata["itos"][token] for token in y[0, len(prompt_ids):].tolist())
    return {
        "prompt": start, "continuation": continuation, "text": start + continuation,
        "init_from": init_from, "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "model_args": arguments, "device": device, "environment": environment,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", choices=("scratch", "resume"), default="scratch")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--start", default="\n", help="Prompt text, or FILE:path to read a UTF-8 prompt")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--data-dir", type=Path)
    options = vars(parser.parse_args())
    if options["checkpoint"] is not None:
        options["init_from"] = "resume"
    result = sample(**options)
    print(f"[{result['init_from']}; {result['device']}; new characters only]", file=sys.stderr)
    print(result["continuation"])


if __name__ == "__main__":
    main()
