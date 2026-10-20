import argparse
from dataclasses import fields
from pathlib import Path

import torch

from xn.config import Config
from xn.doctor import doctor
from xn.train import load_checkpoint, train

CONFIG_FILE = "xn.json"


def load_config(path: str | None) -> Config:
    if path:
        return Config.load(path)
    return Config.load(CONFIG_FILE) if Path(CONFIG_FILE).exists() else Config()


def main() -> None:
    # engine entry point; normally invoked by the native `xn` binary (cli/) through `uv run`
    parser = argparse.ArgumentParser(prog="xn", description="XN - Eazy make a LLM.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check the engine's python, dependencies and the project")

    p_data = sub.add_parser("data", help="fetch and inspect training data in ./data")
    data_sub = p_data.add_subparsers(dest="data_cmd", required=True)
    p_add = data_sub.add_parser("add", help="download URLs (web pages, text, Aozora Bunko zip)")
    p_add.add_argument("urls", nargs="+", help="URLs or local files")
    p_add.add_argument("--name", help="file name under data/ (single URL only)")
    p_add.add_argument("--force", action="store_true", help="overwrite existing files")
    data_sub.add_parser("list", help="show data files and corpus size")

    p_train = sub.add_parser("train", help="train a model from scratch")
    p_train.add_argument("--config", help=f"path to JSON config (default: ./{CONFIG_FILE})")
    for f in fields(Config):
        if f.name == "content_loader":
            continue  # structured fields (content_loader) are set in the JSON config only
        p_train.add_argument(f"--{f.name}", type=type(f.default), default=None)

    p_gen = sub.add_parser("generate", help="generate text from a checkpoint")
    p_gen.add_argument("--ckpt", help="checkpoint (default: <out_dir>/ckpt.pt from ./xn.json)")
    p_gen.add_argument(
        "--prompt", default="", help="trailing whitespace is stripped (see README)"
    )
    p_gen.add_argument("--max_new_tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top_k", type=int, default=50)
    p_gen.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="> 1 discourages repeating recent tokens; 1 disables")
    p_gen.add_argument("--repetition_window", type=int, default=64,
                       help="how many recent tokens the penalty looks at")

    args = parser.parse_args()
    if args.cmd == "doctor":
        raise SystemExit(doctor())
    if args.cmd == "data":
        from xn import data

        if args.data_cmd == "add":
            data.add(args.urls, args.name, args.force)
        else:
            data.list_data()
    elif args.cmd == "train":
        cfg = load_config(args.config)
        for f in fields(Config):
            if (v := getattr(args, f.name, None)) is not None:
                setattr(cfg, f.name, v)
        train(cfg)
    else:
        model, tok = load_checkpoint(args.ckpt or Path(load_config(None).out_dir) / "ckpt.pt")
        # trailing newline/indent would be split from the whitespace token the model learned
        # (e.g. "\n    " is one token), so let the model produce it itself
        ids = tok.encode(args.prompt.rstrip() or "\n")
        out = model.generate(
            torch.tensor([ids]), args.max_new_tokens, args.temperature, args.top_k,
            args.repetition_penalty, args.repetition_window,
        )
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
