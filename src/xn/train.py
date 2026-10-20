import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from xn.config import Config
from xn.model import GPT
from xn.tokenizer import BPETokenizer, ByteTokenizer, tokenizer_from_dict


def build_tokenizer(cfg: Config, text: str) -> ByteTokenizer | BPETokenizer:
    if cfg.tokenizer == "byte":
        cfg.vocab_size = 256
        return ByteTokenizer()
    if cfg.tokenizer != "bpe":
        raise ValueError(f"unknown tokenizer: {cfg.tokenizer}")
    print(f"training BPE tokenizer (vocab {cfg.vocab_size})...")
    tok = BPETokenizer.train(text, cfg.vocab_size, verbose=True)
    cfg.vocab_size = tok.vocab_size
    return tok


def split_docs(docs: list[str], cfg: Config) -> tuple[str, str | None]:
    """Hold out whole documents for validation so train and val never share text."""
    sep = cfg.content_loader.get("separator", "\n\n")
    n_val = round(len(docs) * cfg.val_fraction)
    if len(docs) < 2 or n_val == 0:
        return sep.join(docs), None
    return sep.join(docs[:-n_val]), sep.join(docs[-n_val:])


def load_data(train_text: str, val_text: str | None, tok, cfg: Config):
    train = torch.tensor(tok.encode(train_text), dtype=torch.long)
    if val_text is not None:
        val = torch.tensor(tok.encode(val_text), dtype=torch.long)
        if len(val) > cfg.block_size + 1:
            return train, val
        print("warning: held-out documents are shorter than block_size; using a token split")
    split = int(len(train) * (1 - cfg.val_fraction))
    return train[:split], train[split:]


def lr_at(step: int, cfg: Config) -> float:
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def get_batch(data: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + cfg.block_size] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


@torch.no_grad()
def evaluate(model: GPT, data: torch.Tensor, cfg: Config) -> float:
    model.eval()
    losses = [model(*get_batch(data, cfg))[1].item() for _ in range(cfg.eval_steps)]
    model.train()
    return sum(losses) / len(losses)


def train(cfg: Config) -> GPT:
    torch.manual_seed(cfg.seed)
    docs = cfg.load_docs()
    train_text, val_text = split_docs(docs, cfg)
    tok = build_tokenizer(cfg, train_text)  # never sees validation text
    train_data, val_data = load_data(train_text, val_text, tok, cfg)
    model = GPT(cfg).to(cfg.device)
    # no weight decay on biases / LayerNorm (1-D params)
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"params: {model.num_params() / 1e6:.2f}M  docs: {len(docs)}  "
        f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}"
    )

    best = float("inf")
    t0 = time.time()
    for step in range(1, cfg.max_steps + 1):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        _, loss = model(*get_batch(train_data, cfg))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            val = evaluate(model, val_data, cfg)
            tps = step * cfg.batch_size * cfg.block_size / (time.time() - t0)
            print(
                f"step {step:5d} | train {loss.item():.4f} | val {val:.4f} "
                f"| lr {lr:.2e} | {tps:,.0f} tok/s"
            )
            if val < best:
                best = val
                ckpt = {"config": asdict(cfg), "model": model.state_dict(), "tokenizer": tok.to_dict()}
                torch.save(ckpt, out / "ckpt.pt")
    return model


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GPT, ByteTokenizer | BPETokenizer]:
    ckpt = torch.load(path, map_location=device)
    model = GPT(Config(**ckpt["config"]))
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        raise SystemExit(f"{path}: incompatible checkpoint (made by an older XN?) - retrain it") from e
    tok = tokenizer_from_dict(ckpt.get("tokenizer", {"type": "byte"}))
    return model.to(device).eval(), tok
