import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

from xn.loader import DEFAULT_LOADER, load_content, load_documents


@dataclass
class Config:
    # tokenizer: "byte" (vocab 256) or "bpe" (trained on the content_loader corpus, vocab `vocab_size`)
    tokenizer: str = "bpe"
    # model
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    # training
    # where training text comes from; see xn/loader.py
    content_loader: dict = field(default_factory=lambda: deepcopy(DEFAULT_LOADER))
    out_dir: str = "runs/default"
    batch_size: int = 16
    max_steps: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5  # cosine decay floor
    warmup_steps: int = 100  # linear LR warmup
    weight_decay: float = 0.1
    # fraction of documents held out for validation (token tail split if there is only one)
    val_fraction: float = 0.1
    eval_interval: int = 100
    eval_steps: int = 20
    seed: int = 1337
    device: str = "cpu"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        d = json.loads(Path(path).read_text())
        if "data" in d:
            raise SystemExit(
                f'{path}: "data" was removed; use "content_loader" instead, e.g. '
                f'"content_loader": {{"sources": [{{"path": "{d["data"]}"}}]}}'
            )
        return cls(**d)

    def load_text(self) -> str:
        return load_content(self.content_loader)

    def load_docs(self) -> list[str]:
        return load_documents(self.content_loader)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
