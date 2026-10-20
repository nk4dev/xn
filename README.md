# XN - Eazy make a LLM.

XN is a small toolkit for training GPT-style language models from scratch on a CPU (or a GPU, if you have one), then generating text from them.

- **`xn`** is a native command-line tool written in Rust. It creates projects and checks your setup on its own.
- **The engine** (`src/xn`) is written in Python with PyTorch and does the training and generation. `xn` runs it through [uv](https://docs.astral.sh/uv/), so you never manage a Python environment yourself.

No GPU or no setup? [Open the sample notebook in Google Colab](https://colab.research.google.com/github/nk4dev/xn/blob/main/notebooks/xn_colab.ipynb) ([`notebooks/xn_colab.ipynb`](notebooks/xn_colab.ipynb)): it installs XN, fetches data, trains on a free GPU and keeps checkpoints on Google Drive so `--resume` works across sessions.

Documentation: https://apps.nknighta.me/xn/ (source in [`docs/`](docs/README.md)).

## Requirements

- [Rust](https://rustup.rs/) (cargo), to build `xn`
- [uv](https://docs.astral.sh/uv/), to run the engine. uv installs Python and PyTorch for you; Python 3.11 or newer is needed.

## Install

From the root of this repository:

```sh
curl -fsSL https://apps.nknighta.me/xn/install.sh | sh
# or, from a checkout of this repository:
./install.sh
```

The script:

1. Checks for uv and cargo. If either is missing, it stops and tells you how to get it. Pass `--install-deps` to install them with their official installers instead.
2. Installs the engine's dependencies with `uv sync` (the CUDA build of PyTorch when an NVIDIA GPU is found, else the CPU build; override with `XN_TORCH=cpu|cuda`).
3. Builds `xn` and installs it to `~/.cargo/bin`. Use `--bin-dir ~/.local/bin` to choose another directory; it must end in `/bin`.
4. Warns if that directory is not on your `PATH`, then runs `xn doctor`.

Running it again rebuilds and replaces `xn`.

When piped from curl, it clones https://github.com/nk4dev/xn.git (override with `XN_REPO`) into `~/.local/share/xn`, or `$XN_HOME`, and updates that clone on later runs. Pass options after `sh -s --`, e.g. `curl -fsSL https://apps.nknighta.me/xn/install.sh | sh -s -- --install-deps`.

`xn` remembers where its source was when it was built and runs the engine from there. If you move that directory, run `./install.sh` again, or set `XN_ENGINE` to the new location.

To do the same steps by hand: `uv sync` (or `uv sync --no-default-groups --group dev --group cuda` for an NVIDIA GPU), then `cargo install --path cli`.

## Quick start

```sh
xn init myllm               # create a project
cd myllm
xn train                    # train with the settings in ./xn.json
xn generate --prompt "To be"
```

`xn init` writes:

```
myllm/
├── xn.json          # project settings
├── data/sample.txt  # sample training text (English and Japanese)
└── .gitignore       # ignores runs/
```

Training prints train/validation loss, the learning rate and throughput (tokens per second), and saves the best checkpoint (lowest validation loss) to `runs/default/ckpt.pt`. The default settings take a few minutes on a CPU; for a fast smoke test, run `xn train --max_steps 100`.

## Commands

| Command | Description |
|---|---|
| `xn init [dir] [--preset NAME]` | Create a project in `dir` (default: current directory). Fails if `xn.json` already exists. `--preset xn-llm` (small text model) or `xn-llm-code` (small JS/TS code model) writes that preset's settings instead of the defaults. |
| `xn data add URL... [--name FILE] [--force]` | Download training text into `data/` (see below). |
| `xn data repo OWNER/NAME [--include GLOBS] [--ref REF] [--max_kb N] [--force]` | Download a GitHub repo's source files (default `*.js,*.ts`) into `data/OWNER_NAME/`. Only repos with a permissive license (MIT, Apache-2.0, BSD, ISC, ...) are accepted; the repo, commit and license are recorded in `data/SOURCES.json`. |
| `xn data list` | Show files in `data/` and how much text `content_loader` will load. |
| `xn doctor` | Check Python, uv, the engine, PyTorch, the device, and `./xn.json` with its training data. Exits with status 1 if anything is wrong. |
| `xn train [options]` | Train a model from scratch. |
| `xn generate [options]` | Generate text from a trained model. |
| `xn --help`, `xn --version` | Show help or the version. |

Run `train` and `generate` from your project directory: both read `./xn.json`.

### `xn data`

`xn data add` downloads each URL (or copies a local file) into `data/` as UTF-8 text, ready for the default `content_loader`:

- **Web pages**: scripts and styles are dropped and only the visible text is kept.
- **Plain text**: Shift_JIS and EUC-JP are converted to UTF-8.
- **[Aozora Bunko](https://www.aozora.gr.jp/)** zip files: the header, footer, ruby readings (`《》`) and editor notes (`［＃］`) are removed.

```sh
xn data add https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip   # Botchan
xn data add https://example.com/article.html --name article.txt
xn data list
```

Existing files are skipped unless you pass `--force`.

For code, `xn data repo` downloads a GitHub repository's source files after checking its license. Copyleft (GPL, ...) and unlicensed repositories are refused, so the model you train can be shared; keep `data/SOURCES.json` (and each `LICENSE.upstream`) with it for attribution. Set `GITHUB_TOKEN` if you hit the API rate limit.

```sh
xn init coder --preset xn-llm-code && cd coder
xn data repo expressjs/express
xn data repo vuejs/core --include "*.ts"
```

### `xn train`

| Option | Description |
|---|---|
| `--config FILE` | Use a different settings file instead of `./xn.json`. |
| `--resume` | Continue an interrupted run from `<out_dir>/last.pt` (saved at every evaluation and when stopped with Ctrl-C or `kill`). Model, tokenizer and data settings come from the saved run; `--max_steps` may be raised to train longer. |
| `--<setting> VALUE` | Override any setting in `xn.json` for this run, for example `--max_steps 2000 --lr 0.001`. `content_loader` can only be set in the file. |

### `xn generate`

| Option | Default | Description |
|---|---|---|
| `--prompt TEXT` | empty | Text for the model to continue. Trailing spaces and newlines are removed, because the model learned a line break and the next line's indentation as one token; it adds them itself. |
| `--max_new_tokens N` | 200 | Number of tokens to generate. |
| `--temperature T` | 0.8 | Lower is more predictable; higher is more varied. |
| `--top_k K` | 50 | Sample only from the K most likely tokens. |
| `--repetition_penalty P` | 1.1 | Values above 1 make tokens from the last `--repetition_window` tokens less likely, which reduces loops. 1 turns it off. |
| `--repetition_window N` | 64 | How many recent tokens the repetition penalty looks at. |
| `--ckpt FILE` | `<out_dir>/ckpt.pt` | Checkpoint to load. The model's tokenizer is stored inside it. |

For a model that writes new text instead of repeating the sample, give it more data, for example a few Aozora Bunko novels (about 600,000 characters):

```sh
xn data add https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip \
            https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip
xn train --max_steps 1500 --batch_size 32
```

Checkpoints made before XN switched to its current attention layer cannot be loaded; train them again.

## Settings (`xn.json`)

Default file created by `xn init`:

```json
{
  "tokenizer": "bpe",
  "vocab_size": 1024,
  "block_size": 128,
  "n_layer": 4,
  "n_head": 4,
  "n_embd": 128,
  "dropout": 0.1,
  "content_loader": {
    "separator": "\n\n",
    "sources": [{ "path": "data/*.txt", "format": "text" }]
  },
  "out_dir": "runs/default",
  "batch_size": 16,
  "max_steps": 1000,
  "lr": 0.0003,
  "min_lr": 3e-05,
  "warmup_steps": 100,
  "weight_decay": 0.1,
  "val_fraction": 0.1,
  "eval_interval": 100,
  "eval_steps": 20,
  "seed": 1337,
  "device": "auto"
}
```

| Setting | Description |
|---|---|
| `tokenizer` | `"bpe"`: learn a byte-level BPE vocabulary from your training data at the start of training. `"byte"`: use raw UTF-8 bytes (vocabulary of 256). |
| `vocab_size` | BPE vocabulary size. Ignored for `"byte"`. |
| `block_size` | Context length in tokens. |
| `n_layer`, `n_head`, `n_embd` | Number of Transformer layers, attention heads, and embedding size. |
| `dropout` | Dropout rate. |
| `content_loader` | Where the training text comes from (see below). |
| `out_dir` | Where checkpoints are saved. |
| `batch_size`, `max_steps` | Batch size and number of training steps. |
| `lr`, `min_lr`, `warmup_steps` | The learning rate rises linearly to `lr` over `warmup_steps`, then follows a cosine curve down to `min_lr` at `max_steps`. |
| `weight_decay` | AdamW weight decay (not applied to biases and LayerNorm). |
| `val_fraction` | Share held out for validation. Whole documents are held out, so training and validation never share text. With too few documents, the last part of the corpus is used instead. |
| `eval_interval`, `eval_steps` | Evaluate every N steps, averaging over this many batches. |
| `seed` | Random seed. |
| `device` | `"auto"` uses an NVIDIA GPU (CUDA) or Apple GPU (MPS) when available, else the CPU. Or name a torch device: `"cpu"`, `"cuda"`, `"cuda:1"`, `"mps"`. |

The `data` setting from earlier versions has been removed. If `xn.json` still contains it, XN stops and shows how to rewrite it as `content_loader`.

### Training data (`content_loader`)

Every document from every source is joined with `separator` into one corpus. Exact duplicate documents are loaded only once. Files must be UTF-8; any language works.

```json
"content_loader": {
  "separator": "\n\n",
  "sources": [
    { "path": "data/*.txt", "format": "text" },
    { "path": "data/chat.jsonl", "format": "jsonl", "field": "text" },
    { "path": "data/qa.csv", "format": "csv", "field": "answer" },
    { "path": "corpus/" }
  ]
}
```

| Key | Description |
|---|---|
| `path` | A file, a glob pattern (`**` matches subdirectories), or a directory (every file inside, recursively). Relative to the project directory. |
| `format` | `text` (default): the whole file is one document. `jsonl`: one document per line, read from `field`. `csv`: one document per row, read from the column `field`. |
| `field` | JSON key or CSV column to read. Default: `text`. |

## Environment variables

| Variable | Description |
|---|---|
| `XN_ENGINE` | Path to the engine (the directory containing `pyproject.toml`). Default: the repository `xn` was built from. |
| `XN_UV` | uv executable to use. Default: `uv` on `PATH`. |
| `XN_TORCH` | PyTorch build: `auto` (default; `cuda` if `nvidia-smi` works, else `cpu`), `cpu` or `cuda` (CUDA 12.6). |

## Development

```sh
uv run pytest              # engine tests
uv run ruff check          # engine lint
cd cli && cargo clippy     # CLI lint
uv run python -m xn train  # run the engine directly, without the Rust CLI
```

After changing the defaults in `src/xn/config.py`, regenerate the template that `xn init` embeds, then rebuild `xn`:

```sh
uv run python -c "from xn.config import Config; Config().save('templates/xn.json')"
cargo install --path cli
```
