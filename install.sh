#!/bin/sh
# Install the xn binary and set up its Python engine. See usage() below.
set -eu

XN_REPO=${XN_REPO:-https://github.com/nk4dev/xn.git}
INSTALL_DEPS=0
BIN_DIR=""

usage() {
  cat <<'USAGE'
Install the xn binary and set up its Python engine.

  curl -fsSL https://apps.nknighta.me/xn/install.sh | sh
  curl -fsSL https://apps.nknighta.me/xn/install.sh | sh -s -- --install-deps
  ./install.sh                     # from a checkout of this repository

Options:
  --install-deps   install missing uv / Rust with their official installers
  --bin-dir DIR    where to put xn (default: ~/.cargo/bin; must end in /bin)
  -h, --help       show this help

Environment:
  XN_REPO   git URL to clone when not run from a checkout
            (default: https://github.com/nk4dev/xn.git)
  XN_HOME   where to clone (default: ~/.local/share/xn)
USAGE
}
if [ -t 2 ]; then B='\033[1;34m' Y='\033[1;33m' R='\033[1;31m' N='\033[0m'; else B='' Y='' R='' N=''; fi
info() { printf "${B}==>${N} %s\n" "$*"; }
warn() { printf "${Y}warning:${N} %s\n" "$*" >&2; }
die() { printf "${R}error:${N} %s\n" "$*" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --install-deps) INSTALL_DEPS=1 ;;
    --bin-dir) [ $# -ge 2 ] || die "--bin-dir needs a directory"; BIN_DIR=$2; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done

if [ -n "$BIN_DIR" ]; then
  case "$BIN_DIR" in */bin) ;; *) die "--bin-dir must end in /bin (e.g. ~/.local/bin)" ;; esac
fi

# --- locate the source -----------------------------------------------------
# when piped (curl | sh) $0 is the shell, not a file: never treat cwd as the checkout then
SCRIPT_DIR=""
if [ -f "$0" ]; then SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd); fi
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/cli/Cargo.toml" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
  SRC=$SCRIPT_DIR
else
  has git || die "git is required to clone $XN_REPO"
  SRC=${XN_HOME:-$HOME/.local/share/xn}
  if [ -d "$SRC/.git" ]; then
    info "updating $SRC"
    git -C "$SRC" pull --ff-only
  else
    info "cloning $XN_REPO into $SRC"
    git clone --depth 1 "$XN_REPO" "$SRC"
  fi
fi
info "source: $SRC"

# --- dependencies ----------------------------------------------------------
ensure_uv() {
  has uv && return
  [ "$INSTALL_DEPS" = 1 ] || die "uv not found. Install it (https://docs.astral.sh/uv/) or rerun with --install-deps"
  info "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
  has uv || die "uv was installed but is not on PATH; open a new shell and rerun"
}

ensure_cargo() {
  has cargo && return
  [ -x "$HOME/.cargo/bin/cargo" ] && { PATH="$HOME/.cargo/bin:$PATH"; return; }
  [ "$INSTALL_DEPS" = 1 ] || die "cargo not found. Install Rust (https://rustup.rs/) or rerun with --install-deps"
  info "installing Rust"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  PATH="$HOME/.cargo/bin:$PATH"
}

has curl || [ "$INSTALL_DEPS" = 0 ] || die "curl is required for --install-deps"
ensure_uv
ensure_cargo
info "uv: $(uv --version)"
info "cargo: $(cargo --version)"

# --- engine ----------------------------------------------------------------
info "installing the Python engine (uv sync)"
uv sync --project "$SRC"

# --- binary ----------------------------------------------------------------
if [ -n "$BIN_DIR" ]; then
  # cargo install --root R puts the binary in R/bin
  ROOT=${BIN_DIR%/bin}
  info "building xn into $BIN_DIR"
  cargo install --path "$SRC/cli" --locked --force --root "$ROOT"
else
  BIN_DIR="${CARGO_HOME:-$HOME/.cargo}/bin"
  info "building xn into $BIN_DIR"
  cargo install --path "$SRC/cli" --locked --force
fi

# --- check -----------------------------------------------------------------
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "$BIN_DIR is not on your PATH. Add this to your shell profile:"
     printf '    export PATH="%s:$PATH"\n' "$BIN_DIR" >&2 ;;
esac

info "running xn doctor"
"$BIN_DIR/xn" doctor || warn "xn doctor reported problems (see above)"
info "done. Try: xn init myllm && cd myllm && xn train"
