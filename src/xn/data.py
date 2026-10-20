"""`xn data`: fetch training text into ./data and inspect what is there."""

import io
import re
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path("data")
USER_AGENT = "xn-data/0.1 (+https://github.com/)"
TEXT_SUFFIXES = (".txt", ".md")


class _HTMLText(HTMLParser):
    """Collect visible text, dropping script/style and keeping block structure."""

    BLOCK = frozenset({"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"})
    SKIP = frozenset({"script", "style", "noscript", "head", "rp", "rt"})

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _HTMLText()
    p.feed(html)
    return "".join(p.parts)


def decode(raw: bytes, charset: str | None = None) -> str:
    for enc in filter(None, [charset, "utf-8", "cp932", "euc-jp"]):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def clean_aozora(text: str) -> str:
    """Strip Aozora Bunko header, footer, ruby and editor notes."""
    lines = text.splitlines()
    # header ends at the second line of dashes (legend of ruby notation)
    dashes = [i for i, line in enumerate(lines[:80]) if re.fullmatch(r"-{20,}", line.strip())]
    if len(dashes) >= 2:
        lines = lines[dashes[1] + 1 :]
    elif lines:
        lines = lines[2:]  # title + author
    for i, line in enumerate(lines):
        if line.startswith("底本："):
            lines = lines[:i]
            break
    text = "\n".join(lines)
    text = re.sub(r"《[^》]*》", "", text)  # ruby readings
    text = re.sub(r"［＃[^］]*］", "", text)  # editor notes
    text = text.replace("｜", "")  # ruby start marker
    return text


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def fetch(url: str) -> tuple[bytes, str | None, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as res:
        return res.read(), res.headers.get_content_charset(), res.headers.get_content_type()


def to_text(raw: bytes, charset: str | None, ctype: str, url: str) -> str:
    if raw[:2] == b"PK":  # zip (Aozora Bunko distributes text as zip)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not names:
                raise SystemExit(f"{url}: zip contains no .txt file")
            raw, charset, ctype = z.read(names[0]), None, "text/plain"
    text = decode(raw, charset)
    if ctype == "text/html" or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        text = html_to_text(text)
    if "aozora.gr.jp" in url or "底本：" in text:
        text = clean_aozora(text)
    return normalize(text)


def default_name(url: str) -> str:
    stem = Path(urllib.parse.urlparse(url).path).stem or "download"
    return re.sub(r"[^\w.-]+", "_", stem) + ".txt"


def add(urls: list[str], name: str | None = None, force: bool = False) -> None:
    if name and len(urls) > 1:
        raise SystemExit("--name can only be used with a single URL")
    DATA_DIR.mkdir(exist_ok=True)
    for url in urls:
        dest = DATA_DIR / (name or default_name(url))
        if dest.suffix not in TEXT_SUFFIXES:
            dest = dest.with_suffix(".txt")
        if dest.exists() and not force:
            print(f"skip {dest} (exists; use --force to overwrite)")
            continue
        if Path(url).is_file():
            raw, charset, ctype = Path(url).read_bytes(), None, "text/plain"
        else:
            print(f"fetching {url}")
            raw, charset, ctype = fetch(url)
        text = to_text(raw, charset, ctype, url)
        dest.write_text(text, encoding="utf-8")
        print(f"saved {dest} ({len(text):,} chars)")


def list_data() -> None:
    from xn.config import Config

    files = sorted(f for f in DATA_DIR.rglob("*") if f.is_file()) if DATA_DIR.is_dir() else []
    total = 0
    for f in files:
        n = len(f.read_text(encoding="utf-8", errors="replace"))
        total += n
        print(f"{n:>12,}  {f}")
    print(f"{total:>12,}  chars in {len(files)} file(s) under {DATA_DIR}/")
    if Path("xn.json").exists():
        docs = Config.load("xn.json").load_docs()
        chars = sum(map(len, docs))
        print(f"content_loader: {len(docs)} unique document(s), {chars:,} chars")


# permissive licenses whose code may be used for training and the resulting model shared
# (keep the attribution in data/SOURCES.json); copyleft / unlicensed repos are refused
PERMISSIVE = frozenset(
    {"MIT", "MIT-0", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD", "Unlicense",
     "CC0-1.0", "Python-2.0", "PSF-2.0", "Zlib", "BlueOak-1.0.0"}
)
CODE_SKIP = re.compile(r"(^|/)(node_modules|dist|build|vendor|coverage|\.git)/|\.min\.\w+$")
SOURCES = DATA_DIR / "SOURCES.json"


def _github_json(path: str) -> dict:
    import json
    import os

    req = urllib.request.Request(f"https://api.github.com/{path}", headers={"User-Agent": USER_AGENT})
    if token := os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.load(res)


def add_repo(
    repo: str, include: list[str], ref: str | None = None, max_kb: int = 200, force: bool = False
) -> None:
    """Download a GitHub repo's source files into data/<owner>_<name>/ if its license is permissive."""
    import fnmatch
    import json

    repo = repo.removeprefix("https://github.com/").strip("/")
    info = _github_json(f"repos/{repo}")
    spdx = (info.get("license") or {}).get("spdx_id") or "NONE"
    if spdx not in PERMISSIVE:
        raise SystemExit(
            f"{repo}: license {spdx} is not in the permissive allow-list "
            f"({', '.join(sorted(PERMISSIVE))}); not downloading"
        )
    ref = ref or info["default_branch"]
    sha = _github_json(f"repos/{repo}/commits/{ref}")["sha"]
    dest = DATA_DIR / repo.replace("/", "_")
    if dest.exists() and not force:
        print(f"skip {dest} (exists; use --force to overwrite)")
        return
    print(f"fetching {repo}@{sha[:10]} ({spdx})")
    raw, _, _ = fetch(f"https://codeload.github.com/{repo}/zip/{sha}")
    n = chars = 0
    license_text = None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for member in z.infolist():
            path = member.filename.split("/", 1)[-1]  # drop the "<repo>-<sha>/" root
            if member.is_dir() or not path:
                continue
            if "/" not in path and re.match(r"(?i)licen[cs]e|copying", path):
                license_text = z.read(member).decode("utf-8", "replace")
            name = path.rsplit("/", 1)[-1]
            if CODE_SKIP.search(path) or member.file_size > max_kb * 1024:
                continue
            if not any(fnmatch.fnmatch(name, pat) for pat in include):
                continue
            try:
                text = z.read(member).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out = dest / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            n += 1
            chars += len(text)
    if not n:
        raise SystemExit(f"{repo}: no files match {', '.join(include)}")
    if license_text:
        (dest / "LICENSE.upstream").write_text(license_text, encoding="utf-8")
    sources = json.loads(SOURCES.read_text()) if SOURCES.exists() else []
    sources = [s for s in sources if s["repo"] != repo] + [
        {"repo": repo, "commit": sha, "license": spdx, "files": n, "chars": chars,
         "url": f"https://github.com/{repo}/tree/{sha}"}
    ]
    SOURCES.write_text(json.dumps(sources, indent=2) + "\n")
    print(f"saved {n} files ({chars:,} chars) to {dest}/ - recorded in {SOURCES}")
