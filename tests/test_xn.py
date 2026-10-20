import pytest
import torch

from xn import GPT, BPETokenizer, ByteTokenizer, Config
from xn.train import load_checkpoint, train


def test_tokenizer_roundtrip():
    tok = ByteTokenizer()
    s = "Hello, 世界!"
    assert tok.decode(tok.encode(s)) == s


def test_bpe_train_roundtrip():
    text = "吾輩は猫である。名前はまだ無い。hello hello world " * 20
    tok = BPETokenizer.train(text, 300)
    assert 256 < tok.vocab_size <= 300
    assert len(tok.encode(text)) < len(text.encode("utf-8"))
    for s in [text, "未知の文字列 xyz!"]:
        assert tok.decode(tok.encode(s)) == s


def test_content_loader(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.jsonl").write_text('{"text": "foo"}\n{"text": "bar"}\n')
    (tmp_path / "c.csv").write_text("body,x\nbaz,1\n")
    cfg = Config(content_loader={"separator": "|", "sources": [
        {"path": str(tmp_path / "*.txt")},
        {"path": str(tmp_path / "b.jsonl"), "format": "jsonl"},
        {"path": str(tmp_path / "c.csv"), "format": "csv", "field": "body"},
    ]})
    assert cfg.load_text() == "hello|foo|bar|baz"


def test_config_rejects_removed_data(tmp_path):
    p = tmp_path / "xn.json"
    p.write_text('{"data": "a.txt"}')
    with pytest.raises(SystemExit, match="content_loader"):
        Config.load(p)


def test_model_forward_and_generate():
    cfg = Config(vocab_size=256, n_layer=1, n_head=2, n_embd=32, block_size=16)
    model = GPT(cfg)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss.item() > 0
    assert model.generate(x[:, :4], 20).shape == (2, 24)


def test_train_saves_checkpoint(tmp_path):
    data = tmp_path / "d.txt"
    data.write_text("abcdefg " * 200)
    cfg = Config(n_layer=1, n_head=2, n_embd=32, block_size=16, batch_size=4,
                 max_steps=5, eval_interval=5, eval_steps=2,
                 content_loader={"sources": [{"path": str(data)}]}, out_dir=str(tmp_path / "run"))
    train(cfg)
    model, tok = load_checkpoint(tmp_path / "run" / "ckpt.pt")
    assert model.cfg.n_layer == 1
    assert isinstance(tok, BPETokenizer)
    assert model.cfg.vocab_size == tok.vocab_size


def test_doctor(tmp_path, monkeypatch, capsys):
    from xn.doctor import doctor

    monkeypatch.chdir(tmp_path)
    (tmp_path / "xn.json").write_text('{"data": "x.txt"}')
    assert doctor() == 1
    assert "✗ project" in capsys.readouterr().out


def test_template_matches_config_defaults():
    # templates/xn.json is embedded in the native CLI for `xn init`
    import json
    from dataclasses import asdict
    from pathlib import Path

    tpl = Path(__file__).parent.parent / "templates" / "xn.json"
    assert json.loads(tpl.read_text()) == asdict(Config())


def test_split_docs_holds_out_whole_documents():
    from xn.train import split_docs

    cfg = Config(val_fraction=0.25, content_loader={"separator": "|", "sources": []})
    assert split_docs(["a", "b", "c", "d"], cfg) == ("a|b|c", "d")
    assert split_docs(["only"], cfg) == ("only", None)


def test_loader_dedupes_documents(tmp_path):
    (tmp_path / "a.txt").write_text("same")
    (tmp_path / "b.txt").write_text("same")
    cfg = Config(content_loader={"sources": [{"path": str(tmp_path / "*.txt")}]})
    assert cfg.load_docs() == ["same"]


def test_lr_schedule():
    from xn.train import lr_at

    cfg = Config(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=110)
    assert lr_at(5, cfg) == 5e-4
    assert lr_at(10, cfg) == 1e-3
    assert abs(lr_at(110, cfg) - 1e-4) < 1e-12


def test_clean_aozora():
    from xn.data import clean_aozora

    raw = "題\n著者\n" + "-" * 30 + "\n凡例\n" + "-" * 30 + "\n吾輩《わがはい》は｜猫［＃傍点］である。\n底本：某\n"
    assert clean_aozora(raw).strip() == "吾輩は猫である。"


def test_html_to_text():
    from xn.data import html_to_text

    text = html_to_text("<html><script>x()</script><p>Hello</p><p>World</p></html>")
    assert "x()" not in text
    assert text.split() == ["Hello", "World"]


def test_repetition_penalty_discourages_seen_tokens():
    class Fixed(GPT):
        def forward(self, idx, targets=None):  # token 7 always the clear favourite
            logits = torch.zeros(idx.size(0), idx.size(1), self.cfg.vocab_size)
            logits[..., 7] = 5.0
            logits[..., 8] = 4.0
            return logits, None

    model = Fixed(Config(vocab_size=256, n_layer=1, n_head=2, n_embd=32, block_size=16))
    prompt = torch.tensor([[7] * 8])
    assert (model.generate(prompt, 10, top_k=1)[0, 8:] == 7).all()
    # 7 (5.0 -> 2.5) drops below 8 (4.0); once 8 is also seen, 7 (2.5) beats 8 (2.0) again
    out = model.generate(prompt, 2, top_k=1, repetition_penalty=2.0)[0, 8:].tolist()
    assert out == [8, 7]
