import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "vercel_build.mjs"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def build(tmp_path, api_url=None):
    (tmp_path / "frontend" / "js").mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ["PATH"], "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    if api_url is not None:
        env["SKINSENSE_API_URL"] = api_url
    proc = subprocess.run(["node", str(SCRIPT)], cwd=tmp_path, env=env, capture_output=True, text=True)
    config = tmp_path / "frontend" / "js" / "config.js"
    return proc, (config.read_text() if config.exists() else None)


def test_vercel_config_points_at_the_static_front_end():
    cfg = json.loads((ROOT / "vercel.json").read_text())
    assert cfg["outputDirectory"] == "frontend"
    assert cfg["buildCommand"] == "node scripts/vercel_build.mjs" and SCRIPT.exists()
    keys = {h["key"] for rule in cfg["headers"] for h in rule["headers"]}
    assert {"X-Content-Type-Options", "X-Frame-Options", "Referrer-Policy"} <= keys


def test_committed_default_config_means_same_origin():
    text = (ROOT / "frontend" / "js" / "config.js").read_text()
    assert 'window.SKINSENSE_API_URL = "";' in text


def test_the_page_loads_the_config_before_the_app():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert html.index("js/config.js") < html.index("js/app.js")


@needs_node
def test_build_writes_the_api_url_and_trims_trailing_slashes(tmp_path):
    proc, text = build(tmp_path, "https://api.example.com///")
    assert proc.returncode == 0, proc.stderr
    assert text.strip() == 'window.SKINSENSE_API_URL = "https://api.example.com";'


@needs_node
def test_build_without_a_url_leaves_it_empty(tmp_path):
    proc, text = build(tmp_path)
    assert proc.returncode == 0 and text.strip() == 'window.SKINSENSE_API_URL = "";'


@needs_node
@pytest.mark.parametrize("bad", ["javascript:alert(1)", "api.example.com", 'https://x"; alert(1); "', "https://a b.com"])
def test_build_rejects_anything_that_is_not_a_plain_http_url(tmp_path, bad):
    proc, text = build(tmp_path, bad)
    assert proc.returncode == 1 and text is None, "an unsafe value must never reach the generated script"
    assert "must be a plain http(s) URL" in proc.stderr, "must fail for the right reason, not crash"


@needs_node
def test_plain_http_to_a_public_host_triggers_a_mixed_content_warning(tmp_path):
    proc, _ = build(tmp_path, "http://api.example.com")
    assert proc.returncode == 0 and "mixed content" in (proc.stdout + proc.stderr).lower()
