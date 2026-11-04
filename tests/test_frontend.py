"""Static checks that keep index.html and app.js in sync (no browser needed)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
HTML = (FRONTEND / "index.html").read_text(encoding="utf-8")
JS = (FRONTEND / "js" / "app.js").read_text(encoding="utf-8")


def test_every_element_id_used_by_the_script_exists_in_the_page():
    used = set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", JS))
    have = set(re.findall(r'id="([^"]+)"', HTML))
    assert not sorted(used - have), f"script looks up ids missing from index.html: {sorted(used - have)}"
    for step in ("stepUpload", "stepAnalyze", "stepReview"):   # built dynamically in setStep()
        assert step in have


def test_ids_are_unique():
    ids = re.findall(r'id="([^"]+)"', HTML)
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, dupes


def test_page_references_files_that_exist():
    for ref in re.findall(r'(?:href|src)="((?:css|js|img)/[^"?]+)', HTML):
        assert (FRONTEND / ref).is_file(), ref


def test_icons_used_in_markup_and_script_are_defined_in_the_sprite():
    defined = set(re.findall(r'<symbol id="i-([a-z-]+)"', HTML))
    used = set(re.findall(r'href="#i-([a-z-]+)"', HTML)) | set(re.findall(r"icon\('([a-z-]+)'", JS))
    used |= set(re.findall(r"\{ success: 'check', error: 'x-circle', warn: 'alert', info: 'info' \}", JS) and
                ["check", "x-circle", "alert", "info"])
    assert not sorted(used - defined), sorted(used - defined)


def test_all_innerhtml_values_are_escaped_or_static():
    """Every ${...} interpolated into an innerHTML template must be esc()-wrapped, an icon/constant, or numeric-safe."""
    safe = ("esc(", "icon(", "file", "directionsUrl(", "r.tone", "ic)", "tone", "(item.image_url ?", "item.image_url ?")
    offenders = []
    for m in re.finditer(r"innerHTML\s*=\s*(`[^`]*`)", JS, re.S):
        for expr in re.findall(r"\$\{([^}]*)\}", m.group(1)):
            e = expr.strip()
            if not e.startswith(safe) and "esc(" not in e and not re.fullmatch(r"[a-zA-Z_.]*(tone)", e):
                offenders.append(e)
    assert not offenders, offenders


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_script_has_valid_syntax():
    result = subprocess.run(["node", "--check", str(FRONTEND / "js" / "app.js")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_served_by_the_backend(client):
    page = client.get("/")
    assert page.status_code == 200 and "SkinSense AI" in page.text
    assert client.get("/css/app.css").status_code == 200
    assert client.get("/js/app.js").status_code == 200
    assert client.get("/js/app_v2.js").status_code == 404      # old files are gone
