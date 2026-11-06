import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_space  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_space.build(tmp_path_factory.mktemp("space") / "space")


def test_space_metadata_matches_the_container(bundle):
    readme = (bundle / "README.md").read_text(encoding="utf-8")
    front = readme.split("---")[1]
    assert "sdk: docker" in front and "app_port: 8000" in front
    docker = (bundle / "Dockerfile").read_text(encoding="utf-8")
    assert "EXPOSE 8000" in docker and '"--port", "8000"' in docker, "app_port must be the port uvicorn listens on"
    assert "SKINSENSE_FRAME_ANCESTORS=https://huggingface.co" in docker


def test_every_dockerfile_copy_source_exists_in_the_bundle(bundle):
    docker = (bundle / "Dockerfile").read_text(encoding="utf-8")
    sources = []
    for line in docker.splitlines():
        if line.startswith("COPY "):
            sources += [p for p in line.split()[1:-1] if not p.startswith("--")]
    assert sources, "expected COPY lines"
    missing = [s for s in sources if not (bundle / s).exists()]
    assert not missing, missing


def test_the_validated_calibration_ships_with_the_app(bundle):
    assert (bundle / "backend" / "calibration.json").exists()
    assert (bundle / "backend" / "fusion.json").exists()
    assert (bundle / "data" / "knowledge_base.json").exists()
    assert (bundle / "frontend" / "index.html").exists() and (bundle / "frontend" / "js" / "config.js").exists()


def test_the_bundle_never_contains_weights_secrets_data_or_tests(bundle):
    bad = [p.name for p in bundle.rglob("*") if p.is_file() and (
        p.suffix in {".pth", ".db", ".npz", ".pid", ".log"} or p.name in {".secret_key", ".image_key", ".env", "splits.json"}
        or p.name.startswith("test_") or "__pycache__" in p.parts)]
    assert not bad, bad
    assert not (bundle / "dataset").exists() and not (bundle / "runs").exists() and not (bundle / "tests").exists()


def test_every_local_module_the_backend_imports_is_in_the_bundle(bundle):
    repo_modules = {p.stem for p in (ROOT / "backend").glob("*.py")}
    missing = set()
    for py in (bundle / "backend").glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            names = [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module else \
                    [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else []
            missing |= {n for n in names if n in repo_modules and not (bundle / "backend" / f"{n}.py").exists()}
    assert not missing, missing


def test_the_bundle_is_small_enough_for_a_plain_git_push(bundle):
    biggest = max(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    assert biggest < 10 * 1024 * 1024, "files over 10 MB need Git LFS on Hugging Face"
