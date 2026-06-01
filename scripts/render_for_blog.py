"""Render the 4 notebooks to HTML for the blog tutorials section.

Output goes to ~/Documents/crw-blog/tutorials/numerical-linalg-for-ml/.
Each output gets a human-readable <title> for browser tabs.
"""
from pathlib import Path
import base64
import hashlib
import re
import subprocess
import urllib.request

# Add Subresource Integrity to the CDN scripts nbconvert injects (require.js,
# MathJax). Without this, a compromised CDN could serve malware to readers --
# the polyfill.io class of attack. Self-maintaining: hashes are computed from
# the live bytes, so this keeps working if nbconvert bumps script versions.
SCRIPT_SRC_RE = re.compile(r'<script\b[^>]*\bsrc=(["\'])(?P<url>https?://[^"\']+)\1[^>]*>')


def _sri_hash(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        digest = hashlib.sha384(resp.read()).digest()
    return "sha384-" + base64.b64encode(digest).decode("ascii")


def harden_external_scripts(html: str) -> str:
    """Inject integrity + crossorigin into external <script> tags lacking it."""
    seen: dict[str, str] = {}
    for m in SCRIPT_SRC_RE.finditer(html):
        tag, url = m.group(0), m.group("url")
        if "integrity=" in tag:
            continue
        if url not in seen:
            try:
                seen[url] = _sri_hash(url)
            except Exception as exc:  # noqa: BLE001 -- fail soft, never block a render
                print(f"  WARNING: could not hash {url} ({exc}); left without SRI")
                seen[url] = ""
        sri = seen[url]
        if not sri:
            continue
        new_tag = tag[:-1] + f' integrity="{sri}" crossorigin="anonymous">'
        html = html.replace(tag, new_tag)
        print(f"  + SRI {url}")
    return html


NOTEBOOKS = {
    "01_krylov.ipynb": "NLA for ML 1: Krylov methods for the Hessian",
    "02_randomized.ipynb": "NLA for ML 2: Randomized methods and the empirical NTK",
    "03_estimation.ipynb": "NLA for ML 3: Trace, density of states, perturbation",
    "04_capstone.ipynb": "NLA for ML 4: Spectroscopy of a network across training",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "notebooks"
OUT_DIR = Path.home() / "Documents" / "crw-blog" / "tutorials" / "numerical-linalg-for-ml"


def render_one(nb_name: str, title: str) -> None:
    src = NB_DIR / nb_name
    subprocess.run(
        ["uv", "run", "jupyter", "nbconvert", "--to", "html",
         "--output-dir", str(OUT_DIR), str(src)],
        check=True,
        cwd=REPO_ROOT,
    )
    out = OUT_DIR / nb_name.replace(".ipynb", ".html")
    html = out.read_text(encoding="utf-8")
    old_title = f"<title>{nb_name.replace('.ipynb', '')}</title>"
    new_title = f"<title>{title}</title>"
    if old_title not in html:
        raise RuntimeError(f"Expected title tag {old_title!r} not found in {out}")
    html = html.replace(old_title, new_title)
    html = harden_external_scripts(html)
    out.write_text(html, encoding="utf-8")
    print(f"  -> {out.name}: {title}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for nb, title in NOTEBOOKS.items():
        render_one(nb, title)
    print(f"\nrendered {len(NOTEBOOKS)} notebooks to {OUT_DIR}")


if __name__ == "__main__":
    main()
