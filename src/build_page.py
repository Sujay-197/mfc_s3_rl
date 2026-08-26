"""Build the standalone results page from artifacts/page_template.html.

Inlines the demo video and training curve as base64 data URIs so the page is a
single self-contained file that opens anywhere, with no server and no relative
asset paths.

    python src/build_page.py
"""

import base64
import pathlib

ART = pathlib.Path("artifacts")


def main():
    template = ART / "page_template.html"
    video = ART / "demo.mp4"
    curve = ART / "training_curve.png"
    for path in (template, video, curve):
        if not path.exists():
            raise SystemExit(f"missing {path} -- run src/report.py first")

    html = template.read_text(encoding="utf-8")
    html = html.replace("__VID__", base64.b64encode(video.read_bytes()).decode())
    html = html.replace("__PNG__", base64.b64encode(curve.read_bytes()).decode())
    if "__VID__" in html or "__PNG__" in html:
        raise SystemExit("placeholder substitution failed")

    out = ART / "page.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({round(len(html) / 1024)} KB)")


if __name__ == "__main__":
    main()
