#!/usr/bin/env python3
# render_report_pdf.py - export report/report.md to PDF using Chrome headless + KaTeX

import html as html_lib
import re
import subprocess
import tempfile
from pathlib import Path
from markdown_it import MarkdownIt

REPO_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = REPO_ROOT / "report" / "report.md"
PDF_PATH = REPO_ROOT / "report" / "CM3604_Coursework_Report.pdf"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]


def find_browser() -> str:
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    raise FileNotFoundError("Could not find Google Chrome or Brave Browser to export PDF.")


def render_markdown_preserving_math(md_text: str) -> str:
    math_blocks: list[tuple[bool, str]] = []

    def stash_display(m: re.Match) -> str:
        idx = len(math_blocks)
        math_blocks.append((True, m.group(1)))
        return f"@@MATHBLOCK{idx}@@"

    def stash_inline(m: re.Match) -> str:
        idx = len(math_blocks)
        math_blocks.append((False, m.group(1)))
        return f"@@MATHBLOCK{idx}@@"

    code_spans: list[str] = []

    def stash_code(m: re.Match) -> str:
        idx = len(code_spans)
        code_spans.append(m.group(0))
        return f"@@CODESPAN{idx}@@"

    text = re.sub(r"`[^`\n]+`", stash_code, md_text)
    text = re.sub(r"\$\$([\s\S]+?)\$\$", stash_display, text)
    text = re.sub(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)", stash_inline, text)

    for idx, c in enumerate(code_spans):
        text = text.replace(f"@@CODESPAN{idx}@@", c)

    md = MarkdownIt("commonmark").enable("table")
    rendered = md.render(text)

    for idx, (is_display, expr) in enumerate(math_blocks):
        escaped = html_lib.escape(expr)
        if is_display:
            replacement = f'<div class="math-display">$${escaped}$$</div>'
        else:
            replacement = f'<span class="math-inline">${escaped}$</span>'
        rendered = rendered.replace(f"@@MATHBLOCK{idx}@@", replacement)

    return rendered


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    body_html = render_markdown_preserving_math(md_text)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CM3604 Deep Learning - Group Coursework Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{
    delimiters: [
      {{left: '$$', right: '$$', display: true}},
      {{left: '$', right: '$', display: false}}
    ],
    throwOnError: false
  }});"></script>
<style>
  @page {{
    size: A4;
    margin: 16mm 16mm 16mm 16mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.48;
    color: #1a1a1a;
    max-width: 100%;
  }}
  h1 {{
    font-size: 16.5pt;
    color: #3b1360;
    border-bottom: 2px solid #3b1360;
    padding-bottom: 6px;
    margin-top: 0;
  }}
  h2 {{
    font-size: 13pt;
    color: #3b1360;
    margin-top: 18px;
    margin-bottom: 6px;
    border-bottom: 1px solid #d9d0e3;
    padding-bottom: 3px;
  }}
  h3 {{
    font-size: 11pt;
    color: #2b2b2b;
    margin-top: 13px;
    margin-bottom: 4px;
  }}
  h4 {{
    font-size: 10.5pt;
    color: #333;
    margin-top: 10px;
    margin-bottom: 3px;
  }}
  .math-display {{
    margin: 8px 0;
  }}
  h2, h3, h4 {{
    break-inside: avoid;
    page-break-inside: avoid;
    break-after: avoid;
    page-break-after: avoid;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0 14px 0;
    font-size: 8.5pt;
  }}
  tr {{
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  td:first-child code, th:first-child {{
    white-space: nowrap;
  }}
  th, td {{
    border: 1px solid #c8c8c8;
    padding: 4px 5px;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    background-color: #f3eef8;
    color: #2c0e47;
    font-weight: 600;
  }}
  tr:nth-child(even) td {{
    background-color: #fafafa;
  }}
  code {{
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 9pt;
    background: #f4f4f6;
    padding: 1px 4px;
    border-radius: 3px;
  }}
  hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 14px 0;
  }}
  a {{
    color: #3b1360;
    text-decoration: none;
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    browser = find_browser()
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        tmp.write(html)
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--virtual-time-budget=4000",
                f"--print-to-pdf={PDF_PATH}",
                f"file://{tmp_path}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"Rendered PDF report: {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
