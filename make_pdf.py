import markdown
import pathlib
import subprocess
import sys

base = pathlib.Path("C:/Users/redlam/Downloads/AI challenge/mvtec-poc")
md_text = (base / "rapport.md").read_text(encoding="utf-8")

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc"],
)

css = """
@page { size: A4; margin: 2cm 2.2cm 2cm 2.2cm; }
body {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #222;
    max-width: 100%;
}
h1 {
    font-size: 17pt;
    margin-top: 0;
    border-bottom: 2px solid #2c3e50;
    padding-bottom: 6px;
    color: #2c3e50;
}
h2 {
    font-size: 13pt;
    color: #2c3e50;
    border-bottom: 1px solid #ccc;
    padding-bottom: 3px;
    margin-top: 22px;
}
h3 { font-size: 11pt; color: #34495e; margin-top: 14px; }
p  { margin: 6px 0 8px 0; }
code, pre {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 8.5pt;
    background: #f4f4f4;
    border-radius: 3px;
}
pre {
    padding: 8px 10px;
    border-left: 3px solid #bbb;
    white-space: pre-wrap;
    word-break: break-all;
}
code { padding: 1px 4px; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 9pt;
}
th {
    background: #2c3e50;
    color: white;
    padding: 6px 8px;
    text-align: left;
}
td { padding: 5px 8px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) td { background: #f9f9f9; }
blockquote {
    border-left: 3px solid #aaa;
    margin: 8px 0;
    padding: 4px 12px;
    color: #555;
    font-style: italic;
}
img {
    max-width: 48%;
    height: auto;
    display: inline-block;
    margin: 4px 1%;
    border: 1px solid #ddd;
    border-radius: 3px;
    vertical-align: top;
}
table img {
    max-width: 100%;
    display: block;
    margin: 0 auto;
}
ul, ol { margin: 4px 0 8px 0; padding-left: 20px; }
li { margin-bottom: 3px; }
hr { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
"""

html_full = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Rapport MVTec — Corel &amp; Reda</title>
  <style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>
"""

html_path = base / "rapport_temp.html"
html_path.write_text(html_full, encoding="utf-8")
print(f"HTML intermédiaire : {html_path}")

out_pdf = base / "rapport.pdf"
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

cmd = [
    chrome,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={out_pdf}",
    "--no-pdf-header-footer",
    str(html_path.as_uri()),
]

print("Lancement Chrome headless...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
if result.returncode == 0:
    print(f"PDF généré : {out_pdf}")
else:
    print("STDERR:", result.stderr[:500])
    sys.exit(1)
