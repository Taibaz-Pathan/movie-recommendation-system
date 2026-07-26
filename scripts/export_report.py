import markdown
import pathlib

md = pathlib.Path('reports/progress_week1_4.md').read_text()
body = markdown.markdown(md, extensions=['tables'])

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Progress Report — Weeks 1 to 4</title>
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Roboto', sans-serif;
    font-size: 1rem;
    line-height: 1.75;
    color: #1a1a1a;
    background: #ffffff;
    max-width: 820px;
    margin: 60px auto;
    padding: 0 32px 80px;
  }

  h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: #111;
  }

  h2 {
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 48px;
    margin-bottom: 14px;
    color: #111;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 6px;
  }

  p {
    margin-bottom: 14px;
  }

  strong {
    font-weight: 700;
    color: #111;
  }

  ul, ol {
    margin: 10px 0 16px 24px;
  }

  li {
    margin-bottom: 6px;
  }

  code {
    font-family: 'Courier New', monospace;
    font-size: 0.88rem;
    background: #f4f4f4;
    padding: 1px 5px;
    border-radius: 3px;
    color: #333;
  }

  pre {
    background: #f4f4f4;
    padding: 16px 20px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 16px 0;
    font-size: 0.88rem;
    line-height: 1.6;
    border-left: 4px solid #ccc;
  }

  pre code {
    background: none;
    padding: 0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 0.95rem;
  }

  th {
    background: #f0f0f0;
    text-align: left;
    padding: 10px 14px;
    font-weight: 700;
    border: 1px solid #ddd;
  }

  td {
    padding: 9px 14px;
    border: 1px solid #ddd;
    vertical-align: top;
  }

  tr:nth-child(even) td {
    background: #fafafa;
  }

  hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 36px 0;
  }

  em {
    color: #555;
    font-style: italic;
  }

  a { color: #1a73e8; }
</style>
</head>
<body>
""" + body + """
</body>
</html>"""

pathlib.Path('reports/progress_week1_4.html').write_text(html)
print('Done — reports/progress_week1_4.html')
