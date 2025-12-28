# -*- coding: utf-8 -*-
"""
간단 웹 수집기: URL → 본문 추출 → .md 저장
- trafilatura가 있으면 그걸로 본문 추출, 없으면 BeautifulSoup fallback
- 저장 위치: data/raw/web/{slug}.md
"""
from __future__ import annotations
import os, re, argparse, pathlib, datetime, requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup  # pip install beautifulsoup4

try:
    import trafilatura  # pip install trafilatura
except Exception:
    trafilatura = None

ROOT = pathlib.Path(__file__).resolve().parents[3]   # 프로젝트 루트(FINAL)
OUT_DIR = ROOT / "data" / "raw" / "web"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def slugify(url: str) -> str:
    u = urlparse(url)
    s = (u.netloc + u.path).strip("/").replace("/", "_")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s[:180] or "index"

def fetch_html(url: str, timeout: int = 20) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; edu-rag/1.0)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r.text or ""

def extract_text(html: str, url: str) -> tuple[str, str]:
    """
    return (title, clean_text)  — 가능한 한 본문만
    """
    if trafilatura is not None:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=True, url=url)
        if extracted:
            # trafilatura는 제목을 함께내기도 하지만 안전하게 soup로 타이틀 보완
            soup = BeautifulSoup(html, "html.parser")
            title = (soup.title.string.strip() if soup.title and soup.title.string else "")
            return (title, extracted)

    # fallback: 매우 단순한 본문 추출
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    for tag in soup(["script", "style", "noscript", "header", "footer", "aside"]):
        tag.decompose()
    text = "\n".join(x.strip() for x in soup.get_text("\n").splitlines() if x.strip())
    return (title, text)

def save_markdown(url: str, title: str, text: str) -> pathlib.Path:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    name = f"{slugify(url)}.md"
    path = OUT_DIR / name
    md = f"""---
source_url: {url}
title: {title}
fetched_at: {ts}
---

# {title or '제목 없음'}

{text}
"""
    path.write_text(md, encoding="utf-8")
    return path

def fetch_and_save(urls: list[str]) -> list[pathlib.Path]:
    out = []
    for u in urls:
        try:
            html = fetch_html(u)
            title, body = extract_text(html, u)
            out.append(save_markdown(u, title, body))
            print(f"[OK] {u}")
        except Exception as e:
            print(f"[ERR] {u}: {e}")
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--urls", nargs="+", required=True, help="하나 이상 URL")
    args = p.parse_args()
    fetch_and_save(args.urls)
