# -*- coding: utf-8 -*-
"""
CSV/HWPX → 텍스트로 변환해서 RAG에 넣기 쉽게 만드는 스크립트
- CSV  : 인코딩 자동판별 → UTF-8 CSV와 요약 MD 동시 생성
- HWPX : XML에서 본문 추출 → .txt 생성
출력: data/processed/
"""

from __future__ import annotations
from pathlib import Path
import csv, io, zipfile, sys
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

# ---------- CSV ----------
def read_csv_robust(path: str) -> tuple[pd.DataFrame, str]:
    # 흔한 인코딩 후보들 시도
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception:
            pass
    # 마지막 시도: 바이너리 읽고 추정
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("cp949", "euc-kr", "utf-8", "utf-8-sig"):
        try:
            _ = raw.decode(enc)
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            return df, enc
        except Exception:
            continue
    raise RuntimeError(f"CSV 인코딩 판별 실패: {path}")

def save_csv_variants(p: Path):
    df, enc = read_csv_robust(str(p))
    # 1) UTF-8로 정규화
    out_csv = OUT / f"{p.stem}_utf8.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 2) 요약 .md (의존성 없이 최대한 안전하게)
    head = df.head(50)
    md_lines = [f"# CSV: {p.name}", "", f"- detected_encoding: {enc}", ""]
    md_lines.append("## 열(Columns)")
    md_lines.append(", ".join(map(str, head.columns)))
    md_lines.append("\n## 상위 50행 미니표")

    # tabulate가 없으면 to_string으로 대체
    try:
        md_lines.append(head.to_markdown(index=False))
    except Exception:
        md_lines.append("```\n" + head.to_string(index=False) + "\n```")

    (OUT / f"{p.stem}.md").write_text("\n".join(md_lines), encoding="utf-8")


# ---------- HWPX ----------
def hwpx_to_text(path: str) -> str:
    # HWPX는 zip(XML) 구조: word/section*.xml 또는 Contents/section*.xml 등
    text_parts: list[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        # 본문에 해당하는 섹션 추정
        target = [n for n in names if "section" in n.lower()] or names
        for name in target:
            try:
                data = zf.read(name).decode("utf-8", errors="ignore")
                # 매우 단순한 태그 제거(빠른용) – 필요시 정교화 가능
                data = data.replace("</w:t>", "").replace("<w:t>", "")
                data = data.replace("<hp:t>", "").replace("</hp:t>", "")
                # 태그 전체 제거(라이트)
                import re
                s = re.sub(r"<[^>]+>", "", data)
                s = re.sub(r"[ \t]+", " ", s)
                text_parts.append(s.strip())
            except Exception:
                continue
    return "\n\n".join([t for t in text_parts if t])

def save_hwpx_text(p: Path):
    txt = hwpx_to_text(str(p))
    (OUT / f"{p.stem}.txt").write_text(txt, encoding="utf-8")

# ---------- 메인 ----------
def main():
    # 1) CSV
    for p in RAW.glob("*.csv"):
        print("[CSV] ", p)
        save_csv_variants(p)
    # 2) HWPX
    for p in RAW.glob("*.hwpx"):
        print("[HWPX]", p)
        save_hwpx_text(p)
    print("[DONE] output ->", OUT)

if __name__ == "__main__":
    main()
