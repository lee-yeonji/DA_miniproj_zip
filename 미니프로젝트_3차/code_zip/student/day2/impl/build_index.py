# -*- coding: utf-8 -*-
"""
Day2 인덱싱 엔트리포인트
- 목표: 코퍼스 생성 → 임베딩 → FAISS 저장 + docs.jsonl 저장
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except Exception:
    pass


import argparse, numpy as np
from typing import List

from student.day2.impl.ingest import build_corpus, save_docs_jsonl
from student.day2.impl.embeddings import Embeddings
from student.day2.impl.store import FaissStore  # 제공됨


def build_index(paths: List[str], index_dir: str, model: str | None = None, batch_size: int = 128):
    """
    절차:
      1) corpus = build_corpus(paths)
         - [{"id":..., "text":..., "meta":{...}}, ...]
      2) texts = [item["text"] for item in corpus]
      3) emb = Embeddings(model=model, batch_size=batch_size)
         vecs = emb.encode(texts)  # (N, D) L2 정규화된 np.ndarray
      4) index_path = os.path.join(index_dir, "faiss.index")
         docs_path  = os.path.join(index_dir, "docs.jsonl")
      5) store = FaissStore(dim=vecs.shape[1], index_path=index_path, docs_path=docs_path)
         store.add(vecs, corpus); store.save()
      6) save_docs_jsonl(corpus, docs_path)
    """
   
     # # 1) 코퍼스 생성
    corpus = build_corpus(paths)  # list[dict{id, text, meta}]
    
    if not corpus:
        # 빈 코퍼스일 땐 인덱스 파일만 비워두고 종료
        os.makedirs(index_dir, exist_ok=True)
        docs_path = os.path.join(index_dir, "docs.jsonl")
        save_docs_jsonl([], docs_path)
        return

    # 2) 텍스트만 추출
    texts = [item.get("text", "") for item in corpus]

    # 3) 임베딩 생성
    emb = Embeddings(model=model, batch_size=batch_size)
    vecs = emb.encode(texts)  # np.ndarray [N, D]
    if not isinstance(vecs, np.ndarray) or vecs.ndim != 2:
        raise ValueError("Embeddings.encode() must return a 2D numpy array of shape (N, D).")

    # 4) 경로 준비
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss.index")
    docs_path  = os.path.join(index_dir, "docs.jsonl")

    # 5) FAISS 저장
    store = FaissStore(dim=vecs.shape[1], index_path=index_path, docs_path=docs_path)
    store.add(vecs, corpus)
    store.save()

    # 6) 원본 문서 메타 저장
    save_docs_jsonl(corpus, docs_path)

"""
실행 방법! 꼭 터미널에 아래 코드를 복사해서 붙여넣고 실행 먼저!

python -m student.day2.impl.build_index `
  --paths data/raw `
  --index_dir indices/day2 `
  --model text-embedding-3-small `
  --batch_size 128

"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--index_dir", default="indices/day2")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    build_index(args.paths, args.index_dir, args.model, args.batch_size)
