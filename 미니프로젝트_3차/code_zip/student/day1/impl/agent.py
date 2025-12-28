# -*- coding: utf-8 -*-
"""
Day1 본체
- 역할: 웹 검색 / 주가 / 기업개요(추출+요약)를 병렬로 수행하고 결과를 정규 스키마로 병합
"""

from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.adk.models.lite_llm import LiteLlm
from ...common.schemas import Day1Plan
from .merge import merge_day1_payload
# 외부 I/O
from .tavily_client import search_tavily, extract_url 
from .finance_client import get_quotes
from .web_search import (
    looks_like_ticker,
    search_company_profile,
    extract_and_summarize_profile,
)

DEFAULT_WEB_TOPK = 6
MAX_WORKERS = 4
DEFAULT_TIMEOUT = 20

import os
try:
    # .env 자동 로드(에디터에서 python.envFile을 쓰면 생략 가능)
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OPENAI_API_KEY가 설정돼 있으면 경량 모델 초기화, 실패하면 None로 두고 요약 생략
_SUM: Optional[LiteLlm]
try:
    _SUM = LiteLlm(model="openai/gpt-4o-mini")
except Exception:
    _SUM = None


def _summarize(text: str) -> str:
    """
    입력 텍스트를 LLM으로 3~5문장 수준으로 요약합니다.
    실패 시 빈 문자열("")을 반환해 상위 로직이 안전하게 진행되도록 합니다.
    """
    global _SUM

    # [1단계] _SUM이 None이면 "" 반환 (요약 생략)
    if _SUM is None:
        return ""

    # 요약 프롬프트 구성: LLM에게 요약을 지시
    prompt = f"다음 텍스트를 3~5문장으로 간결하게 요약해주세요. 원문:\n\n{text}"

    try:
        # [2단계] _SUM.invoke() 호출
        response = _SUM.invoke(prompt)

        # [3단계] 응답 객체에서 본문 텍스트 추출 및 반환
        if isinstance(response, str):
            summary = response
        elif hasattr(response, 'text') and response.text:
            summary = response.text
        else:
            summary = ""

        return summary.strip()

    except Exception:
        # [4단계] 예외 발생 시 빈 문자열 반환 (실패 시 안전하게 처리)
        return "" # <--- DAY1-I-02 구현 (raise NotImplementedError 대체)


class Day1Agent:
    def __init__(self, tavily_api_key: Optional[str], web_topk: int = DEFAULT_WEB_TOPK, request_timeout: int = DEFAULT_TIMEOUT):
        """
        필드 저장만 담당합니다.
        - tavily_api_key: Tavily API 키(없으면 웹 호출 실패 가능)
        - web_topk: 기본 검색 결과 수
        - request_timeout: 각 HTTP 호출 타임아웃(초)
        """
        self.tavily_api_key = tavily_api_key
        self.web_topk = web_topk
        self.request_timeout = request_timeout

    def handle(self, query: str, plan: Day1Plan) -> Dict[str, Any]:
        """
        병렬 파이프라인:
          1) results 스켈레톤 만들기
             results = {"type":"web_results","query":query,"analysis":asdict(plan),"items":[],
                        "tickers":[], "errors":[], "company_profile":"", "profile_sources":[]}
          2) ThreadPoolExecutor(max_workers=MAX_WORKERS)에서 작업 제출:
             - plan.do_web: search_tavily(검색어, 키, top_k=self.web_topk, timeout=...)
             - plan.do_stocks: get_quotes(plan.tickers)
             - (기업개요) looks_like_ticker(query) 또는 plan에 tickers가 있을 때:
                 · search_company_profile(query, api_key, topk=2) → URL 상위 1~2개
                 · extract_and_summarize_profile(urls, api_key, summarizer=_summarize)
          3) as_completed로 결과 수집. 실패 시 results["errors"]에 '작업명:에러' 저장.
          4) merge_day1_payload(results) 호출해 최종 표준 스키마 dict 반환.
        """
        analysis = asdict(plan) if is_dataclass(plan) else getattr(plan, "__dict__", {})

        results: Dict[str, Any] = {
            "type": "web_results",
            "query": query,
            "analysis": analysis,
            "items": [],
            "tickers": [],
            "errors": [],
            "company_profile": "",
            "profile_sources": [],
        }

        futures = {}
        def submit_profile_job(q: str):
            # 검색 → 상위 URL 정제 → 추출/요약까지 한 번에 처리
            def job() -> Tuple[str, List[str]]:
                search_res = search_company_profile(q, self.tavily_api_key, topk=2, timeout=self.request_timeout)
                urls = [extract_url(r.get("url")) for r in (search_res or []) if r.get("url")]
                urls = [u for u in urls if u][:2]
                if not urls:
                    return "", []
                summary = extract_and_summarize_profile(urls, self.tavily_api_key, summarizer=_summarize)
                return summary or "", urls
            return job

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            # 웹 검색
            if plan.do_web:
                q = " ".join(plan.web_keywords) if plan.web_keywords else query
                futures[ex.submit(search_tavily, q, self.tavily_api_key, self.web_topk, self.request_timeout)] = "web"
            # 주가
            if plan.do_stocks and plan.tickers:
                futures[ex.submit(get_quotes, plan.tickers, self.request_timeout)] = "stock"
            # 기업개요: 질의가 티커처럼 보이거나, 계획에 티커가 있는 경우 시도
            if looks_like_ticker(query) or (plan.tickers and len(plan.tickers) > 0) or ("기업" in query or "회사" in query or "profile" in query.lower()):
                futures[ex.submit(submit_profile_job(query))] = "profile"

            for fut in as_completed(futures):
                kind = futures[fut]
                try:
                    data = fut.result(timeout=self.request_timeout)
                    if kind == "web":
                        # search_tavily 표준 반환(list[dict]) 가정
                        results["items"] = data or []
                    elif kind == "stock":
                        # get_quotes 표준 반환(list[dict]) 가정
                        results["tickers"] = data or []
                    elif kind == "profile":
                        # (summary, urls)
                        summary, urls = data if isinstance(data, tuple) else ("", [])
                        if summary:
                            results["company_profile"] = summary
                        if urls:
                            results["profile_sources"] = urls[:2]
                except Exception as e:
                    results["errors"].append(f"{kind}: {type(e).__name__}: {e}")

        # 표준 스키마로 병합
        return merge_day1_payload(results)

