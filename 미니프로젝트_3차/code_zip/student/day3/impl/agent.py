# -*- coding: utf-8 -*-
"""
Day3Agent: 정부사업 공고 에이전트(Agent-as-a-Tool)
- 입력: query(str), plan(Day3Plan)
- 동작: fetch → normalize → rank
- 출력: {"type":"gov_notices","query": "...","items":[...]}  // items는 정규화된 공고 리스트
"""

from __future__ import annotations
from typing import Dict, Any, List

import os
from student.common.schemas import Day3Plan

# 수집 → 정규화 → 랭크 모듈
from . import fetchers          # NIPA, Bizinfo, 일반 Web 수집
from .normalize import normalize_all   # raw → 공통 스키마 변환
from .rank import rank_items           # 쿼리와의 관련도/마감일/신뢰도 등으로 정렬


def _set_source_topk(plan: Day3Plan) -> Day3Plan:
    """
    fetchers 모듈의 (기본)소스별 TopK 상수와 plan 값을 싱크.
    - 이 함수는 실습 편의를 위해 제공. 내부에서 fetchers.NIPA_TOPK 등 값 갱신.
    """
    def _coerce_min1(v, fallback: int) -> int:
        try:
            n = int(v)
        except Exception:
            n = int(fallback)
        return max(1, n)

    # 현재 fetchers의 기본값을 안전하게 참조
    default_nipa    = getattr(fetchers, "NIPA_TOPK", 10)
    default_bizinfo = getattr(fetchers, "BIZINFO_TOPK", 10)
    default_web     = getattr(fetchers, "WEB_TOPK", 5)

    plan.nipa_topk    = _coerce_min1(getattr(plan, "nipa_topk", default_nipa), default_nipa)
    plan.bizinfo_topk = _coerce_min1(getattr(plan, "bizinfo_topk", default_bizinfo), default_bizinfo)
    plan.web_topk     = _coerce_min1(getattr(plan, "web_topk", default_web), default_web)

    # fetchers 상수에 반영
    fetchers.NIPA_TOPK    = plan.nipa_topk
    fetchers.BIZINFO_TOPK = plan.bizinfo_topk
    fetchers.WEB_TOPK     = plan.web_topk

    return plan


class Day3Agent:
    def __init__(self):
        """
        외부 API 키 등 환경변수 확인 (없어도 동작은 하되 결과가 빈 배열일 수 있음)
        - 예: os.getenv("TAVILY_API_KEY", "")
        """
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self.default_headers = {"User-Agent": os.getenv("HTTP_USER_AGENT", "Day3Agent/1.0")}
        
    def _safe_fetch(self, func: callable, *args, **kwargs) -> List[dict]:
        """
        주어진 함수(예: fetchers.fetch_nipa)를 호출하고, 예외 발생 시
        에이전트가 중단되는 것을 방지하기 위해 빈 리스트를 반환합니다.
        """
        try:
            # func(*args, **kwargs)를 호출하고 결과를 반환
            return func(*args, **kwargs)
        except Exception as e:
            # 오류 발생 시 오류 메시지를 출력하고 빈 리스트 반환
            print(f"[Day3Agent] Error during fetch: {func.__name__} failed with {type(e).__name__}: {e}")
            return []

    def handle(self, query: str, plan: Day3Plan = Day3Plan()) -> Dict[str, Any]:
        """
        End-to-End 파이프라인:
          1) _set_source_topk(plan)  // 입력 plan의 topk를 fetchers에 반영
          2) fetch 단계
             - NIPA: fetchers.fetch_nipa(query, plan.nipa_topk)
             - Bizinfo: fetchers.fetch_bizinfo(query, plan.bizinfo_topk)
             - Web fallback(옵션): plan.use_web_fallback and plan.web_topk > 0 이면 fetchers.fetch_web(...)
             → raw 리스트에 모두 누적
          3) normalize 단계: normalize_all(raw)
             - 출처가 제각각인 raw를 공통 스키마(제목/title, URL, 마감/기간, 주체/부처 등)로 변환
          4) rank 단계: rank_items(norm, query)
             - 질의 관련도, 마감 임박도, 신뢰도 점수 등을 반영해 정렬/필터링
          5) 결과 페이로드 구성:
             { "type": "gov_notices", "query": query, "items": ranked }
        예외 처리:
          - 각 단계에서 예외가 난다면 최소한 비어 있는 리스트라도 반환하도록 하거나,
            상위에서 try/except로 감싼다(이번 과제에선 간단 구현 권장).
        """
        # 1) 소스별 TopK 싱크
        plan = _set_source_topk(plan)

        raw: List[dict] = []

        # 2) fetch 단계
        # 각각의 fetch에서 예외가 나더라도 전체는 계속 진행
        if getattr(plan, "nipa_topk", 0) > 0:
            raw += self._safe_fetch(fetchers.fetch_nipa, query, plan.nipa_topk)
        if getattr(plan, "bizinfo_topk", 0) > 0:
            raw += self._safe_fetch(fetchers.fetch_bizinfo, query, plan.bizinfo_topk)
        if getattr(plan, "use_web_fallback", False) and getattr(plan, "web_topk", 0) > 0:
            raw += self._safe_fetch(fetchers.fetch_web, query, plan.web_topk)

        # 3) normalize
        try:
            norm = normalize_all(raw) if raw else []
        except Exception:
            norm = []

        # 4) rank
        try:
            ranked = rank_items(norm, query) if norm else []
        except Exception:
            ranked = []

        # 5) 결과 페이로드
        return {
            "type": "gov_notices",
            "query": query,
            "items": ranked,
        }
