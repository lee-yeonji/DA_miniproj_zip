# -*- coding: utf-8 -*-
"""
Day1: 웹+주가+기업개요 에이전트
- 역할: 사용자 질의를 받아 Day1 본체 호출 → 결과 렌더 → 파일 저장(envelope) → 응답
- 본 파일은 "UI용 래퍼"로, 실질적인 수집/요약 로직은 impl/agent.py 등에 있음.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import re

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from student.common.schemas import Day1Plan
from student.common.writer import render_day1, render_enveloped
from student.common.fs_utils import save_markdown
from student.day1.impl.agent import Day1Agent
from student.day1.impl.web_search import looks_like_ticker

# ------------------------------------------------------------------------------
# TODO[DAY1-A-01] 모델 선택
#  목적:
#    - Day1 래퍼에서 간단한 텍스트 가공(필요 시)나 메타 로직에 쓰일 수 있는 경량 LLM을 지정.
#    - 주 로직은 impl에 있으므로, 여기서는 가벼운 모델이면 충분.
#  지침:
#    - LiteLlm(model="openai/gpt-4o-mini")와 같이 할당.
#    - 모델 문자열은 환경/과금에 맞춰 수정 가능.
# ------------------------------------------------------------------------------
MODEL = LiteLlm(model="openai/gpt-4o-mini")


def _extract_tickers_from_query(query: str) -> List[str]:
    """
    사용자 질의에서 '티커 후보'를 추출합니다.
    예시:
      - "AAPL 주가 알려줘"      → ["AAPL"]
      - "삼성전자 005930 분석"  → ["005930"]
      - "NVDA/TSLA 비교"       → ["NVDA", "TSLA"]
    구현 포인트:
      1) 두 타입 모두 잡아야 함
         - 영문 대문자 1~5자 (미국 티커 일반형) + 선택적 .XX (예: BRK.B 처럼 도메인 일부가 있을 수 있으나, 여기선 단순히 대문자 1~5자를 1차 타깃)
         - 숫자 6자리 (국내 종목코드)
      2) 중복 제거(순서 유지)
      3) 불필요한 특수문자 제거 후 패턴 매칭
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY1-A-02] 구현 지침
    #  - re.findall을 이용해 패턴을 두 번 찾고(영문/숫자), 순서대로 합친 뒤 중복 제거하세요.
    #  - 영문 패턴 예: r"\b[A-Z]{1,5}\b"
    #  - 숫자 패턴 예: r"\b\d{6}\b"
    #  - 반환: ['AAPL', '005930'] 형태의 리스트
    # ----------------------------------------------------------------------------
    import re

    # 1) 특수문자 일부 제거 (예: 슬래시 등)
    clean_query = re.sub(r"[^\w\s]", " ", query.upper())

    # 2) 영문 티커(대문자 1~5자) 추출
    alpha_matches = re.findall(r"\b[A-Z]{1,5}\b", clean_query)

    # 3) 숫자 티커(6자리 숫자) 추출
    digit_matches = re.findall(r"\b\d{6}\b", clean_query)

    # 4) 중복 제거하면서 순서 유지하며 합치기
    seen = set()
    tickers: List[str] = []
    for tk in alpha_matches + digit_matches:
        if tk not in seen:
            seen.add(tk)
            tickers.append(tk)

    return tickers

def _normalize_kr_tickers(tickers: List[str]) -> List[str]:
    """
    한국식 6자리 종목코드에 '.KS'를 붙여 yfinance 호환 심볼로 보정합니다.
    예:
      ['005930', 'AAPL'] → ['005930.KS', 'AAPL']
    구현 포인트:
      1) 각 원소가 6자리 숫자면 뒤에 '.KS'를 붙임
      2) 이미 확장자가 붙은 경우(예: '.KS')는 그대로 둠
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY1-A-03] 구현 지침
    #  - 숫자 6자리 탐지: re.fullmatch(r"\d{6}", sym)
    #  - 맞으면 f"{sym}.KS" 로 변환
    #  - 아니면 원본 유지
    # ----------------------------------------------------------------------------
    import re

    normalized: List[str] = []
    for sym in tickers:
        if re.fullmatch(r"\d{6}", sym):
            normalized.append(f"{sym}.KS")
        else:
            normalized.append(sym)
    return normalized


def _handle(query: str) -> Dict[str, Any]:
    """
    Day1 전체 흐름(오케스트레이션):
      1) 키 준비: os.getenv("TAVILY_API_KEY", "")
      2) 티커 추출 → 한국형 보정
      3) Day1Plan 구성
         - do_web=True (웹 검색은 기본 수행)
         - do_stocks=True/False (티커가 존재하면 True)
         - web_keywords: [query] (필요시 키워드 가공 가능)
         - tickers: 보정된 티커 리스트
      4) Day1Agent(tavily_api_key=...) 인스턴스 생성
      5) agent.handle(query, plan) 호출 → payload(dict) 수신
    반환:
      merge된 표준 스키마 dict (impl/merge.py 참고)
    """
    import os
    from student.day1.impl.web_search import looks_like_ticker

    api_key = os.getenv("TAVILY_API_KEY","")
    tickers = _normalize_kr_tickers(_extract_tickers_from_query(query))

    plan = Day1Plan(
        do_web=True,
        do_stocks=bool(tickers),
        web_keywords=[query],
        tickers=tickers,
        output_style="report",
    )
    agent = Day1Agent(
      tavily_api_key=api_key,
      web_topk=6,
      request_timeout=20,
    )
    return agent.handle(query, plan)


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    **kwargs,
) -> Optional[LlmResponse]:
    """
    UI 엔트리포인트:
      1) llm_request.contents[-1]에서 사용자 메시지 텍스트(query) 추출
      2) _handle(query) 호출 → payload 획득
      3) 본문 마크다운 렌더: render_day1(query, payload)
      4) 저장: save_markdown(query, route='day1', markdown=본문MD) → 경로
      5) envelope: render_enveloped('day1', query, payload, saved_path)
      6) LlmResponse로 반환
      7) 예외시 간단한 오류 텍스트 반환
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY1-A-05] 구현 지침
    #  - last = llm_request.contents[-1]; last.role == "user" 인지 확인
    #  - query = last.parts[0].text
    #  - payload = _handle(query)
    #  - body_md = render_day1(query, payload)
    #  - saved = save_markdown(query=query, route="day1", markdown=body_md)
    #  - md = render_enveloped(kind="day1", query=query, payload=payload, saved_path=saved)
    #  - return LlmResponse(content=types.Content(parts=[types.Part(text=md)], role="model"))
    #  - 예외시: "Day1 에러: {e}"
    # ----------------------------------------------------------------------------
    try:
        last = llm_request.contents[-1]
        if last.role != "user":
            return None
        query = last.parts[0].text
        payload = _handle(query)

        body_md = render_day1(query, payload)
        saved = save_markdown(query=query, route="day1", markdown=body_md)
        md = render_enveloped(
          kind="day1", 
          query=query, 
          payload=payload, 
          saved_path=saved
        )

        return LlmResponse(
          content=types.Content(
            parts=[types.Part(text=md)], 
            role="model"
          )
        )
    
    except Exception as e:
        return LlmResponse(
          content=types.Content(
            parts=[types.Part(text=f"Day1 에러: {e}")], 
            role="model",
          )
        )


# ------------------------------------------------------------------------------
# TODO[DAY1-A-06] Agent 메타데이터 다듬기
#  - name: 영문/숫자/언더스코어만 (하이픈 금지)
#  - description: 에이전트 기능 요약
#  - instruction: 출력 형태/톤/근거표시 등 지침
# ------------------------------------------------------------------------------
day1_web_agent = Agent(
    name="Day1WebAgent",
    model=MODEL,
    description=(
    "사용자의 기업 관련 질문에 대해, 최신 웹 검색 결과를 기반으로 신뢰할 수 있는 "
    "기업 프로필 정보를 신속히 수집·정리합니다. "
    "필요시 기업의 주식 티커를 인식하여 최신 주가 스냅샷을 함께 제공하며, "
    "추출한 내용을 AI 요약기로 간결하게 정리해 5~7문장 수준의 전문적인 설명을 만들어냅니다. "
    "출처 링크 표기를 통해 정보의 신뢰성을 보장하며, 과도한 재무 상세 내용은 생략해 "
    "사용자에게 쉽게 이해되는 명확한 답변을 제공합니다."
  ),
    instruction=(
    "사용자의 질문에 대해 명확하고 간결한 한국어로 답변하라. "
    "답변은 표준 마크다운 형식으로 작성하며, 제목과 리스트를 적절히 활용하여 가독성을 높여라. "
    "웹 검색 결과를 토대로 핵심 사업, 주요 제품과 서비스, 수익원, 주요 시장 및 고객층, 경쟁 우위, 최근 이슈 등을 "
    "5~7문장으로 요약하되, 각 문장은 20~30자 내외로 간결하게 작성할 것. "
    "주가 정보가 포함된 경우 최신 가격과 통화 단위를 명확히 제시하라. "
    "재무 상세 내용이나 불필요한 전문 용어는 피하고, 누구나 이해하기 쉬운 표현을 사용할 것. "
    "모든 주요 정보 아래에는 신뢰할 수 있는 출처의 URL을 마크다운 링크 형태로 반드시 표시하라. "
    "정보가 부족하거나 모호한 경우에는 추가 질문을 제안하거나, 해당 사실을 분명히 알릴 것. "
    "답변 톤은 항상 전문적이고 객관적이며 중립적으로 유지하라."
  ),
    tools=[],
    before_model_callback=before_model_callback,
)
