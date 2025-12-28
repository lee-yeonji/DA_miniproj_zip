# -*- coding: utf-8 -*-
"""
Day3: 정부사업 공고 에이전트
- 역할: 사용자 질의를 받아 Day3 본체(impl/agent.py)의 Day3Agent.handle을 호출
- 결과를 writer로 표/요약 마크다운으로 렌더 → 파일 저장(envelope 포함) → LlmResponse 반환
- 이 파일은 의도적으로 '구현 없음' 상태입니다. TODO만 보고 직접 채우세요.
"""

from __future__ import annotations
from typing import Dict, Any, Optional

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

# Day3 본체
from student.day3.impl.agent import Day3Agent
# 공용 렌더/저장/스키마
from student.common.fs_utils import save_markdown
from student.common.writer import render_day3, render_enveloped
from student.common.schemas import Day3Plan


MODEL = LiteLlm(model="openai/gpt-4o-mini")

def _handle(query: str) -> Dict[str, Any]:

    plan = Day3Plan(
        nipa_topk=3,
        bizinfo_topk=2,
        web_topk=2,
        use_web_fallback=True,
    )
    agent = Day3Agent()
    payload = agent.handle(query, plan)
    return payload

# ------------------------------------------------------------------------------
# TODO[DAY3-A-03] before_model_callback:
#  요구사항
#   1) llm_request에서 사용자 최근 메시지를 찾아 query 텍스트를 꺼낸다.
#   2) _handle(query)로 payload를 만든다.
#   3) writer로 본문 MD를 만든다: render_day3(query, payload)
#   4) 파일 저장: save_markdown(query=query, route='day3', markdown=본문MD)
#   5) envelope로 감싸기: render_enveloped(kind='day3', query=query, payload=payload, saved_path=경로)
#   6) LlmResponse로 최종 마크다운을 반환한다.
#  예외 처리
#   - try/except로 감싸고, 실패 시 "Day3 에러: {e}" 형식의 짧은 메시지로 반환
def _pluck_query(req: LlmRequest) -> str:
    for m in reversed(getattr(req, "contents", []) or []):
        if getattr(m, "role", None) == "user":
            p = getattr(m, "parts", []) or []
            if p and getattr(p[0], "text", None):
                return p[0].text.strip()
    return ""


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    **kwargs,
) -> Optional[LlmResponse]:
    try:
        query = _pluck_query(llm_request)
        if not query:
            raise ValueError("질의 텍스트가 없습니다.")

        # 처리 → payload
        plan = Day3Plan()
        payload = Day3Agent().handle(query, plan)

        # 렌더 → 저장 → envelope
        body_md = render_day3(query, payload)
        saved   = save_markdown(query, "day3", body_md)
        md      = render_enveloped("day3", query, payload, saved)

        # 응답
        return LlmResponse(content=types.Content(parts=[types.Part(text=md)], role="model"))

    except Exception as e:
        err = f"before_model_callback 에러: {type(e).__name__}: {e}"
        return LlmResponse(content=types.Content(parts=[types.Part(text=err)], role="model"))


# ------------------------------------------------------------------------------
# TODO[DAY3-A-04] 에이전트 메타데이터:
#  - name/description/instruction 문구를 명확하게 다듬으세요.
#  - MODEL은 위 TODO[DAY3-A-01]에서 설정한 LiteLlm 인스턴스를 사용합니다.
# ------------------------------------------------------------------------------
agents_config = {
    "day3_gov_fetcher": {
        "name": "Day3 Gov Fetcher",
        "description": "정부사업 공고를 수집하고 기본 필드(title, url, snippet)를 리턴합니다.",
        "instruction": [
            "DO: 신뢰 가능한 공식 도메인 중심으로 수집할 것.",
            "DO: 제목/URL/요약(snippet)을 표준 키로 정리할 것.",
            "DON'T: 광고·중복 결과 포함 금지.",
            "OUTPUT: {'items':[{'title':'','url':'','snippet':''}]}"
        ],
    },

    "day3_gov_profiler": {
        "name": "Day3 Gov Profiler",
        "description": "수집된 공고의 핵심 요건·지원대상·마감 정보를 5문장 이내로 요약합니다.",
        "instruction": [
            "DO: 요건·대상·혜택·마감일을 포함해 간결히 요약할 것.",
            "DO: 불확실한 정보는 표시하거나 제외할 것.",
            "DON'T: 추정/과장 금지.",
            "OUTPUT: {'summary':'...', 'key_points':['...','...']}"
        ],
    },

    "day3_gov_ranker": {
        "name": "Day3 Gov Ranker",
        "description": "쿼리와의 관련도·마감 임박·신뢰도를 기준으로 공고를 정렬합니다.",
        "instruction": [
            "DO: 스코어 근거(간단 규칙 또는 가중치)를 일관되게 적용할 것.",
            "DO: 동일/유사 공고는 병합 혹은 하위로 내릴 것.",
            "DON'T: 임의 필드 추가 금지.",
            "OUTPUT: {'items':[{'title':'','url':'','snippet':'','score':0.0}]}"
        ],
    },

    "day3_callback": {
        "name": "Day3 Callback",
        "description": "LLM 호출 전 결과를 마크다운으로 렌더·저장하고 응답 포맷을 구성합니다.",
        "instruction": [
            "DO: query를 추출한 뒤 `_handle()` 실행 결과를 사용하라.",
            "DO: artifacts/day3 경로에 저장하고 저장 경로를 응답에 포함하라.",
            "DON'T: 민감정보를 로그에 남기지 말 것.",
            "OUTPUT: Markdown 문자열(LlmResponse content)"
        ],
    },
}

day3_gov_agent = Agent(
    name="Day3GovAgent",
    model=MODEL,
    description="정부사업 공고를 수집, 요약, 정렬하는 Day3 에이전트",
    instruction="사용자 질의를 받아 정부사업 공고를 수집 및 처리하고, 결과를 마크다운 형식으로 반환하라.",
    tools=[],
    before_model_callback=before_model_callback,
)
__all__ = ["agents_config"]
