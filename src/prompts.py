# serc_framework/prompts.py
from typing import List

# --- 1. 초기 답변 생성 (3.2. Baseline_0) ---
BASELINE_PROMPT_TEMPLATE = """[지시] 다음 질문에 대해 상세히 답하세요:
{query}
[답변] """

# --- 2. 사실 추출 (3.3. ExtractFacts) ---
EXTRACT_FACTS_TEMPLATE = """[지시] 다음 한 문장에 포함된 **'모든'** 핵심 사실 명제를 각각 분리하여 나열하세요. 각 사실은 독립적인 문장으로 작성해주세요. 사실이 없으면 '없음'이라고 답하세요.
[문장]
{sentence}
[추출된 사실 목록] """

# --- 3. 신드롬 생성 관련 (3.4) ---

# 3.4.1. 검사 노드 정의 (1:1 Tagging)
TAG_ONE_FACT_TEMPLATE = """[지시] 관련된 사실들을 효율적으로 그룹화하여 검증하기 위해, 다음 '사실 1개'의 핵심 주제(e.g., 인물, 장소, 시기, 사건, 개념 등)를 가장 잘 나타내는 **단일 단어** 태그 1개만 생성하세요.
[사실]
{fact_text}
[주제 태그] """

# 3.4.4. (1) 메타 질문 Q_t 생성
def generate_group_question_prompt(tag: str, fact_texts_list: List[str]) -> str:
    """그룹 질문 생성 프롬프트를 동적으로 만듭니다."""
    prompt = f"[지시] 다음은 '{tag}'(으)로 태그된 사실 목록입니다. 이 사실들의 정확성을 '하나의 질문'으로 포괄적으로 검증할 수 있는 **가장 구체적인 단일 질문**을 만드세요. 질문 생성이 불가능하면 '없음'을 반환하세요.\n"
    prompt += "[사실 목록]\n"
    if fact_texts_list:
        for f_text in fact_texts_list:
            prompt += f" - {f_text}\n"
    else:
        prompt += " - (사실 목록 비어 있음)\n"
    prompt += "[검증 질문] "
    return prompt

# 3.4.4. (2) 독립 답변 A_t 생성
VERIFICATION_ANSWER_TEMPLATE = """[지시] 다음 질문에 대해 오직 알려진 '사실'에 기반하여, 질문의 핵심에 직접 답하는 간결한 답변을 생성하세요. 추측하거나 부연 설명을 추가하지 마세요.
[질문]
{question}
[사실적 답변] """

# 3.4.4. (3) 신드롬 S_t 계산 (1:1 패리티 검사)
VALIDATE_EVIDENCE_TEMPLATE = """[지시] 정확한 오류 신호(신드롬)를 생성하기 위해, [원본 사실]과 [검증된 증거]가 의미론적으로 **명백히 '모순'**됩니까? 사소한 표현 차이나 정보 부족은 모순이 아닙니다. [예] 또는 [아니오]로만 답하세요.
 - [원본 사실]: {fact_text}
 - [검증된 증거]: {evidence_text}
[판단] """


# 3.5. (1) 탐색 (Locate)
FIND_SENTENCE_TEMPLATE = """[지시] 다음 [원본 텍스트]에서 [대상 사실] 내용과 가장 의미론적으로 일치하는 **단일 문장**(구절이 아닌 완전한 문장)을 정확히 찾아 **그대로 복사하여** 반환하세요. 일치하는 문장이 없으면 '없음'을 반환하세요.
[원본 텍스트]
{current_baseline}
[대상 사실]
{fact_text}
[찾아낸 문장] """

# 3.5. (2) 믿음 갱신 (Correct Belief)
CORRECT_FACT_TEMPLATE = """[지시] 다음 [원본 사실]에 오류가 있다면, 당신의 지식에 기반하여 올바르게 수정한 '핵심 팩트'만 **간결한 단일 문장**으로 생성하세요. 원본 사실에 오류가 없다면 원본 사실을 그대로 반환하세요.
[원본 사실]
{fact_text}
[수정된 팩트] """

# 3.5. (3) 믿음 전파 (Propagate Belief)
REWRITE_SENTENCE_TEMPLATE = """[지시] 다음 [원본 문장]의 내용 중 [수정된 팩트]와 관련된 부분을 반영하여, 문맥에 맞게 **자연스럽게 수정된 단일 문장**을 생성하세요. 만약 수정할 필요가 없다면 [원본 문장]을 그대로 반환하세요.
[원본 문장]
{bad_sentence}
[수정된 팩트]
{correct_fact_text}
[재작성된 문장] """

# --- 6. CoVe (Chain-of-Verification) Baseline Prompts ---

# 6.1. CoVe 2단계: 검증 계획 수립
COVE_PLAN_PROMPT_TEMPLATE = """[지시]
당신은 '초기 답변'의 사실적 정확성을 검증하는 것을 목표로 합니다.
'초기 답변'을 읽고, '원본 질문'의 맥락에서 답변의 사실 여부를 확인하기 위해 필요한 **검증 질문(Verification Questions)** 목록을 생성하세요.
각 질문은 답변의 특정 사실(인물, 장소, 날짜, 통계, 주장 등)을 확인하는 내용이어야 합니다.
질문은 한 줄에 하나씩 작성하세요.

[원본 질문]
{query}

[초기 답변]
{baseline_response}

[검증 질문 목록]
"""

# 6.2. CoVe 3단계: 검증 실행
# (이 단계는 SERC의 VERIFICATION_ANSWER_TEMPLATE을 재사용할 수 있습니다.)
# (별도로 정의할 필요 없이 main_serc.prompt_get_verification_answer 함수 호출)

# 6.3. CoVe 4단계: 최종 답변 생성 (수정)
COVE_REVISE_PROMPT_TEMPLATE = """[지시]
당신은 '초기 답변'을 '검증 결과'를 바탕으로 수정하여 최종 답변을 생성해야 합니다.
'초기 답변'의 내용 중 '검증 결과'와 모순되거나 사실이 아닌 부분을 수정하세요.
검증 결과가 '초기 답변'의 내용을 뒷받침한다면, 해당 내용을 유지하세요.
'원본 질문'에 대한 최종적이고 정확한 답변을 생성하세요.

[원본 질문]
{query}

[초기 답변]
{baseline_response}

[검증 결과 (질문-답변 쌍)]
{verification_evidence}

[최종 수정된 답변]
"""
# --- 6. Ablation Study (Dense) Prompts ---
GENERATE_QUESTION_FOR_ONE_FACT_TEMPLATE = """[지시] 다음 [사실]의 정확성을 직접적으로 검증할 수 있는 '의문문 1개'를 만드세요.
(예: '그는 스페인에서 태어났다.' -> '그는 스페인에서 태어났는가?')
[사실]
{fact_text}
[검증 질문] """
