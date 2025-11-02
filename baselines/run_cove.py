import logging
import re
from typing import Dict, Any, List

# --- src 폴더의 모듈 임포트 ---
# run_experiment.py가 프로젝트 루트를 sys.path에 추가한다고 가정
try:
    from src.model_wrappers import generate
    from src import prompts
    from src.main_serc import prompt_baseline, prompt_get_verification_answer
except ImportError:
    # 이 파일 단독 실행 시 경로 문제 해결 (선택적)
    import sys, os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(PROJECT_ROOT)
    from src.model_wrappers import generate
    from src import prompts
    from src.main_serc import prompt_baseline, prompt_get_verification_answer

# 로깅 설정
logger = logging.getLogger(__name__)

def _parse_cove_questions(text: str) -> List[str]:
    """CoVe 계획 단계에서 생성된 질문 목록 문자열을 파싱합니다."""
    # 간단히 줄바꿈으로 분리 (모델이 번호 매기기 등을 할 경우 정규식 필요)
    lines = text.strip().splitlines()
    # 번호 매기기(e.g., "1. ", "Q1: ") 제거 (간단한 정규식)
    questions = [re.sub(r"^\s*(\d+\.|Q\d:)\s*", "", line).strip() for line in lines]
    return [q for q in questions if q and q != ""] # 빈 줄 제거

def _format_qa_evidence(qa_list: List[Dict[str, str]]) -> str:
    """검증 Q&A 리스트를 최종 프롬프트에 넣을 문자열로 포맷팅합니다."""
    if not qa_list:
        return "검증 결과 없음."
    
    formatted_str = ""
    for i, qa in enumerate(qa_list, 1):
        formatted_str += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n\n"
    return formatted_str.strip()

def run_cove(query: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    CoVe (Chain-of-Verification)의 4단계를 실행합니다.
    (run_experiment.py의 run_single_item에서 호출됨)
    """
    
    cove_history = {'query': query, 'steps': {}}
    
    try:
        # --- 1단계: 초기 답변 생성 ---
        logging.info("  [CoVe 1/4] 초기 답변 생성 중...")
        # SERC의 prompt_baseline 재사용
        initial_baseline = prompt_baseline(query, model_name, config)
        cove_history['steps']['1_initial_baseline'] = initial_baseline
        logging.info(f"    CoVe Baseline: {initial_baseline[:100]}...")

        # --- 2단계: 검증 계획 수립 ---
        logging.info("  [CoVe 2/4] 검증 계획 수립 중...")
        plan_prompt = prompts.COVE_PLAN_PROMPT_TEMPLATE.format(
            query=query,
            baseline_response=initial_baseline
        )
        plan_response = generate(plan_prompt, model_name, config)
        verification_questions = _parse_cove_questions(plan_response)
        cove_history['steps']['2_verification_plan'] = {
            'raw_response': plan_response,
            'parsed_questions': verification_questions
        }
        logging.info(f"    CoVe Plan: {len(verification_questions)}개 질문 생성됨.")

        # --- 3단계: 검증 실행 ---
        logging.info(f"  [CoVe 3/4] {len(verification_questions)}개 질문 검증 실행 중...")
        verification_results = []
        for q in verification_questions:
            answer = prompt_get_verification_answer(q, model_name, config)
            verification_results.append({'question': q, 'answer': answer})
            logging.debug(f"      Q: {q}\n      A: {answer[:100]}...")
        cove_history['steps']['3_verification_results'] = verification_results
        logging.info(f"    CoVe Execution: {len(verification_results)}개 답변 완료.")

        # --- 4단계: 최종 답변 생성 (수정) ---
        logging.info("  [CoVe 4/4] 최종 답변 생성 중...")
        evidence_str = _format_qa_evidence(verification_results)
        revise_prompt = prompts.COVE_REVISE_PROMPT_TEMPLATE.format(
            query=query,
            baseline_response=initial_baseline,
            verification_evidence=evidence_str
        )
        final_output = generate(revise_prompt, model_name, config)
        cove_history['steps']['4_final_output'] = final_output
        cove_history['final_output'] = final_output # 최종 결과 키
        logging.info(f"    CoVe Final Output: {final_output[:100]}...")

    except Exception as e:
        logging.error(f"CoVe 실행 중 오류 발생: {e}", exc_info=True)
        cove_history['error'] = str(e)
        cove_history['final_output'] = f"Error during CoVe: {e}"

    return cove_history