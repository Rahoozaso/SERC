# src/data_loader.py
import os
import json
import logging
import pandas as pd
import csv # CSV 로딩 위해 추가
from typing import List, Dict, Any
try:
    from .utils import load_jsonl
    from.utils import save_jsonl
except ImportError:
    # 직접 실행 테스트 시 상위 폴더 임포트 허용 (권장하지 않음)
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_jsonl

# 로깅 설정
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


# --- 벤치마크별 로더 함수 ---

def _load_hallulens_longwiki_prompts(file_path: str) -> List[Dict[str, Any]]:
    """HalluLens LongWiki 생성 프롬프트 파일 로딩 (JSONL 형식 가정)"""
    if not file_path.endswith(".jsonl"):
        logger.warning(f"HalluLens LongWiki 프롬프트 파일 형식이 JSONL이 아닐 수 있습니다: {file_path}")

    data = load_jsonl(file_path)
    processed_data = []
    prompt_key = None
    if data:
        potential_keys = ['prompt', 'topic', 'query', 'question']
        for key in potential_keys:
            if key in data[0]:
                prompt_key = key
                break

    if not prompt_key:
         logger.warning(f"HalluLens LongWiki 프롬프트 파일에서 입력 프롬프트 키({potential_keys})를 찾을 수 없습니다.")
         return data 

    for item in data:
        query_text = item.get(prompt_key)
        if query_text:
             processed_item = {
                 'query': query_text,
                 'hallulens_id': item.get('id'), # 원래 ID 유지
                 # LongWiki는 원래 응답이 없으므로 추가 정보는 적을 수 있음
                 **{k:v for k,v in item.items() if k != prompt_key and k != 'query'} # 나머지 필드 추가
             }
             processed_data.append(processed_item)
        else:
             logger.warning(f"HalluLens LongWiki 항목에서 '{prompt_key}' 필드를 찾을 수 없음: {item.get('id')}")

    logger.info(f"HalluLens LongWiki 프롬프트 ({file_path}) 에서 {len(processed_data)}개 항목 처리 완료.")
    return processed_data

def _load_multispanqa(file_path: str) -> List[Dict[str, Any]]:
    """MultiSpanQA 데이터셋 로딩 (JSON 형식 가정)"""
    if not file_path.endswith(".json"):
        logger.warning(f"MultiSpanQA 파일 형식이 JSON이 아닐 수 있습니다: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        loaded_json = json.load(f)

    # MultiSpanQA 데이터 구조 확인 필요 (보통 'data' 키 아래 리스트)
    data_list = None
    if isinstance(loaded_json, dict) and 'data' in loaded_json and isinstance(loaded_json['data'], list):
         data_list = loaded_json['data']
    elif isinstance(loaded_json, list): # 파일 자체가 리스트일 수도 있음
         data_list = loaded_json
    else:
        raise ValueError(f"MultiSpanQA JSON 파일이 예상된 형식이 아닙니다 (dict with 'data' list or list expected): {file_path}")

    processed_data = []
    for article in data_list:
        for paragraph in article.get('paragraphs', []):
            for qa in paragraph.get('qas', []):
                question_text = qa.get('question')
                qa_id = qa.get('id', qa.get('qid')) # ID 필드 확인
                # 답변 필드 확인 (is_impossible 처리 등 필요할 수 있음)
                answers = [ans.get('text') for ans in qa.get('answers', []) if ans.get('text')]

                if question_text:
                    processed_item = {
                        'query': question_text, # 표준화된 'query' 키 사용
                        'multispanqa_id': qa_id,
                        'answers': answers, # 평가를 위해 정답 유지
                        # 필요시 context 등 다른 정보 추가
                    }
                    processed_data.append(processed_item)
                else:
                    logger.warning(f"MultiSpanQA 항목에서 'question' 필드를 찾을 수 없음: ID={qa_id}")

    logger.info(f"MultiSpanQA ({file_path}) 에서 {len(processed_data)}개 질문 처리 완료.")
    return processed_data

def _load_truthfulqa(file_path: str) -> List[Dict[str, Any]]:
    """TruthfulQA 데이터셋 로딩 (CSV 형식 가정)"""
    if not file_path.endswith(".csv"):
        logger.warning(f"TruthfulQA 파일 형식이 CSV가 아닐 수 있습니다: {file_path}")

    processed_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            expected_columns = ['Question', 'Correct Answers', 'Incorrect Answers']
            if not all(col in reader.fieldnames for col in expected_columns):
                 raise ValueError(f"TruthfulQA CSV 파일에 필수 컬럼({expected_columns})이 없습니다. 현재 컬럼: {reader.fieldnames}")

            for row in reader:
                question_text = row.get('Question')
                if question_text:
                    processed_item = {
                        'query': question_text, # 표준화된 'query' 키 사용
                        'truthfulqa_type': row.get('Type'),
                        'truthfulqa_category': row.get('Category'),
                        'correct_answers': row.get('Correct Answers'), # 평가용
                        'incorrect_answers': row.get('Incorrect Answers'), # 평가용
                        'source': row.get('Source')
                    }
                    processed_data.append(processed_item)
                else:
                     logger.warning("TruthfulQA 행에서 'Question' 필드를 찾을 수 없음.")

        logger.info(f"TruthfulQA ({file_path}) 에서 {len(processed_data)}개 질문 처리 완료.")
        return processed_data
    except Exception as e:
         logger.error(f"TruthfulQA CSV 파일 처리 중 오류 발생: {e}", exc_info=True)
         raise # 에러 다시 발생


# --- 메인 데이터 로더 함수 (수정됨) ---
def load_dataset(dataset_name: str, file_path: str) -> List[Dict[str, Any]]:
    """
    데이터셋 이름에 따라 적절한 로더 함수를 호출합니다.
    """
    logger.info(f"데이터셋 로딩 시도: {dataset_name} from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"데이터셋 파일을 찾을 수 없습니다: {file_path}")
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {file_path}")

    # 데이터셋 이름에 따라 적절한 로더 함수 선택
    loader_func = None
    name_lower = dataset_name.lower() # 소문자로 비교

    # ⭐ 선택된 벤치마크에 대한 분기
    if 'hallulens_longwiki' in name_lower: # config 키 이름과 일치
        loader_func = _load_hallulens_longwiki_prompts
    elif 'multispanqa' in name_lower:
        loader_func = _load_multispanqa
    elif 'truthfulqa' in name_lower:
        loader_func = _load_truthfulqa
    # --- (기타 필요한 벤치마크 로더 추가) ---
    # elif 'hotpotqa' in name_lower:
    #     loader_func = _load_hotpotqa
    # elif 'halueval' in name_lower: # HaluEval 사용 시
    #     loader_func = _load_halueval
    else:
        # 특정 로더 없으면 파일 확장자 기반으로 시도 (JSONL만 예시)
        logger.warning(f"'{dataset_name}'에 대한 특정 로더를 찾을 수 없습니다. 파일 확장자 기반 기본 로딩 시도.")
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".jsonl":
             # JSONL은 일반적으로 각 라인이 딕셔너리, 'query'/'question' 키 예상
             data = load_jsonl(file_path)
             # 첫 항목에서 키 확인 (선택 사항)
             if data and not any(key in data[0] for key in ['question', 'query']):
                 logger.warning(f"로드된 JSONL 데이터의 첫 항목에 'question' 또는 'query' 키가 없습니다: {list(data[0].keys())}")
             return data
        # elif file_extension == ".tsv": ...
        # elif file_extension == ".json": ...
        else:
             raise ValueError(f"지원되지 않거나 알 수 없는 데이터셋 이름/형식: {dataset_name} ({file_path})")

    # 선택된 로더 함수 실행
    try:
        data = loader_func(file_path)

        # 최종 유효성 검사 (첫 항목만)
        if not data:
            logger.warning(f"데이터 로드 결과가 비어있습니다: {file_path}")
        elif not isinstance(data[0], dict):
            raise ValueError("로드된 데이터 항목이 딕셔너리 형식이 아닙니다.")
        elif not any(key in data[0] for key in ['question', 'query']): # SERC 입력 키 확인
            logger.warning(f"로드된 데이터의 첫 항목에 'question' 또는 'query' 키가 없습니다: {list(data[0].keys())}")

        return data

    except Exception as e:
        logger.error(f"'{dataset_name}' 데이터셋 로딩 중 오류 발생 ({file_path}): {e}", exc_info=True)
        raise

# --- 직접 실행 테스트용 예시 (수정됨) ---
if __name__ == '__main__':
    # 테스트를 위해 임시 데이터 파일 생성 및 각 로더 함수 테스트
    logger.info("--- 데이터 로더 테스트 시작 ---")
    test_dir = "_temp_test_data"
    os.makedirs(test_dir, exist_ok=True)

    try:
        # HalluLens LongWiki 테스트
        test_file_hl = os.path.join(test_dir, "temp_hallulens_longwiki_prompts.jsonl")
        test_data_hl = [{'prompt': 'Topic 1 about AI', 'id': 'hl1'}, {'prompt': 'Topic 2 about Space'}]
        save_jsonl(test_data_hl, test_file_hl) # utils.save_jsonl 사용
        loaded_hl = load_dataset("hallulens_longwiki", test_file_hl)
        assert len(loaded_hl) == 2 and 'query' in loaded_hl[0]
        logger.info("HalluLens 로더 테스트 성공")

        # MultiSpanQA 테스트 (간단 구조 가정)
        test_file_ms = os.path.join(test_dir, "temp_multispanqa_dev.json")
        test_data_ms = [{'paragraphs': [{'qas': [{'question': 'Q1?', 'id': 'ms1', 'answers': [{'text': 'A1'}]}]}]}]
        with open(test_file_ms, 'w', encoding='utf-8') as f: json.dump(test_data_ms, f)
        loaded_ms = load_dataset("multispanqa_dev", test_file_ms)
        assert len(loaded_ms) == 1 and loaded_ms[0]['query'] == 'Q1?'
        logger.info("MultiSpanQA 로더 테스트 성공")

        # TruthfulQA 테스트
        test_file_tqa = os.path.join(test_dir, "temp_TruthfulQA.csv")
        test_data_tqa = [
            {'Type': 'Misconceptions', 'Category': 'Health', 'Question': 'Does sugar make kids hyper?', 'Best Answer': 'No scientific evidence...', 'Correct Answers': 'No...', 'Incorrect Answers': 'Yes...'},
            {'Type': 'Myths', 'Category': 'Animals', 'Question': 'Do goldfish have a 3-second memory?', 'Best Answer': 'No, they can remember...', 'Correct Answers': 'No...', 'Incorrect Answers': 'Yes...'}
        ]
        # CSV 파일 쓰기
        with open(test_file_tqa, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=test_data_tqa[0].keys())
            writer.writeheader()
            writer.writerows(test_data_tqa)
        loaded_tqa = load_dataset("truthfulqa", test_file_tqa)
        assert len(loaded_tqa) == 2 and loaded_tqa[0]['query'] == 'Does sugar make kids hyper?'
        logger.info("TruthfulQA 로더 테스트 성공")

    except Exception as e:
        logger.error(f"데이터 로더 테스트 중 오류 발생: {e}", exc_info=True)
    finally:
        # 임시 파일 및 디렉토리 정리
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            logger.info(f"임시 테스트 디렉토리 삭제: {test_dir}")
