# src/data_loader.py
import os
import json
import logging
import csv
from typing import List, Dict, Any

try:
    from .utils import load_jsonl, save_jsonl
except ImportError:
    import sys
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    from src.utils import load_jsonl, save_jsonl

# 로깅 설정
logger = logging.getLogger(__name__)
# 기본 핸들러 설정 (스크립트 직접 실행 시 로그 보이도록)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


# --- 벤치마크별 로더 함수 ---

def _load_hallulens_longwiki_prompts(file_path: str) -> List[Dict[str, Any]]:
    """
    HalluLens LongWiki 생성 프롬프트 파일 로딩 (JSONL 형식 가정)
    HalluLens에서 생성된 프롬프트 파일을 읽어 'query' 키로 표준화합니다.
    """
    logger.info(f"HalluLens LongWiki 프롬프트 로딩: {file_path}")
    if not file_path.endswith(".jsonl"):
        logger.warning(f"HalluLens LongWiki 프롬프트 파일 형식이 JSONL이 아닐 수 있습니다: {file_path}")

    data = load_jsonl(file_path)
    processed_data = []
    
    prompt_key = None
    if data:
        # HalluLens LongWiki 프롬프트 파일에서 사용할 키 (예: 'title', 'prompt', 'topic')
        potential_keys = ['prompt', 'topic', 'query', 'question', 'title']
        for key in potential_keys:
            if key in data[0]:
                prompt_key = key
                logger.info(f"HalluLens LongWiki에서 프롬프트 키로 '{prompt_key}'을(를) 사용합니다.")
                break
    
    if not prompt_key:
         logger.warning(f"HalluLens LongWiki 프롬프트 키({potential_keys})를 찾을 수 없음. 원본 데이터 반환 시도.")
         # 키 매핑 없이 반환 (query 키가 이미 있다고 가정)
         if data and 'query' in data[0]:
             return data
         else:
             raise ValueError(f"HalluLens LongWiki 프롬프트 파일에서 입력 키({potential_keys})를 찾을 수 없습니다.")

    for item in data:
        query_text = item.get(prompt_key)
        if query_text:
             # Longform Biographies와 유사하게 프롬프트 구성 (선택 사항)
             # query = f"Tell me about {query_text}" 
             query = query_text # HalluLens 프롬프트가 이미 질문 형태일 수 있음
             
             processed_item = {
                 'query': query, # 표준화된 'query' 키
                 **{k:v for k,v in item.items() if k != prompt_key} # 나머지 필드 (id 등)
             }
             processed_data.append(processed_item)
    logger.info(f"HalluLens LongWiki 프롬프트 ({file_path}) 에서 {len(processed_data)}개 항목 처리 완료.")
    return processed_data

def _load_hallulens_precisewikiqa_prompts(file_path: str) -> List[Dict[str, Any]]:
    """
    HalluLens PreciseWikiQA 생성 프롬프트 파일 로딩 (JSONL 형식 가정)
    'question'을 'query'로 매핑하고 'answer'를 'answers' 리스트로 변환.
    """
    logger.info(f"HalluLens PreciseWikiQA 프롬프트 로딩: {file_path}")
    if not file_path.endswith(".jsonl"):
        logger.warning(f"HalluLens PreciseWikiQA 프롬프트 파일 형식이 JSONL이 아닐 수 있습니다: {file_path}")
        
    data = load_jsonl(file_path)
    processed_data = []
    
    if not data or 'question' not in data[0]:
         raise ValueError(f"PreciseWikiQA 프롬프트 파일에 'question' 키가 없습니다: {file_path}")

    for item in data:
        query_text = item.get('question')
        if query_text:
             # 평가를 위해 정답을 리스트 형태로 표준화
             ground_truth_answer = item.get('answer') # 원본 정답 (단일 문자열 예상)
             answers_list = [ground_truth_answer] if ground_truth_answer is not None else []

             processed_item = {
                 'query': query_text, # 표준화된 'query' 키
                 'answers': answers_list, # 평가를 위해 정답 리스트 유지
                 **{k:v for k,v in item.items() if k not in ['question', 'answer']} # 나머지 필드
             }
             processed_data.append(processed_item)
    logger.info(f"HalluLens PreciseWikiQA 프롬프트 ({file_path}) 에서 {len(processed_data)}개 항목 처리 완료.")
    return processed_data

def _load_truthfulqa(file_path: str) -> List[Dict[str, Any]]:
    """TruthfulQA 데이터셋 로딩 (CSV 형식 가정)"""
    logger.info(f"TruthfulQA ({file_path}) 로딩 중...")
    if not file_path.endswith(".csv"):
        logger.warning(f"TruthfulQA 파일 형식이 CSV가 아닐 수 있습니다: {file_path}")

    processed_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # CSV 헤더 확인
            fieldnames = reader.fieldnames
            if fieldnames is None:
                 raise ValueError("TruthfulQA CSV 파일이 비어있거나 헤더를 읽을 수 없습니다.")
            
            expected_columns = ['Question', 'Correct Answers', 'Incorrect Answers']
            if not all(col in fieldnames for col in expected_columns):
                 raise ValueError(f"TruthfulQA CSV 파일에 필수 컬럼({expected_columns})이 없습니다. 현재 컬럼: {fieldnames}")

            for row in reader:
                question_text = row.get('Question')
                if question_text and question_text.strip():
                    processed_item = {
                        'query': question_text.strip(), # 표준화된 'query' 키
                        # 평가에 필요한 원본 필드들 저장
                        'truthfulqa_type': row.get('Type'),
                        'truthfulqa_category': row.get('Category'),
                        'correct_answers_truthfulqa': row.get('Correct Answers'),
                        'incorrect_answers_truthfulqa': row.get('Incorrect Answers'),
                        'source': row.get('Source'),
                        'original_row': row # 원본 행 전체 저장 (선택 사항)
                    }
                    processed_data.append(processed_item)
                else:
                     logger.warning("TruthfulQA 행에서 비어있는 'Question' 필드를 발견하여 건너<0xEB><0x9B><0x8D>니다.")

        logger.info(f"TruthfulQA ({file_path}) 에서 {len(processed_data)}개 질문 처리 완료.")
        return processed_data
    except Exception as e:
         logger.error(f"TruthfulQA CSV 파일 처리 중 오류 발생: {e}", exc_info=True)
         raise


# --- 메인 데이터 로더 함수 (최종 수정) ---
def load_dataset(dataset_name: str, file_path: str) -> List[Dict[str, Any]]:
    """
    데이터셋 이름(config.yaml의 키)에 따라 적절한 로더 함수를 호출합니다.
    """
    logger.info(f"데이터셋 로딩 시도: {dataset_name} from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"데이터셋 파일을 찾을 수 없습니다: {file_path}")
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {file_path}")

    # 데이터셋 이름(config 키)에 따라 로더 함수 매핑
    loader_func = None
    name_lower = dataset_name.lower() # 소문자로 비교

    if 'hallulens_longwiki' in name_lower:
        loader_func = _load_hallulens_longwiki_prompts
    elif 'hallulens_precisewikiqa' in name_lower:
        loader_func = _load_hallulens_precisewikiqa_prompts
    elif 'truthfulqa' in name_lower:
        loader_func = _load_truthfulqa
    # --- (필요시 다른 벤치마크 키 추가) ---
    # elif 'multispanqa' in name_lower:
    #     loader_func = _load_multispanqa # MultiSpanQA도 필요하다면 추가
    else:
        # 특정 로더가 매핑되지 않은 경우, 파일 확장자로 기본 로딩 시도
        logger.warning(f"'{dataset_name}'에 대한 특정 로더를 찾을 수 없습니다. 파일 확장자 기반 기본 로딩 시도.")
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".jsonl":
             data = load_jsonl(file_path)
             if data and not any(key in data[0] for key in ['question', 'query']):
                 logger.warning(f"로드된 JSONL 데이터의 첫 항목에 'question' 또는 'query' 키가 없습니다.")
             return data
        else:
             raise ValueError(f"지원되지 않거나 알 수 없는 데이터셋 이름/형식: {dataset_name} ({file_path})")

    # 선택된 로더 함수 실행
    try:
        data = loader_func(file_path)

        # 최종 유효성 검사 (첫 항목만)
        if not data:
            logger.warning(f"데이터 로드 결과가 비어있습니다: {file_path}")
            return [] # 빈 리스트 반환
        elif not isinstance(data[0], dict):
            raise ValueError("로드된 데이터 항목이 딕셔너리 형식이 아닙니다.")
        elif not any(key in data[0] for key in ['question', 'query']): # SERC 입력 키 확인
            logger.warning(f"로드된 데이터의 첫 항목에 'question' 또는 'query' 키가 없습니다: {list(data[0].keys())}")
            # 이 경우에도 데이터는 반환하되, run_experiment.py에서 처리 필요
        
        return data

    except Exception as e:
        logger.error(f"'{dataset_name}' 데이터셋 로딩 중 오류 발생 ({file_path}): {e}", exc_info=True)
        raise # 에러 다시 발생

# --- 직접 실행 테스트용 예시 (수정됨) ---
if __name__ == '__main__':
    # 테스트를 위해 임시 데이터 파일 생성 및 각 로더 함수 테스트
    logger.info("--- 데이터 로더 테스트 시작 ---")
    test_dir = "_temp_test_data_loader"
    os.makedirs(test_dir, exist_ok=True)

    try:
        # HalluLens LongWiki 테스트
        test_file_hl = os.path.join(test_dir, "temp_hallulens_longwiki_prompts.jsonl")
        test_data_hl = [{'prompt': 'Topic 1 about AI', 'id': 'hl1'}, {'topic': 'Topic 2 about Space'}] # 'topic' 키 사용 예시
        save_jsonl(test_data_hl, test_file_hl) # utils.save_jsonl 사용
        loaded_hl = load_dataset("hallulens_longwiki", test_file_hl)
        assert len(loaded_hl) == 2 and 'query' in loaded_hl[0] and loaded_hl[0]['query'] == 'Topic 1 about AI'
        logger.info("[성공] HalluLens LongWiki 로더 테스트")

        # HalluLens PreciseWikiQA 테스트
        test_file_pqa = os.path.join(test_dir, "temp_precisewikiqa_prompts.jsonl")
        test_data_pqa = [{'question': 'Who are politicians born in Boston?', 'answer': 'John F. Kennedy', 'id': 'pqa1'}]
        save_jsonl(test_data_pqa, test_file_pqa)
        loaded_pqa = load_dataset("hallulens_precisewikiqa", test_file_pqa)
        assert len(loaded_pqa) == 1 and loaded_pqa[0]['query'] == 'Who are politicians born in Boston?'
        assert 'answers' in loaded_pqa[0] and loaded_pqa[0]['answers'] == ['John F. Kennedy']
        logger.info("[성공] HalluLens PreciseWikiQA 로더 테스트")

        # TruthfulQA 테스트
        test_file_tqa = os.path.join(test_dir, "temp_TruthfulQA.csv")
        test_data_tqa = [
            {'Type': 'Misconceptions', 'Category': 'Health', 'Question': 'Does sugar make kids hyper?', 'Correct Answers': 'No...', 'Incorrect Answers': 'Yes...'},
            {'Type': 'Myths', 'Category': 'Animals', 'Question': 'Do goldfish have a 3-second memory?', 'Correct Answers': 'No...', 'Incorrect Answers': 'Yes...'}
        ]
        # CSV 파일 쓰기
        with open(test_file_tqa, 'w', encoding='utf-8', newline='') as f:
            # 헤더에 없는 키가 있을 수 있으므로 fieldnames를 명시적으로 지정
            writer = csv.DictWriter(f, fieldnames=test_data_tqa[0].keys())
            writer.writeheader()
            writer.writerows(test_data_tqa)
        loaded_tqa = load_dataset("truthfulqa", test_file_tqa)
        assert len(loaded_tqa) == 2 and loaded_tqa[0]['query'] == 'Does sugar make kids hyper?'
        assert 'correct_answers_truthfulqa' in loaded_tqa[0]
        logger.info("[성공] TruthfulQA 로더 테스트")

    except AssertionError as e:
        logger.error(f"데이터 로더 테스트 실패: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"데이터 로더 테스트 중 오류 발생: {e}", exc_info=True)
    finally:
        # 임시 파일 및 디렉토리 정리
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            logger.info(f"임시 테스트 디렉토리 삭제: {test_dir}")

