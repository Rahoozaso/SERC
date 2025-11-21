import os
import json
import logging
import csv
from typing import List, Dict, Any
import pandas as pd

try:
    # 일반적인 실행 시 (예: experiments/run_experiment.py에서 호출)
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

def _load_longform_biographies(file_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Longform Biographies 프롬프트 로딩: {file_path}")
    processed_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                entity_name = ""
                item_data = {} # 원본 .jsonl의 다른 데이터 저장용

                # .jsonl 파일 형식인지(.txt인지) 확인
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        # FActScore/HalluLens LongWiki 등에서 사용하는 키 탐색
                        potential_keys = ['topic', 'entity', 'name', 'title', 'prompt']
                        for key in potential_keys:
                            if key in item:
                                entity_name = item.get(key)
                                item_data = {k:v for k,v in item.items() if k != key}
                                break
                        if not entity_name:
                             logger.warning(f"JSONL {i+1}번째 줄에서 엔티티 키를 찾을 수 없음: {line}")
                             continue
                    else: # JSONL이지만 딕셔너리 아님
                         entity_name = str(item) # 혹시 모르니 문자열로
                except json.JSONDecodeError:
                    # 단순 .txt 파일 (한 줄에 엔티티 이름만 있음)
                    entity_name = line

                if entity_name: 
                    # CoVe 논문(4.1.4절)에서 사용한 프롬프트 형식
                    query = f"Tell me a bio of {entity_name}"
                    
                    processed_item = {
                        'query': query, # 표준화된 'query' 키
                        'topic': entity_name, # FactScore 평가 시 'topic'이 필요함
                        **item_data # .jsonl이었을 경우 나머지 필드
                    }
                    processed_data.append(processed_item)

    except FileNotFoundError:
         logger.error(f"엔티티 파일({file_path})을 찾을 수 없습니다.")
         raise
    except Exception as e:
         logger.error(f"엔티티 파일({file_path}) 처리 중 오류: {e}", exc_info=True)
         raise
             
    logger.info(f"Longform Biographies ({file_path}) 에서 {len(processed_data)}개 프롬프트 생성 완료.")
    return processed_data

def _load_hallulens_precisewikiqa_prompts(file_path: str) -> List[Dict[str, Any]]:
    logger.info(f"HalluLens PreciseWikiQA 프롬프트 로딩: {file_path}")
    if not file_path.endswith(".jsonl"):
        logger.warning(f"HalluLens PreciseWikiQA 프롬프트 파일 형식이 JSONL이 아닐 수 있습니다: {file_path}")
        
    data = load_jsonl(file_path) # utils.py의 함수 사용
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

    if 'longform_bio' in name_lower: # 'hallulens_longwiki' 대신
        loader_func = _load_longform_biographies
    elif 'hallulens_precisewikiqa' in name_lower:
        loader_func = _load_hallulens_precisewikiqa_prompts
    elif 'truthfulqa' in name_lower:
        loader_func = _load_truthfulqa
    else:
        # 특정 로더가 매핑되지 않은 경우, 파일 확장자로 기본 로딩 시도
        logger.warning(f"'{dataset_name}'에 대한 특정 로더를 찾을 수 없습니다. 파일 확장자 기반 기본 로딩 시도.")
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".jsonl":
             data = load_jsonl(file_path)
        elif file_extension == ".json":
             with open(file_path, 'r', encoding='utf-8') as f:
                 data = json.load(f)
             if not isinstance(data, list):
                 raise ValueError("기본 JSON 로더는 리스트 형태만 지원합니다.")
        elif file_extension == ".csv":
             data = pd.read_csv(file_path).to_dict('records')
        elif file_extension == ".txt":
             # Longform Bio와 동일하게 텍스트 라인 로더로 가정
             logger.info(".txt 파일 감지. Longform Biographies 로더를 사용합니다.")
             loader_func = _load_longform_biographies
        else:
             raise ValueError(f"지원되지 않는 파일 확장자: {file_extension}")
        
        # loader_func가 위에서 할당되지 않았다면, data는 이미 로드됨
        if loader_func is None:
             # 공통 키 유효성 검사
             if data and not any(key in data[0] for key in ['question', 'query']):
                 logger.warning(f"로드된 데이터의 첫 항목에 'question' 또는 'query' 키가 없습니다.")
             return data
        
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
    logger.info("--- 데이터 로더 테스트 시작 ---")
    test_dir = "_temp_test_data_loader"
    os.makedirs(test_dir, exist_ok=True)

    try:
        # 1. Longform Biographies (TXT) 테스트
        test_file_bio_txt = os.path.join(test_dir, "temp_bio_entities.txt")
        test_data_bio_txt = "Leonardo da Vinci\nMarie Curie"
        with open(test_file_bio_txt, 'w', encoding='utf-8') as f:
            f.write(test_data_bio_txt)
        
        loaded_bio = load_dataset("longform_bio", test_file_bio_txt)
        assert len(loaded_bio) == 2 and 'query' in loaded_bio[0]
        assert loaded_bio[0]['query'] == 'Tell me a bio of Leonardo da Vinci'
        assert 'topic' in loaded_bio[0] and loaded_bio[0]['topic'] == 'Leonardo da Vinci'
        logger.info("[성공] Longform Biographies (.txt) 로더 테스트")

        # 2. HalluLens PreciseWikiQA 테스트
        test_file_pqa = os.path.join(test_dir, "temp_precisewikiqa_prompts.jsonl")
        test_data_pqa = [{'question': 'Who are politicians born in Boston?', 'answer': 'John F. Kennedy', 'id': 'pqa1'}]
        save_jsonl(test_data_pqa, test_file_pqa) # utils.save_jsonl 사용
        loaded_pqa = load_dataset("hallulens_precisewikiqa", test_file_pqa)
        assert len(loaded_pqa) == 1 and loaded_pqa[0]['query'] == 'Who are politicians born in Boston?'
        assert 'answers' in loaded_pqa[0] and loaded_pqa[0]['answers'] == ['John F. Kennedy']
        logger.info("[성공] HalluLens PreciseWikiQA 로더 테스트")

        # 3. TruthfulQA 테스트
        test_file_tqa = os.path.join(test_dir, "temp_TruthfulQA.csv")
        test_data_tqa = [
            {'Type': 'Misconceptions', 'Category': 'Health', 'Question': 'Does sugar make kids hyper?', 'Correct Answers': 'No...', 'Incorrect Answers': 'Yes...'},
            {'Type': 'Myths', 'Category': 'Animals', 'Question': 'Do goldfish have a 3-second memory?', 'Correct Answers': 'No...', 'Incorrect Answers': 'Yes...'}
        ]
        # CSV 파일 쓰기
        with open(test_file_tqa, 'w', encoding='utf-8', newline='') as f:
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