# experiments/run_tmax_experiment.py
import argparse
import os
import sys
import logging
import itertools # 파라미터 조합을 위해 추가
from typing import Dict, Any, List, Optional, Callable

# --- 프로젝트 루트 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- src 폴더 내 모듈 임포트 ---
from src.main_serc import SERC # SERC 함수
from src.utils import load_config, save_jsonl, get_timestamp
from src.data_loader import load_dataset
# (평가는 evaluate.py로 별도 수행하므로 여기서는 임포트 안함)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_item_serc(item: Dict[str, Any], model_name: str, config: Dict[str, Any],
                         t_max: int, max_facts: int) -> Dict[str, Any]:
    """
    Tuning 실험을 위해 SERC만 실행하는 단일 항목 처리 함수
    (run_experiment.py의 run_single_item을 SERC 전용으로 단순화)
    """
    query = item.get('question', item.get('query'))
    if not query:
        logger.warning("항목에서 'question' 또는 'query' 필드를 찾을 수 없음. 건너<0xEB><0x9B><0x8D>니다.")
        return {"error": "Missing query field"}

    # result_data = {"query": query} # 원본 item에 합칠 것이므로 query만 있어도 됨
    result_data = {} 

    try:
        # SERC 함수 호출 (하이퍼파라미터 직접 전달)
        serc_history = SERC(query=query, model_name=model_name, config=config,
                            t_max=t_max, max_facts_per_group=max_facts)
        
        result_data['serc_result'] = serc_history
        result_data['final_output'] = serc_history.get('final_baseline', '')
        result_data['status'] = 'success'

    except Exception as e:
        logger.error(f"'{query}' 처리 중 오류 발생 (T={t_max}, MF={max_facts}): {e}", exc_info=False) # 로그 간결화
        result_data = {"error": f"Exception during processing: {e}", "status": "error"}

    # 원본 데이터 항목과 결과 합치기
    output_item = {
        **item, 
        "tuning_params": {"t_max": t_max, "max_facts": max_facts},
        "method_result": result_data
    }
    return output_item

# --- 메인 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Run SERC Hyperparameter Tuning (T_max, Max_Facts).")
    
    # --- 기본 인자 ---
    default_config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path,
                        help="Path to the configuration file.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (defined in config) to run tuning on.")
    parser.add_argument("--dataset", type=str, default="hallulens_longwiki",
                        help="Dataset name (key in config data_paths). Default: hallulens_longwiki")
    
    # --- [수정됨] ---
    # 기본값을 None으로 변경하여, 인자를 주지 않으면 전체 데이터를 사용하도록 함
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of data points to process. Default: None (use all data).")
    # --- [수정됨] ---

    parser.add_argument("--output_dir", type=str, default="results/hyperparam_tuning",
                        help="Directory to save the tuning results (relative to project root).")

    # --- 하이퍼파라미터 인자 ---
    parser.add_argument("--t_max_values", nargs='+', type=int, required=True,
                        help="List of T_max values to test (e.g., 1 2 3).")
    parser.add_argument("--max_facts_values", nargs='+', type=int, required=True,
                        help="List of MAX_FACTS_PER_GROUP values to test (e.g., 3 5 7).")

    args = parser.parse_args()

    # --- 파라미터 조합 생성 (Grid Search) ---
    param_grid = list(itertools.product(args.t_max_values, args.max_facts_values))
    
    logging.info(f"--- 하이퍼파라미터 튜닝 시작 ---")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset} (Limit: {args.limit if args.limit is not None else 'All'})") # 로그 수정
    logging.info(f"Parameter Grid ({len(param_grid)} combinations):")
    for t, mf in param_grid:
        logging.info(f"  - T_max={t}, Max_Facts={mf}")

    # --- 설정 로드 ---
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"설정 파일 로드 실패 ({args.config}): {e}")
        return

    # --- 데이터 로드 ---
    dataset_config_key = args.dataset
    relative_path = config.get('data_paths', {}).get(dataset_config_key)
    if not relative_path:
         logger.error(f"Config 파일({args.config})의 'data_paths'에서 '{dataset_config_key}' 키를 찾을 수 없습니다.")
         return
    dataset_path = os.path.join(PROJECT_ROOT, relative_path)
    if not os.path.exists(dataset_path):
        logger.error(f"데이터셋 경로를 찾을 수 없음: {dataset_path}")
        return
    try:
        data = load_dataset(dataset_config_key, dataset_path)
    except Exception as e:
        logger.error(f"데이터셋 로딩 중 오류 발생 ({dataset_path}). 종료합니다.", exc_info=True)
        return

    # --- [수정됨] ---
    # 데이터 제한 로직 수정
    if args.limit and args.limit > 0: # 0보다 큰 limit 값이 주어진 경우
        if args.limit < len(data):
            data = data[:args.limit]
            logging.info(f"데이터를 처음 {args.limit}개로 제한하여 실행합니다.")
        else:
            logging.info(f"데이터셋 {len(data)}개 전체 사용 (Limit {args.limit}개 >= 데이터셋 크기).")
    else:
        logging.info(f"데이터셋 {len(data)}개 전체를 사용합니다 (Limit 인자 없음).")
    # --- [수정됨] ---
        
    if not data:
        logger.error("로드된 데이터가 없습니다. 실험을 중단합니다.")
        return

    # --- 모델 유효성 검사 ---
    if not any(m['name'] == args.model for m in config.get('models', [])):
         logger.error(f"오류: 모델 '{args.model}'이(가) 설정 파일에 정의되지 않았습니다.")
         return

    # --- 파라미터 조합별 실험 실행 ---
    timestamp = get_timestamp()
    from tqdm import tqdm
    
    # 결과 저장 기본 디렉토리
    results_base_dir_abs = os.path.join(PROJECT_ROOT, args.output_dir)

    for t_max, max_facts in param_grid:
        logging.info(f"\n--- 조합 실행: T_max={t_max}, Max_Facts={max_facts} ---")
        
        # 결과 저장 경로 설정
        results_dir = os.path.join(results_base_dir_abs, args.model.replace('/', '_'), args.dataset)
        # 파일명에 limit 정보 추가 (선택적이지만 유용함)
        limit_str = f"_limit{args.limit}" if args.limit else ""
        output_filename = f"serc_t{t_max}_mf{max_facts}{limit_str}_{timestamp}.jsonl"
        output_path = os.path.join(results_dir, output_filename)
        logging.info(f"결과는 다음 경로에 저장됩니다: {output_path}")

        results = []
        # TQDM 루프 (데이터 항목)
        for item in tqdm(data, desc=f"T={t_max}, MF={max_facts}"):
            result_item = run_single_item_serc(item=item, model_name=args.model,
                                               config=config, t_max=t_max, max_facts=max_facts)
            results.append(result_item)

        # 현재 조합에 대한 최종 결과 저장
        try:
            save_jsonl(results, output_path)
            logging.info(f"총 {len(results)}개의 결과가 {output_path}에 저장되었습니다.")
        except Exception as e:
            logging.error(f"최종 결과 저장 실패: {e}")

    logging.info(f"\n--- 모든 하이퍼파라미터 튜닝 조합 실행 완료 ---")

if __name__ == "__main__":
    main()

