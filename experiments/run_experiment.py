import argparse
import os
import sys
import logging
import json 
import pandas as pd
from typing import Dict, Any, List, Optional

# --- 프로젝트 루트 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- src 폴더 내 모듈 임포트 ---
from src.main_serc import SERC, prompt_baseline 
from src.utils import load_config, save_jsonl, get_timestamp 
from src.data_loader import load_dataset

# --- 베이스라인 임포트 (프로젝트 루트의 baselines 폴더 가정) ---
from baselines.run_cove import run_cove

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 단일 항목 처리 함수 ---
def run_single_item(item: Dict[str, Any], method: str, model_name: str, config: Dict[str, Any],
                    t_max: Optional[int]) -> Dict[str, Any]:
    """데이터셋의 단일 항목에 대해 지정된 방법론을 실행합니다."""
    query = item.get('question', item.get('query'))
    if not query:
        logging.warning("항목에서 'question' 또는 'query' 필드를 찾을 수 없음. 건너뜁니다.")
        return {"error": "Missing query field"}

    logging.info(f"질문 처리 중: '{query[:100]}...'")
    result_data = {"query": query} # 결과 저장용

    try:
        if method == 'serc':
            logging.info("SERC 방법론 실행...")
            serc_history = SERC(query=query, model_name=model_name, config=config,
                                t_max=t_max)
                                # [!!! 수정됨 !!!] max_facts_per_group 제거
            result_data['serc_result'] = serc_history
            result_data['final_output'] = serc_history.get('final_baseline', '')

        elif method == 'baseline':
            logging.info("Baseline 생성 실행...")
            baseline_output = prompt_baseline(query=query, model_name=model_name, config=config)
            result_data['baseline_result'] = {'initial_baseline': baseline_output}
            result_data['final_output'] = baseline_output

        elif method == 'cove':
            logging.info("CoVe 방법론 실행...")
            cove_history = run_cove(query=query, model_name=model_name, config=config)
            result_data['cove_result'] = cove_history
            result_data['final_output'] = cove_history.get('final_output', 'Error: CoVe did not return final_output')

        else:
            logging.error(f"알 수 없는 방법론: {method}")
            result_data = {"error": f"Unknown method: {method}"}

    except Exception as e:
        logging.error(f"'{query}' 처리 중 오류 발생 ({method}): {e}", exc_info=True)
        result_data = {"error": f"Exception during processing: {e}"}

    output_item = {**item, "method_result": result_data, "method_used": method}
    return output_item

# --- 메인 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Run SERC framework experiments.")
    default_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    
    parser.add_argument("--config", type=str, default=default_config_path,
                        help="Path to the configuration file.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (must be defined in config file).")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (key in config data_paths).")
    
    parser.add_argument("--method", type=str, required=True,
                        choices=['serc', 'baseline', 'cove'], 
                        help="Method to run (serc, baseline, cove).")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Optional suffix for the output filename.")
    parser.add_argument("--t_max", type=int, default=None,
                        help="Override default T_max for SERC.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of data points to process for testing.")

    
    args = parser.parse_args()

    logging.info(f"실험 시작: Model={args.model}, Dataset={args.dataset}, Method={args.method}")

    # --- 설정 로드 ---
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"설정 파일 로드 실패 ({args.config}): {e}")
        return

    # --- 데이터 로드 ---
    dataset_config_key = args.dataset
    relative_path = config.get('data_paths', {}).get(dataset_config_key)
    if not relative_path:
         logging.error(f"Config 파일({args.config})의 'data_paths'에서 '{dataset_config_key}' 키를 찾을 수 없습니다.")
         return
    dataset_path = os.path.join(PROJECT_ROOT, relative_path)

    if not os.path.exists(dataset_path):
        logging.error(f"데이터셋 경로를 찾을 수 없음: '{dataset_config_key}' (계산된 경로: {dataset_path})")
        return
    try:
        data = load_dataset(dataset_config_key, dataset_path)
    except FileNotFoundError as e:
        logging.error(f"데이터셋 파일({dataset_path})을 찾을 수 없습니다: {e}")
        return
    except ValueError as e:
         logging.error(f"데이터셋 파일({dataset_path}) 형식 오류 또는 처리 불가: {e}")
         return
    except Exception as e: 
        logging.error(f"데이터셋 로딩 중 예상치 못한 오류 발생 ({dataset_path}): {e}", exc_info=True)
        return

    if not data:
        logging.error("로드된 데이터가 없습니다. 실험을 중단합니다.")
        return
    if args.limit:
        data = data[:args.limit]
        logging.info(f"데이터를 처음 {args.limit}개로 제한하여 실행합니다.")
    logging.info(f"{args.dataset} 데이터셋에서 {len(data)}개의 예제를 로드했습니다.")
    if not any(m['name'] == args.model for m in config.get('models', [])):
         logging.error(f"오류: 모델 '{args.model}'이(가) 설정 파일({args.config})에 정의되지 않았습니다.")
         return

    # --- 결과 저장 경로 설정 ---
    results_base_dir_rel = config.get('results_base_dir', 'results')
    results_base_dir_abs = os.path.join(PROJECT_ROOT, results_base_dir_rel)
    results_dir = os.path.join(results_base_dir_abs, args.model.replace('/', '_'), args.dataset)

    timestamp = get_timestamp()
    filename_parts = [args.method]
    if args.method == 'serc':
        t_max_val = args.t_max if args.t_max is not None else config.get('default_t_max', 'def')
        filename_parts.append(f"t{t_max_val}")
    if args.output_suffix:
        filename_parts.append(args.output_suffix)
    filename_parts.append(timestamp)

    output_filename = "_".join(map(str, filename_parts)) + ".jsonl"
    output_path = os.path.join(results_dir, output_filename)
    logging.info(f"결과는 다음 경로에 저장됩니다: {output_path}")

    # --- 실험 실행 루프 ---
    results = []
    from tqdm import tqdm
    for i, item in enumerate(tqdm(data, desc=f"Processing {args.dataset} with {args.method}")):
        result_item = run_single_item(item=item, method=args.method, model_name=args.model,
                                      config=config, t_max=args.t_max)
        results.append(result_item)

        # 중간 저장 로직
        if (i + 1) % 50 == 0:
             logging.info(f"--- 중간 결과 저장 중 ({i+1}/{len(data)}) ---")
             try:
                 save_jsonl(results, output_path)
             except Exception as e:
                 logging.error(f"중간 결과 저장 실패: {e}")

    # --- 최종 결과 저장 ---
    logging.info(f"\n--- 실험 완료. 최종 결과 저장 중 ---")
    try:
        save_jsonl(results, output_path)
        logging.info(f"총 {len(results)}개의 결과가 {output_path}에 저장되었습니다.")
    except Exception as e:
        logging.error(f"최종 결과 저장 실패: {e}")

if __name__ == "__main__":
    main()