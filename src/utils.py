# serc_framework/utils.py
import yaml
import json
import os
from datetime import datetime
from typing import Dict, Any, List

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """YAML 설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"Config loaded successfully from: {config_path}")
            return config
    except FileNotFoundError:
        print(f"*** Error: Config file not found at {config_path} ***")
        # You might want to raise the error or return an empty dict depending on desired behavior
        raise
    except yaml.YAMLError as e:
        print(f"*** Error parsing YAML file {config_path}: {e} ***")
        raise

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """JSONL (JSON Lines) 파일을 로드하여 딕셔너리 리스트로 반환합니다."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip empty lines
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line in {file_path}: {e}\nLine: {line.strip()}")
        print(f"Loaded {len(data)} records from {file_path}")
    except FileNotFoundError:
        print(f"*** Error: Data file not found at {file_path} ***")
        # Depending on usage, you might return an empty list or raise error
        # raise
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """딕셔너리 리스트를 JSONL 파일로 저장합니다."""
    print(f"--- Debug: Attempting to save to: '{file_path}'") # <--- 디버깅 1: 전달된 경로 확인
    try:
        # 파일 경로에서 디렉토리 이름 추출
        dir_name = os.path.dirname(file_path)
        print(f"--- Debug: Calculated directory: '{dir_name}'") # <--- 디버깅 2: 추출된 디렉토리 확인

        # 디렉토리 이름이 비어있지 않은 경우에만 디렉토리 생성 시도
        if dir_name:
            print(f"--- Debug: Ensuring directory exists: '{dir_name}'") # <--- 디버깅 3
            os.makedirs(dir_name, exist_ok=True)
        else:
            # 디렉토리 이름이 비어 있다면, 현재 작업 디렉토리를 의미함
            print(f"--- Debug: Directory name is empty, attempting to save in current working directory.") # <--- 디버깅 4

        # 파일 열기 및 쓰기
        print(f"--- Debug: Opening file for writing: '{file_path}'") # <--- 디버깅 5
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                # ensure_ascii=False는 한국어 등을 올바르게 저장하기 위함
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(data)} records to {file_path}")

    except FileNotFoundError as e: # 구체적인 에러 타입 명시
        print(f"*** Error saving data to {file_path}: FileNotFoundError - Path likely invalid. {e} ***")
    except OSError as e: # 디렉토리 생성 등 다른 OS 관련 에러
         print(f"*** Error during file/directory operation for {file_path}: OSError - {e} ***")
    except TypeError as e:
        print(f"*** Error serializing data for saving to {file_path}: {e} ***")
        print("   Ensure all items in the list are JSON-serializable dictionaries.")
    except Exception as e: # 예상치 못한 다른 에러
         print(f"*** An unexpected error occurred saving to {file_path}: {e} ***")

def get_timestamp() -> str:
    """현재 시간을 'YYYYMMDD_HHMMSS' 형식의 문자열로 반환합니다."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Example Usage (for testing this file directly) ---
# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    # --- [추가된 부분 시작] ---
    # 현재 스크립트 파일(.py)이 있는 디렉토리 경로를 얻습니다.
    # 이렇게 해야 어디서 실행하든 경로가 올바르게 계산됩니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 프로젝트 루트 디렉토리 경로 (src 폴더의 부모 폴더)
    project_root = os.path.dirname(script_dir)
    # --- [추가된 부분 끝] ---

    # --- Test Config Loading ---
    print("\n--- Testing Config Loading ---")
    # 이제 project_root를 사용하여 config 파일 경로를 안전하게 만듭니다.
    config_file_path = os.path.join(project_root, 'config.yaml')
    try:
        config = load_config(config_file_path) # 수정된 경로 사용
        print("Default T_max:", config.get('default_t_max'))
        print("Models defined:", [m.get('name') for m in config.get('models', [])])
    except Exception as e:
        print(f"Could not load config for testing: {e}")
        print(f"Expected config path: {config_file_path}") # 경로 확인용

    # --- Test JSONL Saving & Loading ---
    print("\n--- Testing JSONL Save/Load ---")
    test_data = [
        {'id': 1, 'text': '첫 번째 줄입니다.', 'valid': True},
        {'id': 2, 'text': '두 번째 줄, 한국어 포함.', 'value': 12.3},
        {'id': 3, 'nested': {'key': 'value'}}
    ]
    # 임시 파일을 현재 스크립트와 같은 디렉토리(src)에 저장하도록 경로 지정
    # 이제 script_dir 변수가 존재하므로 오류가 발생하지 않습니다.
    test_file_path = os.path.join(script_dir, 'temp_test_output.jsonl')

    # Save
    save_jsonl(test_data, test_file_path) # 디버깅 버전 save_jsonl 호출

    # Load
    loaded_data = []
    if os.path.exists(test_file_path):
        loaded_data = load_jsonl(test_file_path)
    else:
        print(f"Save FAILED, cannot load {test_file_path}")

    # Verify
    if loaded_data == test_data:
        print("JSONL Save/Load test successful.")
    else:
        print("JSONL Save/Load test FAILED.")
        print("Original:", test_data)
        print("Loaded:", loaded_data)

    # Clean up the temporary file
    if os.path.exists(test_file_path):
        try:
            os.remove(test_file_path)
            print(f"Removed temporary file: {test_file_path}")
        except OSError as e:
            print(f"Error removing temporary file {test_file_path}: {e}")

    # --- Test Timestamp ---
    print("\n--- Testing Timestamp ---")
    print("Current timestamp:", get_timestamp())