import yaml
import json
import os
from datetime import datetime
from typing import Dict, Any, List

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file."""
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
    """Loads a JSONL (JSON Lines) file and returns a list of dictionaries."""
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
    """Saves a list of dictionaries to a JSONL file."""
    print(f"--- Debug: Attempting to save to: '{file_path}'") # <--- Debug 1: Check passed path
    try:
        # Extract directory name from file path
        dir_name = os.path.dirname(file_path)
        print(f"--- Debug: Calculated directory: '{dir_name}'") # <--- Debug 2: Check extracted directory

        # Attempt to create directory only if directory name is not empty
        if dir_name:
            print(f"--- Debug: Ensuring directory exists: '{dir_name}'") # <--- Debug 3
            os.makedirs(dir_name, exist_ok=True)
        else:
            # If directory name is empty, it means current working directory
            print(f"--- Debug: Directory name is empty, attempting to save in current working directory.") # <--- Debug 4

        # Open file and write
        print(f"--- Debug: Opening file for writing: '{file_path}'") # <--- Debug 5
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                # ensure_ascii=False ensures correct saving of non-ASCII characters (e.g., Korean)
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(data)} records to {file_path}")
    except Exception as e: # Catch unexpected errors
         print(f"An unexpected error occurred saving to {file_path}: {e} ")

def get_timestamp() -> str:
    """Returns the current time as a string in 'YYYYMMDD_HHMMSS' format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)


    # --- Test Config Loading ---
    print("\n--- Testing Config Loading ---")
    config_file_path = os.path.join(project_root, 'config.yaml')
    try:
        config = load_config(config_file_path) 
        print("Default T_max:", config.get('default_t_max'))
        print("Models defined:", [m.get('name') for m in config.get('models', [])])
    except Exception as e:
        print(f"Could not load config for testing: {e}")
        print(f"Expected config path: {config_file_path}") 

    # --- Test JSONL Saving & Loading ---
    print("\n--- Testing JSONL Save/Load ---")
    test_data = [
        {'id': 1, 'text': 'This is the first line.', 'valid': True},
        {'id': 2, 'text': 'Second line, including Korean.', 'value': 12.3},
        {'id': 3, 'nested': {'key': 'value'}}
    ]
    test_file_path = os.path.join(script_dir, 'temp_test_output.jsonl')

    # Save
    save_jsonl(test_data, test_file_path) 

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
    
class TokenUsageTracker:
    """
    A singleton class to track token usage globally.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TokenUsageTracker, cls).__new__(cls)
            cls._instance.total_tokens = 0
            cls._instance.input_tokens = 0
            cls._instance.output_tokens = 0
        return cls._instance

    def reset(self):
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_text: str, output_text: str):
        # Calculate approximate token count (1 token ≈ 4 chars) even without Llama-3 tokenizer
        # (Use tokenizer.encode(text) len for accuracy, but approximation is acceptable for comparative experiments)
        in_cnt = len(input_text) / 4
        out_cnt = len(output_text) / 4
        
        self.input_tokens += int(in_cnt)
        self.output_tokens += int(out_cnt)
        self.total_tokens += int(in_cnt + out_cnt)

    def get_usage(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens
        }

# Create global object
token_tracker = TokenUsageTracker()