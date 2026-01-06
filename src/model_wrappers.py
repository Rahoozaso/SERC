import os
from dotenv import load_dotenv
import json
import time
import random
from typing import Dict, Any, Optional
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils import token_tracker
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
load_dotenv()
_loaded_models = {}
_loaded_eval_models = {}

# --- Load Hugging Face Token (if needed) ---
def _get_huggingface_token(config: Dict[str, Any]) -> Optional[str]:
    """Retrieves the Hugging Face token from the environment variable HF_TOKEN or the config file."""
    key_name = 'huggingface_token'
    env_var = 'HF_TOKEN' # Standard environment variable name used by Hugging Face
    token = os.environ.get(env_var)
    if token:
        print(f"Loaded Hugging Face token from environment variable {env_var}.")
        return token
    print(f"Warning: Could not find Hugging Face token in environment variable {env_var} or config. (Cannot load gated models)")
    return None


def evaluate_generate(prompt: str, model_name: str, config: Dict[str, Any], 
                      generation_params_override: Optional[Dict[str, Any]] = None) -> str:
    
    # 1. Model Setup
    model_config = next((m for m in config.get('models', []) if m.get('name') == model_name), None)
    provider = model_config.get('provider') if model_config else ("google" if "gemini" in model_name.lower() else "openai")

    # 2. Parameters
    gen_params = {"temperature": 0.0}
    if model_config and 'generation_params' in model_config:
        gen_params.update(model_config['generation_params'])
    if generation_params_override:
        gen_params.update(generation_params_override)

    # 3. Retry Logic
    max_retries = 5
    base_wait = 10

    for attempt in range(max_retries + 1):
        try:
            # ---------------------------------------------------------
            # Provider 1: Google Gemini (New SDK: google-genai)
            # ---------------------------------------------------------
            if provider == "google" or "gemini" in model_name.lower():
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY is not set.")
                client = genai.Client(api_key=api_key)
                
                safety_settings = [
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                ]
                generate_config = types.GenerateContentConfig(
                    temperature=gen_params.get("temperature", 0.0),
                    max_output_tokens=8192, 
                    response_mime_type="application/json",
                    
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=False,
                        thinking_level="LOW"
                    ),
                    safety_settings=safety_settings
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=generate_config
                )
                
                if response.text:
                    return response.text.strip()
                else:
                    logger.warning(f"[{model_name}] Empty response received.")
                    return json.dumps({"score": 0, "reasoning": "Empty Response", "is_misconception": False})

            # ---------------------------------------------------------
            # Provider 2: OpenAI
            # ---------------------------------------------------------
            elif provider == "openai" or "gpt" in model_name.lower():
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                api_args = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                }
                if gen_params.get("response_mime_type") == "application/json":
                    api_args["response_format"] = {"type": "json_object"}
                
                resp = client.chat.completions.create(**api_args)
                return resp.choices[0].message.content

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = base_wait * (2 ** attempt)
                logger.warning(f"Rate Limit (429) detected. Retrying after {wait_time} seconds... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"API Error: {e}")
                return json.dumps({"score": 0, "reasoning": f"API Error: {str(e)}", "is_misconception": False})

    return json.dumps({"score": 0, "reasoning": "Timeout/RateLimit Failed", "is_misconception": False})


def generate(prompt: str, model_name: str, config: Dict[str, Any],
             generation_params_override: Optional[Dict[str, Any]] = None) -> str:
    
    print(f"\n--- Model Call Start: {model_name} ---")
    print(f"Prompt (Start):\n{prompt[:200]}...\n") 

    model_config = next((m for m in config.get('models', []) if m.get('name') == model_name), None)
    if not model_config:
        print(f" Error: Model '{model_name}' not found in config.yaml")
        return f"Error: Could not find settings for model '{model_name}'."

    provider = model_config.get('provider')
    response = f"Error: Provider '{provider}' is not implemented or failed." 

    try:
        if provider == "dummy":
            time.sleep(0.1) 
            if "[Verification Question]" in prompt or "[검증 질문]" in prompt:
                response = "Dummy: Is this fact correct according to the source?"
            elif "[Factual Answer]" in prompt or "[사실적 답변]" in prompt:
                response = "Dummy: Yes, verifying this from sources."
            elif "[Judgment]" in prompt or "[판단]" in prompt:
                response = random.choice(["[YES]", "[NO]"])
            elif "[Corrected Fact]" in prompt or "[수정된 팩트]" in prompt:
                response = "Dummy: This is a corrected fact."
            elif "[Rewritten Sentence]" in prompt or "[재작성된 문장]" in prompt:
                response = "Dummy: This is the rewritten sentence reflecting corrections."
            else:
                response = f"Dummy response: {prompt[:50]}..."
            
            # [Added] Dummy token calculation (Approx: 4 chars = 1 token)
            in_tokens = len(prompt) // 4
            out_tokens = len(response) // 4
            token_tracker.input_tokens += in_tokens
            token_tracker.output_tokens += out_tokens
            token_tracker.total_tokens += (in_tokens + out_tokens)
            
            print(f"Dummy Response: {response}")

        # --- Local Hugging Face Provider ---
        elif provider == "local_hf":
            model_id = model_config['name']
            cache_key = model_id 

            # Load model and tokenizer if not in cache
            if cache_key not in _loaded_models:
                print(f"Loading local model: {model_id}...")
                loading_params = model_config.get('loading_params', {})
                quantization_config = None
                quant_type = loading_params.get('quantization')

                if quant_type == "bitsandbytes_4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 
                    )
                    print("Using 4-bit quantization (BitsAndBytes).")
                elif quant_type == "bitsandbytes_8bit":
                     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                     print("Using 8-bit quantization (BitsAndBytes).")

                auth_token = None
                if model_config.get('requires_auth_token'):
                    auth_token = _get_huggingface_token(config)
                    if not auth_token:
                        raise ValueError(f"Model {model_id} requires an auth token but none was provided.")
                    else:
                         print("Using Hugging Face auth token.")

                dtype_str = loading_params.get('torch_dtype', 'auto')
                try:
                    torch_dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else 'auto'
                    print(f"Torch dtype setting: {torch_dtype}")
                except TypeError:
                    print(f"Warning: Invalid torch_dtype '{dtype_str}'. Setting to 'auto'.")
                    torch_dtype = 'auto'

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map=loading_params.get('device_map', 'auto'),
                    torch_dtype=torch_dtype,
                    token=auth_token,
                    trust_remote_code=loading_params.get('trust_remote_code', False),
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=auth_token)
                _loaded_models[cache_key] = {"model": model, "tokenizer": tokenizer}
                print(f"Model {model_id} loaded successfully. Device: {model.device}")
            else:
                pass

            model_data = _loaded_models[cache_key]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]

            # --- Prepare Generation Parameters ---
            gen_params = config.get('default_generation_params', {}).copy()
            model_gen_params = model_config.get('generation_params', {})
            if model_gen_params:
                gen_params.update(model_gen_params)
            if generation_params_override:
                gen_params.update(generation_params_override)

            # --- Generate ---
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                temperature=gen_params.get('temperature', 1.0),
                top_p=gen_params.get('top_p', 1.0),
                max_new_tokens=gen_params.get('max_new_tokens', 512),
                repetition_penalty=gen_params.get('repetition_penalty'),
                do_sample=(gen_params.get('temperature', 1.0) > 0.0 and gen_params.get('top_p', 1.0) < 1.0),
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id
            )
            
            response_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # [Key Update] Calculate and record exact token count
            input_tokens_count = inputs['input_ids'].shape[1]
            output_tokens_count = len(response_ids)
            
            # Update global tracker
            token_tracker.input_tokens += input_tokens_count
            token_tracker.output_tokens += output_tokens_count
            token_tracker.total_tokens += (input_tokens_count + output_tokens_count)

            print(f"Local HF Response: {response}")
            print(f"📊 Token Usage: Input({input_tokens_count}) + Output({output_tokens_count}) = {input_tokens_count + output_tokens_count}")

        else:
            print(f"*** Error: Provider '{provider}' is not supported. ***")
            response = f"Error: Unsupported provider '{provider}'."

    except Exception as e:
        print(f"*** Error occurred while calling model {model_name} ({provider}): {e} ***")
        import traceback
        traceback.print_exc()
        response = f"Error: Exception during generation - {type(e).__name__}"

    return response.strip()

# --- Example for direct execution testing ---
if __name__ == '__main__':
    # Load test config (Check path)
    try:
        from utils import load_config # utils.py must be in the same folder or PYTHONPATH
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config.yaml')
        test_config = load_config(config_path)
    except (ImportError, FileNotFoundError) as e:
        print(f"Failed to load test config ({e}). Skipping tests.")
        test_config = None # Prevent test execution

    if test_config:
        # Dummy model test
        print("\n--- Dummy Model Test ---")
        dummy_response = generate("What is the capital of France?", "dummy-model", test_config)
        print("Dummy test output:", dummy_response)

        # Local HF Model Test (Change to model name in config)
        # Requires model download and environment setup (CUDA, etc.) before testing
        print("\n--- Local HF Model Test ---")
        try:
            # Use local model name defined in config.yaml
            local_model_name = "meta-llama/Llama-3.1-8B-Instruct" # Example, use actual name from config
            if any(m['name'] == local_model_name for m in test_config.get('models', [])):
                 # Test with simple prompt, lower temperature slightly for consistency
                 hf_response = generate("Where is the capital of South Korea?",
                                        local_model_name,
                                        test_config,
                                        generation_params_override={"temperature": 0.1, "max_new_tokens": 50})
                 print(f"\n{local_model_name} test output:", hf_response)
            else:
                 print(f"Warning: Model '{local_model_name}' is not defined in config file, skipping test.")

        except Exception as e:
            print(f"\nError occurred during Local HF test execution: {e}")
            print(" (Check model download, CUDA setup, memory shortage, etc.)")