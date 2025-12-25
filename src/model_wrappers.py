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
# --- Hugging Face 토큰 로드 (필요시) ---
def _get_huggingface_token(config: Dict[str, Any]) -> Optional[str]:
    """환경 변수 HF_TOKEN 또는 config 파일에서 Hugging Face 토큰을 가져옵니다."""
    key_name = 'huggingface_token'
    env_var = 'HF_TOKEN' # Hugging Face가 사용하는 표준 환경 변수 이름
    token = os.environ.get(env_var)
    if token:
        print(f"Hugging Face 토큰을 환경 변수 {env_var}에서 로드했습니다.")
        return token
    print(f"경고: Hugging Face 토큰을 환경 변수 {env_var} 또는 config에서 찾을 수 없습니다. (접근 제한 모델 로딩 불가)")
    return None


def evaluate_generate(prompt: str, model_name: str, config: Dict[str, Any], 
                      generation_params_override: Optional[Dict[str, Any]] = None) -> str:
    
    # 1. 모델 설정
    model_config = next((m for m in config.get('models', []) if m.get('name') == model_name), None)
    provider = model_config.get('provider') if model_config else ("google" if "gemini" in model_name.lower() else "openai")

    # 2. 파라미터
    gen_params = {"temperature": 0.0}
    if model_config and 'generation_params' in model_config:
        gen_params.update(model_config['generation_params'])
    if generation_params_override:
        gen_params.update(generation_params_override)

    # 3. 재시도 로직
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
                    raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")

                # [수정됨] configure() 대신 Client 객체 생성
                client = genai.Client(api_key=api_key)
                
                # 안전 설정 (모두 허용)
                safety_settings = [
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                ]

                # Thinking Config 설정
                generate_config = types.GenerateContentConfig(
                    temperature=gen_params.get("temperature", 0.0),
                    max_output_tokens=8192,  # 넉넉한 토큰
                    response_mime_type="application/json",
                    
                    # Thinking 설정 적용
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=False,
                        thinking_level="LOW"
                    ),
                    safety_settings=safety_settings
                )

                # API 호출
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=generate_config
                )
                
                # 결과 처리
                if response.text:
                    return response.text.strip()
                else:
                    logger.warning(f"[{model_name}] 빈 응답 발생.")
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
                logger.warning(f"Rate Limit(429) 감지. {wait_time}초 대기 후 재시도... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"API Error: {e}")
                return json.dumps({"score": 0, "reasoning": f"API Error: {str(e)}", "is_misconception": False})

    return json.dumps({"score": 0, "reasoning": "Timeout/RateLimit Failed", "is_misconception": False})


# --- 메인 생성 함수 ---
def generate(prompt: str, model_name: str, config: Dict[str, Any],
             generation_params_override: Optional[Dict[str, Any]] = None) -> str:
    
    print(f"\n--- 모델 호출 시작: {model_name} ---")
    print(f"프롬프트 (시작):\n{prompt[:200]}...\n") 

    # 모델 설정 찾기
    model_config = next((m for m in config.get('models', []) if m.get('name') == model_name), None)
    if not model_config:
        print(f" 오류: 모델 '{model_name}'을(를) config.yaml에서 찾을 수 없습니다")
        return f"오류: 모델 '{model_name}' 설정을 찾을 수 없습니다."

    provider = model_config.get('provider')
    response = f"오류: Provider '{provider}'이(가) 구현되지 않았거나 실패했습니다." 

    try:
        # --- 더미 Provider ---
        if provider == "dummy":
            time.sleep(0.1) 
            if "[검증 질문]" in prompt:
                response = "더미: 이 사실은 출처에 따라 올바른가요?"
            elif "[사실적 답변]" in prompt:
                response = "더미: 네, 출처에서 이를 확인합니다."
            elif "[판단]" in prompt:
                response = random.choice(["[예]", "[아니오]"])
            elif "[수정된 팩트]" in prompt:
                response = "더미: 이것은 수정된 사실입니다."
            elif "[재작성된 문장]" in prompt:
                response = "더미: 이것은 수정을 반영하여 재작성된 문장입니다."
            else:
                response = f"더미 응답: {prompt[:50]}..."
            
            # [추가] 더미 토큰 계산 (근사치: 4글자 = 1토큰)
            in_tokens = len(prompt) // 4
            out_tokens = len(response) // 4
            token_tracker.input_tokens += in_tokens
            token_tracker.output_tokens += out_tokens
            token_tracker.total_tokens += (in_tokens + out_tokens)
            
            print(f"더미 응답: {response}")

        # --- 로컬 Hugging Face Provider ---
        elif provider == "local_hf":
            model_id = model_config['name']
            cache_key = model_id 

            # 캐시에 모델과 토크나이저가 없으면 로드
            if cache_key not in _loaded_models:
                print(f"로컬 모델 로딩 중: {model_id}...")
                loading_params = model_config.get('loading_params', {})
                quantization_config = None
                quant_type = loading_params.get('quantization')

                if quant_type == "bitsandbytes_4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 
                    )
                    print("4비트 양자화(BitsAndBytes) 사용 중.")
                elif quant_type == "bitsandbytes_8bit":
                     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                     print("8비트 양자화(BitsAndBytes) 사용 중.")

                auth_token = None
                if model_config.get('requires_auth_token'):
                    auth_token = _get_huggingface_token(config)
                    if not auth_token:
                        raise ValueError(f"모델 {model_id}은(는) 인증 토큰이 필요하지만 제공되지 않았습니다.")
                    else:
                         print("Hugging Face 인증 토큰 사용 중.")

                dtype_str = loading_params.get('torch_dtype', 'auto')
                try:
                    torch_dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else 'auto'
                    print(f"Torch dtype 설정: {torch_dtype}")
                except TypeError:
                    print(f"경고: 잘못된 torch_dtype '{dtype_str}'. 'auto'로 설정합니다.")
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
                print(f"모델 {model_id} 로드 완료. Device: {model.device}")
            else:
                pass

            model_data = _loaded_models[cache_key]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]

            # --- 생성 파라미터 준비 ---
            gen_params = config.get('default_generation_params', {}).copy()
            model_gen_params = model_config.get('generation_params', {})
            if model_gen_params:
                gen_params.update(model_gen_params)
            if generation_params_override:
                gen_params.update(generation_params_override)

            # --- 생성 ---
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
            
            # [핵심 수정] 정확한 토큰 수 계산 및 기록
            input_tokens_count = inputs['input_ids'].shape[1]
            output_tokens_count = len(response_ids)
            
            # 전역 트래커 업데이트
            token_tracker.input_tokens += input_tokens_count
            token_tracker.output_tokens += output_tokens_count
            token_tracker.total_tokens += (input_tokens_count + output_tokens_count)

            print(f"로컬 HF 응답: {response}")
            print(f"📊 토큰 사용량: Input({input_tokens_count}) + Output({output_tokens_count}) = {input_tokens_count + output_tokens_count}")

        else:
            print(f"*** 오류: Provider '{provider}'은(는) 지원되지 않습니다. ***")
            response = f"오류: 지원되지 않는 provider '{provider}'."

    except Exception as e:
        print(f"*** {model_name} ({provider}) 모델 호출 중 오류 발생: {e} ***")
        import traceback
        traceback.print_exc()
        response = f"오류: 생성 중 예외 발생 - {type(e).__name__}"

    return response.strip()

# --- 직접 실행 테스트용 예시 ---
if __name__ == '__main__':
    # 테스트용 설정 로드 (경로 주의)
    try:
        from utils import load_config # utils.py가 같은 폴더 또는 PYTHONPATH에 있어야 함
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config.yaml')
        test_config = load_config(config_path)
    except (ImportError, FileNotFoundError) as e:
        print(f"테스트용 config 로드 실패 ({e}). 테스트를 건너<0xEB><0x9B><0x8D>니다.")
        test_config = None # 테스트 실행 방지

    if test_config:
        # 더미 모델 테스트
        print("\n--- 더미 모델 테스트 ---")
        dummy_response = generate("프랑스의 수도는 무엇인가요?", "dummy-model", test_config)
        print("더미 테스트 출력:", dummy_response)

        # 로컬 HF 모델 테스트 (config에 있는 모델 이름으로 변경)
        # 테스트 전 해당 모델 다운로드 및 환경 설정(CUDA 등) 필요
        print("\n--- 로컬 HF 모델 테스트 ---")
        try:
            # config.yaml에 정의된 로컬 모델 이름 사용
            local_model_name = "meta-llama/Llama-3.1-8B-Instruct" # 예시, 실제 config에 있는 이름 사용
            if any(m['name'] == local_model_name for m in test_config.get('models', [])):
                 # 간단한 프롬프트로 테스트, temperature 약간 낮춰서 일관성 보기
                 hf_response = generate("대한민국의 수도는 어디인가요?",
                                        local_model_name,
                                        test_config,
                                        generation_params_override={"temperature": 0.1, "max_new_tokens": 50})
                 print(f"\n{local_model_name} 테스트 출력:", hf_response)
            else:
                 print(f"경고: 모델 '{local_model_name}'이(가) config 파일에 정의되지 않아 테스트를 건너<0xEB><0x9B><0x8D>니다.")

        except Exception as e:
            print(f"\n로컬 HF 테스트 실행 중 오류 발생: {e}")
            print("  (모델 다운로드, CUDA 설정, 메모리 부족 등을 확인하세요)")