from copy import deepcopy
import json
import requests
import ast
import re
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# vLLM API 설정 - 환경 변수로 오버라이드 가능
import os
API_BASE_URL = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")  # vLLM 서버 주소
API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM에서는 보통 빈 문자열 또는 "EMPTY" 사용

print(f"Using vLLM API endpoint: {API_BASE_URL}")

def get_available_models():
    """vLLM 서버에서 사용 가능한 모델 목록을 가져옵니다."""
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(f"{API_BASE_URL}/models", headers=headers, timeout=30)
        
        if response.status_code == 200:
            models_data = response.json()
            if "data" in models_data:
                models = [model["id"] for model in models_data["data"]]
                return models
            else:
                print("Warning: Unexpected response format from models API")
                return []
        else:
            print(f"Warning: Failed to get models list (status {response.status_code})")
            return []
            
    except Exception as e:
        print(f"Warning: Error getting models list: {e}")
        return []

def select_model(available_models):
    """사용 가능한 모델 중에서 적절한 모델을 선택합니다."""
    if not available_models:
        return None
    
    print(f"Available models ({len(available_models)}):")
    for i, model in enumerate(available_models):
        print(f"  {i+1}. {model}")
    
    # 단일 모델인 경우 자동 선택
    if len(available_models) == 1:
        selected_model = available_models[0]
        print(f"Auto-selected: {selected_model}")
        return selected_model
    
    # 여러 모델이 있는 경우 사용자 입력 받기
    try:
        while True:
            choice = input(f"Select model (1-{len(available_models)}) or press Enter for first model: ").strip()
            
            if choice == "":
                selected_model = available_models[0]
                break
            
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_models):
                    selected_model = available_models[choice_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_models)}")
            except ValueError:
                print("Please enter a valid number")
                
    except KeyboardInterrupt:
        print("\nUsing first model as default")
        selected_model = available_models[0]
    
    print(f"Selected model: {selected_model}")
    return selected_model
    
syst_prompt = """
Please provide the answers to the following telecommunications related multiple choice questions. The questions will be in a JSON format, the answers must also be in a JSON format as follows:
 {
"question 1": {
"question": question,
"answer": "option {answer id}: {answer string}"
},
...
}
"""

def check_questions_with_val_output(questions_dict, model):
    questions_only = deepcopy(questions_dict)
    answers_only = {}
    for q in questions_dict:
        answers_only[q] = {
            "question": questions_dict[q]["question"],
            "answer": questions_dict[q]["answer"]
        }
    
        questions_only[q].pop("answer")
        
        if 'explanation' in questions_only[q]:
            questions_only[q].pop('explanation')

        if 'category' in questions_only:
            questions_only[q].pop('category')
    
    user_prompt = "Here are the questions: \n "
    user_prompt += json.dumps(questions_only)
    
    # vLLM API 호출
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": syst_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 4096
    }
    
    response = requests.post(f"{API_BASE_URL}/chat/completions", 
                           headers=headers, 
                           json=payload,
                           timeout=300)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    generated_output = response.json()
    predicted_answers_str = generated_output["choices"][0]["message"]["content"]

    
    # 코드 블록 처리 개선
    def extract_json_from_codeblock(text):
        """다양한 형태의 코드 블록에서 JSON 추출"""
        # 여러 마커 패턴 시도
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```JSON\s*(.*?)\s*```', 
            r'```\s*\{(.*?)\}\s*```',
            r'```\s*(.*?)\s*```'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                json_content = matches[0].strip()
                if json_content.startswith('{'):
                    return json_content
                # 중괄호가 없으면 추가
                elif '{' in json_content and '}' in json_content:
                    start_brace = json_content.find('{')
                    return json_content[start_brace:]
        
        return text
    
    if "```" in predicted_answers_str:
        predicted_answers_str = extract_json_from_codeblock(predicted_answers_str)
    
    # JSON 정리 및 파싱 시도
    predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')
    predicted_answers_str = predicted_answers_str[predicted_answers_str.find("{"):]
    
    # 강화된 JSON 파싱 함수
    def robust_json_parse(json_str):
        """다양한 방법으로 JSON 파싱 시도"""
        parsing_attempts = []
        
        # 방법 1: 기본 json.loads
        try:
            result = json.loads(json_str)
            parsing_attempts.append(("json.loads", "success"))
            return result, parsing_attempts
        except json.JSONDecodeError as e:
            parsing_attempts.append(("json.loads", f"failed: {str(e)[:100]}"))
        
        # 방법 2: ast.literal_eval
        try:
            result = ast.literal_eval(json_str)
            parsing_attempts.append(("ast.literal_eval", "success"))
            return result, parsing_attempts
        except (ValueError, SyntaxError) as e:
            parsing_attempts.append(("ast.literal_eval", f"failed: {str(e)[:100]}"))
        
        # 방법 3: JSON 정리 후 재시도
        try:
            # 여러 정리 작업 수행
            cleaned_str = json_str.strip()
            # 마지막 쉼표 제거
            cleaned_str = re.sub(r',\s*}', '}', cleaned_str)
            cleaned_str = re.sub(r',\s*]', ']', cleaned_str)
            # 누락된 따옴표 수정 시도
            cleaned_str = re.sub(r'(\w+):', r'"\1":', cleaned_str)
            # 잘못된 따옴표 수정
            cleaned_str = cleaned_str.replace("'", '"')
            
            result = json.loads(cleaned_str)
            parsing_attempts.append(("cleaned_json", "success"))
            return result, parsing_attempts
        except json.JSONDecodeError as e:
            parsing_attempts.append(("cleaned_json", f"failed: {str(e)[:100]}"))
        
        # 방법 4: 정규식을 이용한 강제 파싱
        try:
            # question 패턴 추출
            question_pattern = r'"question\s*(\d+)"\s*:\s*\{[^}]*"question"\s*:\s*"([^"]+)"\s*,\s*"answer"\s*:\s*"([^"]+)"\s*\}'
            matches = re.findall(question_pattern, json_str, re.IGNORECASE | re.DOTALL)
            
            if matches:
                result = {}
                for q_num, question, answer in matches:
                    key = f"question {q_num}"
                    result[key] = {
                        "question": question.strip(),
                        "answer": answer.strip()
                    }
                parsing_attempts.append(("regex_parsing", f"success: extracted {len(matches)} questions"))
                return result, parsing_attempts
            else:
                parsing_attempts.append(("regex_parsing", "failed: no matches found"))
        except Exception as e:
            parsing_attempts.append(("regex_parsing", f"failed: {str(e)[:100]}"))
        
        return None, parsing_attempts
    
    # 파싱 시도
    parsed_predicted_answers, parsing_log = robust_json_parse(predicted_answers_str)
    
    # 파싱 실패 시 상세한 오류 정보 제공
    if parsed_predicted_answers is None:
        error_details = "\n".join([f"  {method}: {result}" for method, result in parsing_log])
        raise Exception(f"Failed to parse JSON response after multiple attempts:\n{error_details}\n\nOriginal response: {predicted_answers_str[:500]}...")
    
    # 응답 형식 정규화 및 검증
    def normalize_answer_format(answers_dict):
        """응답 형식을 정규화하고 검증"""
        normalized = {}
        for q_key, q_data in answers_dict.items():
            if isinstance(q_data, dict):
                # 필수 필드 확인
                if "question" in q_data and "answer" in q_data:
                    normalized[q_key] = {
                        "question": str(q_data["question"]).strip(),
                        "answer": str(q_data["answer"]).strip()
                    }
                # question 필드가 없는 경우 추론 시도
                elif "answer" in q_data:
                    normalized[q_key] = {
                        "question": "Unknown question",
                        "answer": str(q_data["answer"]).strip()
                    }
            # 단순 문자열인 경우 answer로 처리
            elif isinstance(q_data, str):
                normalized[q_key] = {
                    "question": "Unknown question", 
                    "answer": q_data.strip()
                }
        return normalized
    
    parsed_predicted_answers = normalize_answer_format(parsed_predicted_answers)
    
    accepted_questions = {}
    
    for q in questions_dict:
        if q in parsed_predicted_answers and q in answers_only:
            if parsed_predicted_answers[q] == answers_only[q]:
                accepted_questions[q] = questions_dict[q]

    return accepted_questions, parsed_predicted_answers

def process_single_question_batch(question_batch_data):
    """단일 배치를 처리하는 함수 (멀티프로세스용)"""
    batch_id, questions_dict, model, max_attempts = question_batch_data
    
    for attempt in range(max_attempts):
        try:
            accepted_questions, parsed_predicted_answers = check_questions_with_val_output(questions_dict, model)
            
            # 결과 정리
            results = {}
            for q in questions_dict:
                results[q] = deepcopy(questions_dict[q])
                results[q]['tested answer'] = parsed_predicted_answers[q]['answer'] if q in parsed_predicted_answers else "Error: No answer"
                results[q]['correct'] = q in accepted_questions
                
            return batch_id, results, True  # 성공
            
        except Exception as e:
            error_msg = str(e)
            print(f"Batch {batch_id} attempt {attempt + 1} failed: {error_msg}")
            
            # 파싱 오류인 경우 더 자세한 정보 출력
            if "Failed to parse JSON response" in error_msg:
                print(f"  Parsing error details for batch {batch_id}")
                
            if attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 10)  # 지수 백오프 (최대 10초)
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            
    return batch_id, {}, False  # 실패

def check_questions_parallel(all_questions, model, n_questions=5, max_attempts=5, n_processes=None):
    """멀티프로세스로 질문들을 병렬 처리"""
    if n_processes is None:
        n_processes = min(cpu_count(), 4)  # CPU 코어 수와 4 중 작은 값 사용
    
    print(f"Using {n_processes} processes for parallel evaluation")
    
    # 배치 생성
    batches = []
    shuffled_idx = list(range(len(all_questions)))
    
    for start_id in range(0, len(all_questions), n_questions):
        end_id = min(start_id + n_questions, len(all_questions))
        
        batch_questions = {}
        for k in range(start_id, end_id):
            q_name = f"question {shuffled_idx[k]}"
            if q_name in all_questions:
                batch_questions[q_name] = all_questions[q_name]
        
        if batch_questions:  # 빈 배치가 아닌 경우만 추가
            batch_id = start_id // n_questions
            batches.append((batch_id, batch_questions, model, max_attempts))
    
    # 멀티프로세스 실행
    all_results = {}
    successful_batches = 0
    
    with Pool(processes=n_processes) as pool:
        batch_results = pool.map(process_single_question_batch, batches)
        
        for batch_id, results, success in batch_results:
            if success:
                all_results.update(results)
                successful_batches += 1
            else:
                print(f"Batch {batch_id} failed after all attempts")
    
    print(f"Completed {successful_batches}/{len(batches)} batches successfully")
    return all_results
