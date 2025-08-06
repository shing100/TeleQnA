from copy import deepcopy
import json
import requests
import ast
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

    
    predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')
    predicted_answers_str = predicted_answers_str[predicted_answers_str.find("{"):]
    
    parsed_predicted_answers = ast.literal_eval(predicted_answers_str)
    
    for q in parsed_predicted_answers:
        if "answer" in parsed_predicted_answers[q] and "question" in parsed_predicted_answers[q]:
            parsed_predicted_answers[q] = {
                "question": parsed_predicted_answers[q]["question"],
                "answer": parsed_predicted_answers[q]["answer"]
            }
    
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
            print(f"Batch {batch_id} attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(1)  # 재시도 전 잠시 대기
            
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
