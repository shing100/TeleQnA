from evaluation_tools import *
import os 
import json
import numpy as np
import pandas as pd
import sys
import time

# 명령행 인자 또는 환경 변수로 모델 지정 가능
specified_model = None
if len(sys.argv) > 1:
    specified_model = sys.argv[1]
    print(f"Model specified via command line: {specified_model}")
elif "VLLM_MODEL" in os.environ:
    specified_model = os.environ["VLLM_MODEL"]
    print(f"Model specified via environment variable: {specified_model}")

if specified_model:
    model = specified_model
    print(f"Using specified model: {model}")
else:
    # 자동으로 사용 가능한 모델 가져오기
    print("Getting available models from vLLM server...")
    available_models = get_available_models()

    if available_models:
        model = select_model(available_models)
        if model is None:
            print("Error: No suitable model found")
            exit(1)
    else:
        print("Error: Could not retrieve models from vLLM server")
        print("Please check if vLLM server is running at the configured endpoint")
        print("Usage: python run.py [model_name] or set VLLM_MODEL environment variable")
        exit(1)
questions_path = "TeleQnA.txt"
save_path = os.path.join(model+"_answers.txt")

n_questions = 5 # Batch the questions asked to reduce time
max_attempts = 5 # Maximal number of trials before skipping the question
n_processes = int(os.getenv("VLLM_PROCESSES", "4"))  # 환경 변수로 프로세스 수 조정 가능

print("Evaluating {} with {} parallel processes".format(model, n_processes))

with open(questions_path, encoding="utf-8") as f:
    loaded_json = f.read()
all_questions = json.loads(loaded_json)

# 기존 결과가 있다면 로드 (resume 기능)
if os.path.exists(save_path):
    with open(save_path) as f:
        loaded_json = f.read()
    existing_results = json.loads(loaded_json)
    
    # 이미 처리된 질문들을 제외
    remaining_questions = {}
    for q_name, q_data in all_questions.items():
        if q_name not in existing_results:
            remaining_questions[q_name] = q_data
    
    print("Resuming from previous run. {} questions remaining.".format(len(remaining_questions)))
    all_questions_to_process = remaining_questions
    existing_count = len(existing_results)
else:
    existing_results = {}
    all_questions_to_process = all_questions
    existing_count = 0
    
if len(all_questions_to_process) == 0:
    print("All questions already processed!")
    results = existing_results
else:
    print("Processing {} questions with multiprocessing...".format(len(all_questions_to_process)))
    start_time = time.time()
    
    # 멀티프로세스로 병렬 처리
    new_results = check_questions_parallel(
        all_questions_to_process, 
        model, 
        n_questions=n_questions, 
        max_attempts=max_attempts,
        n_processes=n_processes
    )
    
    # 기존 결과와 합치기
    results = {**existing_results, **new_results}
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")

# 최종 결과 저장
with open(save_path, 'w') as f:
    res_str = json.dumps(results)
    f.write(res_str)

# 통계 계산
categories = [ques['category'] for ques in results.values()]
correct = [ques['correct'] for ques in results.values()]

res = pd.DataFrame.from_dict({
    'categories': categories,
    'correct': correct
})

summary = res.groupby('categories').mean()
summary['counts'] = res.groupby('categories').count()['correct'].values

print("Total number of questions answered: {}".format(len(categories)))
print(summary)
print()
print()
print("Final result: {}".format(np.mean([q['correct'] for q in results.values()])))