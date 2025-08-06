from evaluation_tools import *
import os 
import json
import numpy as np
import pandas as pd
import sys

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

print("Evaluating {}".format(model))

with open(questions_path, encoding="utf-8") as f:
    loaded_json = f.read()
all_questions = json.loads(loaded_json)

end = len(all_questions)
shuffled_idx = np.arange(len(all_questions))

if os.path.exists(save_path):
    with open(save_path) as f:
        loaded_json = f.read()
    results = json.loads(loaded_json)
    
    start = len(results)
    categories = [ques['category'] for ques in results.values()]
    correct = [ques['correct'] for ques in results.values()]
else:
    results = {}
    start = 0
    categories = []
    correct = []
    

print("Start at question: {}".format(start))

k = 0

for start_id in range(start, end, n_questions):
    attempts = 0
    end_id = np.minimum(start_id + n_questions, len(all_questions)-1)
            
    q_names = ["question {}".format(shuffled_idx[k]) for k in range(start_id, end_id)]
    selected_questions = {}
    
    for q_name in q_names:
        selected_questions[q_name] = all_questions[q_name]

    while attempts < max_attempts:
        try:
            accepted_questions, parsed_predicted_answers = check_questions_with_val_output(selected_questions, model)
            
            for q in selected_questions:  
                parsed_predicted_answers[q]['answer']
                results[q] = deepcopy(selected_questions[q])
                results[q]['tested answer'] = parsed_predicted_answers[q]['answer']
                results[q]['correct'] = q in accepted_questions
                correct += [results[q]['correct']]
                categories += [selected_questions[q]['category']]
        
            break
            
        except Exception as e:
            attempts += 1
            print(f"Attempt {attempts} failed. Error: {e}")
            print("Retrying...")
        
    else:
        print(f"Failed after {max_attempts} attempts.")

    k += 1
    if k % 5 == 0:
        with open(save_path, 'w') as f:
            res_str = json.dumps(results)
            f.write(res_str)

        res = pd.DataFrame.from_dict({
            'categories': categories,
            'correct': correct
        })

        summary = res.groupby('categories').mean()
        summary['counts'] = res.groupby('categories').count()['correct'].values
        
        print("Total number of questions answered: {}".format(len(categories)))
        print(summary)

with open(save_path, 'w') as f:
    res_str = json.dumps(results)
    f.write(res_str)

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