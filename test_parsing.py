#!/usr/bin/env python3
"""
JSON 파싱 테스트 스크립트
다양한 LLM 응답 형식에 대한 파싱 테스트
"""

import json
import ast
import re

def extract_and_parse_json(response_text):
    """
    LLM 응답에서 JSON을 추출하고 파싱하는 함수
    """
    predicted_answers_str = response_text
    
    # ```json 코드 블록이 있는 경우 처리
    if "```json" in predicted_answers_str:
        # ```json과 ``` 사이의 내용 추출
        start_marker = "```json"
        end_marker = "```"
        start_idx = predicted_answers_str.find(start_marker) + len(start_marker)
        end_idx = predicted_answers_str.find(end_marker, start_idx)
        if end_idx != -1:
            predicted_answers_str = predicted_answers_str[start_idx:end_idx].strip()
    elif "```" in predicted_answers_str:
        # 일반 코드 블록 처리 (```로 시작하는 경우)
        lines = predicted_answers_str.split('\n')
        in_code_block = False
        json_lines = []
        for line in lines:
            if line.strip().startswith('```') and not in_code_block:
                in_code_block = True
                continue
            elif line.strip() == '```' and in_code_block:
                break
            elif in_code_block:
                json_lines.append(line)
        if json_lines:
            predicted_answers_str = '\n'.join(json_lines)
    
    # JSON 정리 및 파싱 시도
    predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')
    predicted_answers_str = predicted_answers_str[predicted_answers_str.find("{"):]
    
    # 여러 방법으로 JSON 파싱 시도
    parsed_predicted_answers = None
    
    # 방법 1: ast.literal_eval 시도
    try:
        parsed_predicted_answers = ast.literal_eval(predicted_answers_str)
        print("✅ ast.literal_eval로 파싱 성공")
    except (ValueError, SyntaxError) as e:
        print(f"❌ ast.literal_eval 실패: {e}")
    
    # 방법 2: json.loads 시도
    if parsed_predicted_answers is None:
        try:
            parsed_predicted_answers = json.loads(predicted_answers_str)
            print("✅ json.loads로 파싱 성공")
        except json.JSONDecodeError as e:
            print(f"❌ json.loads 실패: {e}")
    
    # 방법 3: 더 관대한 JSON 정리 후 재시도
    if parsed_predicted_answers is None:
        try:
            # 불완전한 JSON 정리
            cleaned_str = re.sub(r',\s*}', '}', predicted_answers_str)  # 마지막 쉼표 제거
            cleaned_str = re.sub(r',\s*]', ']', cleaned_str)  # 배열 마지막 쉼표 제거
            parsed_predicted_answers = json.loads(cleaned_str)
            print("✅ 정리 후 json.loads로 파싱 성공")
        except json.JSONDecodeError as e:
            print(f"❌ 정리 후 json.loads 실패: {e}")
    
    return parsed_predicted_answers

# 테스트 케이스들
test_cases = [
    # 케이스 1: 일반 JSON
    {
        "name": "일반 JSON",
        "response": '''
{
"question 0": {
"question": "What is MIMO?",
"answer": "option 1: Multiple Input Multiple Output"
},
"question 1": {
"question": "What is 5G?",
"answer": "option 2: Fifth Generation"
}
}
        '''
    },
    
    # 케이스 2: ```json 코드 블록
    {
        "name": "```json 코드 블록",
        "response": '''
Here are the answers:

```json
{
"question 0": {
"question": "What is MIMO?",
"answer": "option 1: Multiple Input Multiple Output"
},
"question 1": {
"question": "What is 5G?",
"answer": "option 2: Fifth Generation"
}
}
```

These answers are based on my telecommunications knowledge.
        '''
    },
    
    # 케이스 3: ``` 코드 블록
    {
        "name": "``` 코드 블록",
        "response": '''
```
{
"question 0": {
"question": "What is MIMO?",
"answer": "option 1: Multiple Input Multiple Output"
},
"question 1": {
"question": "What is 5G?",
"answer": "option 2: Fifth Generation"
}
}
```
        '''
    },
    
    # 케이스 4: 마지막 쉼표가 있는 JSON
    {
        "name": "마지막 쉼표가 있는 JSON",
        "response": '''
{
"question 0": {
"question": "What is MIMO?",
"answer": "option 1: Multiple Input Multiple Output",
},
"question 1": {
"question": "What is 5G?",
"answer": "option 2: Fifth Generation",
}
}
        '''
    },
    
    # 케이스 5: 설명과 함께
    {
        "name": "설명과 함께",
        "response": '''
Based on my analysis, here are the answers to the telecommunications questions:

```json
{
"question 0": {
"question": "What is MIMO?",
"answer": "option 1: Multiple Input Multiple Output"
},
"question 1": {
"question": "What is 5G?",
"answer": "option 2: Fifth Generation"
}
}
```

I selected these answers based on standard telecommunications definitions.
        '''
    }
]

if __name__ == "__main__":
    print("🧪 JSON 파싱 테스트 시작\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📋 테스트 {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            result = extract_and_parse_json(test_case['response'])
            if result:
                print(f"🎯 파싱 결과: {len(result)}개 질문")
                for q_key in result:
                    if 'answer' in result[q_key]:
                        print(f"   {q_key}: {result[q_key]['answer']}")
            else:
                print("❌ 파싱 실패")
        except Exception as e:
            print(f"💥 예외 발생: {e}")
        
        print("\n")
    
    print("✅ 모든 테스트 완료")