#!/usr/bin/env python3
"""
vLLM 서버 응답 테스트 스크립트
실제 vLLM 서버와의 통신을 테스트
"""

import os
import json
from evaluation_tools import get_available_models, select_model, check_questions_with_val_output

# 테스트용 샘플 질문들
sample_questions = {
    "question 0": {
        "question": "What does MIMO stand for in telecommunications?",
        "option 1": "Multiple Input Multiple Output",
        "option 2": "Modular Input Modular Output",
        "option 3": "Maximum Input Maximum Output",
        "option 4": "Multi-Interface Multi-Operation",
        "answer": "option 1: Multiple Input Multiple Output",
        "explanation": "MIMO stands for Multiple Input Multiple Output, a wireless technology that uses multiple antennas.",
        "category": "Lexicon"
    },
    "question 1": {
        "question": "What is the diversity gain for the detection of each symbol in the Alamouti scheme?",
        "option 1": "0",
        "option 2": "4", 
        "option 3": "2",
        "option 4": "1",
        "answer": "option 3: 2",
        "explanation": "The Alamouti scheme provides a diversity gain of 2 for the detection of each symbol.",
        "category": "Research publications"
    }
}

def test_vllm_connection():
    """vLLM 서버 연결 및 응답 테스트"""
    print("🔍 vLLM 서버 연결 테스트 시작")
    print("=" * 60)
    
    # 1. 모델 목록 가져오기
    print("1️⃣ 사용 가능한 모델 확인...")
    available_models = get_available_models()
    
    if not available_models:
        print("❌ vLLM 서버에서 모델을 가져올 수 없습니다.")
        print("   - vLLM 서버가 실행 중인지 확인하세요")
        print("   - API 엔드포인트가 올바른지 확인하세요")
        return False
    
    print(f"✅ {len(available_models)}개 모델 발견:")
    for model in available_models:
        print(f"   - {model}")
    
    # 2. 모델 선택
    print("\n2️⃣ 모델 선택...")
    model = available_models[0]  # 첫 번째 모델 자동 선택
    print(f"✅ 선택된 모델: {model}")
    
    # 3. 테스트 질문 전송
    print("\n3️⃣ 샘플 질문 전송 및 응답 테스트...")
    try:
        accepted_questions, parsed_predicted_answers = check_questions_with_val_output(
            sample_questions, model
        )
        
        print("✅ API 호출 성공!")
        print(f"📊 처리된 질문: {len(parsed_predicted_answers)}개")
        print(f"🎯 정답 매칭: {len(accepted_questions)}개")
        
        print("\n📝 상세 결과:")
        for q_key, response in parsed_predicted_answers.items():
            correct_answer = sample_questions[q_key]["answer"]
            model_answer = response.get("answer", "No answer")
            is_correct = q_key in accepted_questions
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} {q_key}:")
            print(f"      정답: {correct_answer}")
            print(f"      모델: {model_answer}")
            print(f"      매칭: {'예' if is_correct else '아니오'}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False

def test_json_parsing_edge_cases():
    """JSON 파싱 극단 케이스 테스트"""
    print("\n🧪 JSON 파싱 극단 케이스 테스트")
    print("=" * 60)
    
    # 의도적으로 문제가 있는 응답들을 시뮬레이션
    problematic_responses = [
        # 불완전한 JSON
        '''{
"question 0": {
"question": "What is MIMO?",
"answer": "option 1: Multiple Input Multiple Output"
},
"question 1": {
"question": "What is''',
        
        # 여분의 텍스트가 있는 경우
        '''Here is my answer:
```json
{
"question 0": {"question": "What is MIMO?", "answer": "option 1: Multiple Input Multiple Output"}
}
```
Let me know if you need more details!''',
    ]
    
    for i, response in enumerate(problematic_responses, 1):
        print(f"\n🧩 극단 케이스 {i}:")
        print(f"응답 길이: {len(response)} 문자")
        
        # 여기서 실제로 파싱 로직을 테스트할 수 있지만,
        # 실제 vLLM 호출 없이는 전체 플로우를 테스트하기 어려움
        print("   (실제 vLLM 서버가 있을 때 전체 테스트 가능)")

if __name__ == "__main__":
    print("🚀 TeleQnA vLLM 통합 테스트")
    print("=" * 60)
    
    # 환경 변수 확인
    api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
    print(f"📡 API 엔드포인트: {api_base}")
    
    # vLLM 서버 연결 테스트
    if test_vllm_connection():
        print("\n🎉 vLLM 서버 테스트 성공!")
    else:
        print("\n⚠️  vLLM 서버 테스트 실패")
        print("   vLLM 서버를 시작하고 다시 시도하세요:")
        print("   python -m vllm.entrypoints.openai.api_server --model YOUR_MODEL")
    
    # JSON 파싱 극단 케이스 테스트
    test_json_parsing_edge_cases()
    
    print("\n✅ 모든 테스트 완료")