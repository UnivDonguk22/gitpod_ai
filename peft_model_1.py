# 트랜스포머 라이브러리에서 사용할 클래스, 기능 가져오기
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# peft 라이브러리
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel, PeftConfig

# huggingface_hub 라이브러리
from huggingface_hub import notebook_login

import os
import time


def load_pretrained_model(model_name):
    #사전 학습 모델에 사용할 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #지정한 모델 이름 사용해 사전 학습된 casual LM 로드
    foundation_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True)
    
    return tokenizer, foundation_model

def load_tuning_model(tuning_model_path):
    # PeftModel 클래스를 사용해 학습된 Peft 모델 불러오기
    loaded_model = PeftModel.from_pretrained(
        foundation_model,    # 프롬프트 튜닝에 사용될 기본 모델
        tuning_model_path,     # Peft 모델이 저장된 위치
        is_trainable = False # 불러온 모델은 훈련될 필요 없음
    )
    return loaded_model

def run_tuning_model(input_text, loaded_model, tokenizer):
    # 지정된 토크나이저 사용해 입력 텍스트 토큰화
    input1 = tokenizer(input_text, return_tensors = "pt", padding = True)

    # input_id, attention_mask에 기반해 불러온 Peft 모델을 사용해 텍스트 생성
    loaded_model_outputs = loaded_model.generate(
        input_ids = input1["input_ids"],
        attention_mask = input1["attention_mask"],
        max_new_tokens = 32,
        eos_token_id = tokenizer.eos_token_id
    )

    # 생성된 토큰 id를 사람이 읽을 수 있는 텍스트로 디코딩
    decoded_output = tokenizer.batch_decode(loaded_model_outputs, skip_special_tokens = True)

    # 디코딩된 결과물 출력
    print(decoded_output)

# main 실행 코드
    # 1. 사전학습 모델 불러오기
tokenizer, foundation_model = load_pretrained_model("bigscience/bloomz-560m")

    # 2. 학습 모델 불러오기
loaded_model = load_tuning_model("peft_model/")

    # 3. 학습 모델 텍스트 요청
run_tuning_model("Show me Einstein's quotes :", loaded_model, tokenizer)