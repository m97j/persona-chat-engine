
# Persona Chat Engine 🎭

## 📌 개요
`Persona Chat Engine`은 NPC/Persona 기반 대화형 AI 시스템으로, 게임 내 캐릭터 또는 챗봇이 플레이어와 맥락 있는 대화를 할 수 있도록 설계되었습니다.

---

## 📂 디렉토리 구조
```

persona-chat-engine/
│
├── ai-server/        # 대화 파이프라인 관리, 게임 서버와 통신
│   ├── app.py
│   ├── schemas.py
│   ├── agent_manager.py
│   ├── dialogue_manager.py
│   ├── preprocess.py
│   ├── postprocess.py
│   ├── generator.py
│   ├── rag.py
│   ├── config.py
│   ├── utils/
│   │   ├── hf_client.py
│   │   └── model_loader.py
│   └── requirements.txt
│
├── hf-serve/         # Hugging Face 모델 추론 API
│   ├── model_utils.py
│   ├── server.py
│   └── requirements.txt
│
├── train/            # (옵션) 모델 학습 관련 자료
│   ├── README.md     # Colab 학습 링크
│   └── dataset/      # (옵션) json 데이터 샘플
│
└── docker-compose.yml


````

---

## ⚙️ 아키텍처
1. **Game Server** → 플레이어 대사 입력
2. **AI Server (Preprocess)** → 조건 검증
3. **HF-Serve** → 모델 추론 (persona, npc_id 반영)
4. **AI Server (Postprocess)** → 윤리 필터링 / delta 값 추출
5. **Game Server** → 대사 전송 + 상태 업데이트

---

## 🚀 배포 방법
### 1. Docker Compose
```bash
docker-compose up --build
````

### 2. 개별 실행

```bash
cd hf-serve && python main.py
cd ai-server && uvicorn main:app --reload
```

---

## 📊 학습 (옵션)

* Colab Notebook: [Train Model on Colab](https://colab.research.google.com/...)
* Hugging Face Model: [HF Model](https://huggingface.co/my-model)

---

## 📽 시연 영상

(업데이트 예정)

---

## 📌 기술 스택

* Python 3.10
* FastAPI
* Hugging Face Transformers
* Docker / Docker Compose
* (옵션) LoRA/RoRA Fine-Tuning



