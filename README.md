
# Persona Chat Engine – AI NPC Dialogue System 🎭

[![GitHub stars](https://img.shields.io/github/stars/m97j/persona-chat-engine)](https://github.com/m97j/persona-chat-engine)


## 📑 목차
- [📌 개요](#-개요)
- [🧭 아키텍처](#-아키텍처)
- [⚙️ AI 서버 (ai-server/)](#%EF%B8%8F-ai-서버-ai_server)
- [🚀 Hugging Face Serve (hf-serve/)](#-hugging-face-spaces-hf-serve)
- [📊 모델 학습 (train/)](#-모델-학습-train)
- [📦 배포 계획](#-배포-계획)
- [🎥 시연 자료](#-시연-자료)
- [🏁 프로젝트 성과](#-프로젝트-성과)

---

## 📌 개요
**Persona Chat Engine**은 게임 내 NPC(Non-Player Character)와의 상호작용을 위한 AI 대화 엔진입니다.  
플레이어의 선택/행동, NPC 상태를 반영해 자연스러운 대사와 퀘스트 진행을 생성하며, **Delta/Flag** 기반의 상태 변화(신뢰·관계)와 이벤트 트리거를 함께 예측합니다.

- **목표:** 게임 환경에서 몰입감 있는 AI NPC 대화와 퀘스트 반응 생성
- **핵심 기술:** Transformer 기반 LLM, QLoRA 파인튜닝, 멀티헤드 학습(Delta/Flag), 커스텀 프롬프트 포맷, RAG 기반 flag 해석

---

## 🧭 아키텍처

* ### 모델 아키텍처
```mermaid
flowchart LR
  subgraph Input
    T[Tokens]
    E[Tokenizer]
  end
  T --> E --> I[Token Embedding]
  P[RoPE on Q,K]:::op
  subgraph Block_xN
    direction LR
    RN1[RMSNorm]
    MHA[Multi-Head Attention]:::op
    ADD1[Residual Add]
    RN2[RMSNorm]
    FFN[SwiGLU Feed-Forward]:::op
    ADD2[Residual Add]
  end
  I --> P --> RN1 --> MHA --> ADD1 --> RN2 --> FFN --> ADD2
  subgraph Output
    H[LM Head]
    S[Softmax]
  end
  Block_xN --> H --> S

  classDef op fill:#eef,stroke:#669,stroke-width:1px;
```

* ### 전체 프로젝트 통신 구조
```mermaid
graph TD
Client[Unity Client] --input text--> GameServer[Node.js Game Server]
GameServer --ask ai--> AIServer[Python AI Server]
AIServer <--> Preprocess
AIServer --prompt--> HFServe[HuggingFace Spaces]
HFServe --> inference
HFServe --result--> AIServer
AIServer <--> Postprocess
AIServer --npc text, deltas, flags--> GameServer
GameServer --npc text, env flags--> Client
```

---

## ⚙️ AI 서버 (ai_server/)

* ### 역할 & 데이터 흐름

  1. **게임 서버 요청 수신(`app.py`)**
  
     * 최소 입력만 와도 동작: `{ text, npc_id, player_id, ... }`
     * 옵션: 게임 서버가 보낸 상태/컨텍스트가 부족하면 `rag/`에서 NPC 메타(예: `docs/npc_config.json`)를 조회해 보강
  2. **전처리/프롬프트 구성(`pipeline/preprocess.py`, `utils/context_parser.py`, `manager/prompt_builder.py`)**
  
     * 태그/컨텍스트/플레이어 발화를 묶어 **모델 포맷**(`<SYS>`, `<CTX>`, `<PLAYER>`, `<NPC>`)으로 구성
  3. **추론 요청(`utils/hf_client.py`, `models/fallback_model.py`, `pipeline/generator.py`)**
     * preprocess.py에서 통과하지 못한 input은 `models/fallback_model.py`에서 fallback text 생성
     * preprocess.py에 통과한 input은 `pipeline/generator.py` 에서 payload구성
     * 페이로드: `prompt`, `npc_id`, `persona_tags`, `gen_params`(temperature, max\_new\_tokens 등)
     * HF Spaces의 `/predict_main` 으로 HTTP POST
  
  4. **후처리(`pipeline/postprocess.py`)**
  
     * 모델 응답에서 \*\*대사 텍스트, delta(연속값), flag(이벤트)\*\*를 파싱/정규화
     * 예: `flags`는 시그모이드+threshold, `delta`는 범위 클램프·라운딩
  5. **게임 서버 응답(`schemas.py`)**
  
     * 표준화 JSON으로 반환
  
     ```json
     {
       "text": "NPC의 대답...",
       "delta": {"trust": 0.10, "relationship": 0.08},
       "flags": {"give_item": true, "npc_main_story": false, "quest_stage_change": false},
       "meta": {"npc_id": "mother_abandoned_factory"}
     }
     ```

* ### 📁 디렉토리 구조

```bash
ai-server/
├── app.py                  # FastAPI 엔트리포인트
├── config.py               # 서버 설정 및 모델 경로 관리
├── schemas.py              # 요청/응답 데이터 구조 정의
├── requirements.txt        # 의존성 패키지 목록

├── pipeline/               # 대화 흐름 처리 모듈
│   ├── __init__.py
│   ├── preprocess.py       # 입력 전처리 및 프롬프트 구성
│   ├── postprocess.py      # 모델 출력 후처리 및 flag/delta 추출
│   └── generator.py        # 모델 추론 요청 처리

├── rag/                    # RAG 기반 flag 해석 모듈
│   ├── __init__.py
│   ├── rag_generator.py    # NPC별 상황에 따른 flag 텍스트 해석
│   └── docs/
│       └── npc_config.json # NPC별 flag 해석 기준 문서

├── utils/                  # 유틸리티 모듈
│   ├── __init__.py
│   ├── hf_client.py        # Hugging Face API 통신 클라이언트
│   └── context_parser.py   # 대화 맥락 파싱 및 구조화

├── models/                 # 모델 로딩 및 fallback 처리
│   └── model_loader.py     # 모델 로딩 유틸리티
```

* ### 주요 모듈

  * **dialogue_manager.py**: 전체 대화 흐름을 제어하며, fallback 처리, 프롬프트 생성, 모델 추론, 후처리까지 담당
  * **preprocess.py**: 플레이어 입력과 NPC 상태를 기반으로 전처리
  * **postprocess.py**: 모델 출력에서 `<RESPONSE>`, `<FLAG>`, `<DELTA>` 태그를 파싱하고, RAG를 통해 flag를 텍스트로 해석
  * **rag_generator.py**: NPC ID, 퀘스트 단계, flag 이름을 기반으로 문서 검색 및 텍스트 반환


* ### 🧩 RAG 기반 Flag 해석 흐름

  * 모델은 수치 기반 flag를 예측
  * `postprocess.py`는 RAG를 통해 해당 수치를 텍스트로 해석
  * 게임 서버는 이를 기반으로 실제 아이템 지급, 퀘스트 진행 등을 결정

```mermaid
graph TD
ModelOutput["FLAG: give_item=0.92, npc_main_story=0.87"]
ModelOutput --> Postprocess
Postprocess --> RAG["retrieve(npc_id:quest_stage:flag_name)"]
RAG --> FlagText["give_item → 금목걸이 지급"]
FlagText --> GameServer
```

---

## 🚀 Hugging Face Spaces (hf-serve/)

### 역할

* **모델 호스팅 + API 엔드포인트**

  * Base LLM(Qwen2.5-3B-Instruct) + **LoRA 어댑터**를 로드해 추론
  * **REST 엔드포인트** 제공: `POST /predict_main` → `{ text, delta[], flags{} }` JSON 반환
* **Gradio UI(옵션)**

  * 같은 Space에서 간단한 인터랙티브 테스트 UI 제공 (버튼·텍스트박스 기반)

### 구성 요소

* **`server.py`:** FastAPI를 기반으로 한 RESTful API 서버 구현
* **`model_utils.py`:** 베이스 모델 + 어댑터 로딩, 토크나이즈/생성
* **`requirements.txt`:** 필요한 Python 패키지 목록 [`transformers`, `peft`, `accelerate`, `fastapi`/`gradio`, 등]

### 배포

* **[Hugging Face Spaces](https://huggingface.co/spaces/m97j/PersonaChatEngine):** `hf-serve/` 디렉토리의 코드를 Hugging Face Spaces에 배포하여 API 엔드포인트 제공
* **AI 서버 통합:** AI 서버는 해당 API 엔드포인트를 호출하여 NPC의 응답을 수신

자세한 설명
> 📄 [Hugging Face Spaces](https://huggingface.co/spaces/m97j/PersonaChatEngine)

---

## 📊 모델 학습 (train/)

- **데이터 구조**: JSONL (npc_id, tags, context, player_utterance, response, delta, flag)
- **학습 방식**: QLoRA 4bit Adapter, MultiHeadTrainer (LM Loss + Delta MSE + Flag BCE)
- **자동 브랜치 관리**: fine-tuning 완료 시 feature branch 생성 → latest 브랜치 덮어쓰기

자세한 설명
> 📄 [Colab Notebook](https://colab.research.google.com/drive/1_-qH8kdoU2Jj58TdaSnswHex-BFefInq?usp=sharing)
> 📄 [HF Model](https://huggingface.co/m97j/npc_LoRA-fps)


---

## 📦 배포 계획

### 1. Docker Hub 업로드
- ai-server Docker 이미지 빌드 후 `m97j/persona-chat-engine:latest`로 푸시
- ARM 아키텍처 기반 빌드 지원

### 2. Oracle Cloud 배포
- **선택 이유**: Always Free ARM VM (최대 24GB RAM) → ai-server 모델 로드 시 약 10GB RAM 필요
- 현재 로컬 테스트 중 (Oracle Cloud 가입 실패: 전화번호·카드 인증 문제로 메일 문의 완료)

### 3. 로컬 테스트
- uvicorn으로 ai-server 실행
- Swagger `/docs`에서 API 호출 테스트

---

## 🎥 시연 자료
> **Swagger 기반 로컬 테스트 영상 예정**  
> (아래는 틀 예시 – 실제 영상은 추후 추가)

```
[영상 썸네일]
▶ NPC 대화 요청 → Delta/Flag 응답 → Game-server 적용
```

---

## 🏁 프로젝트 성과
- NPC 신뢰도·관계 상태·퀘스트 이벤트 반영 대화 가능
- Delta/Flag Head로 게임 상태 변화 동시 처리
- RAG 기반 컨텍스트 검색으로 상황별 응답 품질 향상
- Oracle Cloud ARM 무료 인스턴스 + Docker Hub + HF Spaces 통합 배포 설계

---

## 📁 포트폴리오 연계

* **[FPS Game](https://github.com/m97j/fpsgame)**: 이벤트 테스트 및 게임 루프 연계
* **[Persona Chat Engine](https://github.com/m97j/persona-chat-engine)**: 멀티 NPC, 스토리/퀘스트 전개 파이프라인
* 이 두 프로젝트는 통합적으로 플레이어 경험 설계와 AI NPC 구현 능력을 강조

---

