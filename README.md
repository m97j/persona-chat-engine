# Persona Chat Engine – AI NPC Dialogue System 🎭

[![GitHub stars](https://img.shields.io/github/stars/m97j/persona-chat-engine)](https://github.com/m97j/persona-chat-engine)
[![HF Space](https://img.shields.io/badge/HF%20Spaces(ai_server)-Live-blue)](https://huggingface.co/spaces/m97j/PersonaChatEngine_ai_server)
[![HF Space](https://img.shields.io/badge/HF%20Spaces(hf_serve)-Live-blue,)](https://huggingface.co/spaces/m97j/PersonaChatEngine_hf-serve)
[![HF Model](https://img.shields.io/badge/HF%20Model-npc_LoRA--fps-ff69b4)](https://huggingface.co/m97j/npc_LoRA-fps)
[![Colab](https://img.shields.io/badge/Colab-Notebook-yellow)](https://colab.research.google.com/drive/1_-qH8kdoU2Jj58TdaSnswHex-BFefInq?usp=sharing)

## 📑 Table of Contents
- [📌 Overview](#-Overview)
- [🧭 Architecture & Project Structure](#-Architecture--Project-Structure)
- [⚙️ AI Server (ai-server/)](#-ai-server--Summary)
- [🚀 Hugging Face Inference Serve (hf-serve/)](#-hf-serve--hugging-face-spaces-inference-server)
- [📊 Model Training (train/)](#-train--model-training)
- [🛳️ Deployment Overview (HF Spaces, Dockerfile-based)](#%EF%B8%8F-Deployment-Overview-hf-spaces-dockerfile-based)
- [🎥 Demo & Results](#-Demo--Results)
- [🏁 Project Achievements](#-Project-Achievements)

---

## 📌 Overview

**Persona Chat Engine** is an AI conversation engine for in-game NPC interactions.
It predicts Delta/Flag (trust, relationship, and event triggers) along with natural dialogue based on player choices/actions and NPC states.

* **Core Technologies**: Transformer-based LLM, (Q)LoRA fine-tuning, multi-head learning (Delta/Flag), RAG-based interpretation
* **Output**: Text response + state transition (continuous value) + event flag (multi-label)

---

## 🧭 Architecture & Project Structure

* ### Model Architecture
```mermaid
flowchart LR
subgraph Input
U["Player Utterance"] --> Tok["Tokenizer"]
end

Tok --> Emb["Token Embedding + RoPE"]

subgraph DecoderOnly["Decoder-only Transformer xN (LoRA on Attention/FFN)"]
Attn["Multi-Head Attention (Causal, GQA)"]
R1["Residual + RMSNorm"]
FFN["SwiGLU Feed-Forward"] 
R2["Residual + RMSNorm"] 
end 

Emb --> Attn --> R1 --> FFN --> R2 

R2 --> LMHead["LM Head → Next Token"] 
R2 --> Pool["STATE-token Pooling"] 
Pool --> DeltaHead["Delta Head (2: trust, relationship) [-1,1]"] 
Pool --> FlagHead["Flag Head (NUM_FLAGS, scores 0..1)"] 

classDef op fill:#eef,stroke:#669,stroke-width:1px;
```

---

## 📁 Overview by Root Directory

### ⚙️ `ai-server/` — **Summary**

* **Role**: Receive game server request (FastAPI) → Preprocessing → Call HF Spaces inference → Postprocessing (Delta/Flag) → Return result
* **Configuration**: `app.py` (endpoint), `pipeline/` (pre/postprocess, generator), `rag/` (conditions/meta documentation), `utils/` (HF client)
* **Deployment**: (For detailed runtime descriptions, see **HF Spaces README**)
→ \*\*Spaces **builds/runs** directly from the `Dockerfile` in the repo root, and **automatically rebuilds/restarts** upon **Git push**
* **Details**: 👉 **[HF Spaces\[ai_server\]](https://huggingface.co/spaces/m97j/PersonaChatEngine_ai_server)**

---

### 🚀 `hf-serve/` — **Hugging Face Spaces (Inference Server)**

* **Role**: Loads **Base LLM (Qwen2.5-3B-Instruct)** + **LoRA Adapter** and provides **REST API** (`POST /predict_main`)
* **Key Points**

* `model_utils.py`: Tokenize/generate + LoRA merge/adapt
* `server.py`: FastAPI/Gradio (optional) endpoint
* `requirements.txt`: Inference Server lightweight dependency
* **Details**:
👉 [HF Spaces\[hf-serve\]](https://huggingface.co/spaces/m97j/PersonaChatEngine_hf-serve)
👉 [HF Hub\[model card\]](https://huggingface.co/m97j/npc_LoRA-fps)

---

### 📊 `train/` — **Model Training**

* **Data**: JSONL (`npc_id`, `tags`, `context`, `player_utterance`, `response`, `delta`, `flag`)
* **Training**: **LoRA(QLoRA 4bit)**, **MultiHeadTrainer** (LM Loss + Delta Huber + Flag BCE + Threshold MSE)
* **Output**: LoRA adapter, additional heads (`delta_head.pt`, `flag_head.pt`, `threshold_head.pt`), `flags.json`, `thresholds.json`
* **Branch Strategy**: Automatically increment **feature/** + overwrite `latest`
* **Details**: 👉 [**Colab Notebook**](https://colab.research.google.com/drive/1_-qH8kdoU2Jj58TdaSnswHex-BFefInq?usp=sharing)

---

## 🛳️ Deployment Overview (HF Spaces, Dockerfile-based)

```mermaid
flowchart LR
Repo[GitHub Repo] -- Connect --> HF[Hugging Face Spaces]
HF -- Build with root Dockerfile --> Image[Container]
Image --> Run[Space Runtime]
Repo -- git push --> HF:::hot
classDef hot fill:#E67E22,color:#fff,stroke:#A04000
```

---

## 🧩 Technology Highlights

* **Multi-head Learning**: Simultaneously optimize LM (token prediction) and **Delta/Flag** branches → Game-state-informed responses
* **STATE-token Pooling**: Pool embeddings based on `<STATE>` tokens → State head input consistency
* **RAG Interpretation**: Match flag scores/thresholds with document-based conditions to convert them into **game action text**
* **Postprocessing Validation**: Multi-faceted evaluation using threshold tuning, macro/micro F1, and AUROC/AUPRC
* **Operation**: Reduce operational complexity and increase change implementation speed with the **Spaces self-build** pipeline

---

## 🎥 Demo & Results

* To be updated

---

## 🗺️ Roadmap

* Spaces Multi-Model/Branch Rollout (Blue/Green)
* Automated game server A/B testing
* LoRA quantization/on-demand loading optimization

---

## 📎 Reference Links

* **HF Spaces (Live & Detailed Documentation)**:
  * [ai_server](https://huggingface.co/spaces/m97j/PersonaChatEngine_ai_server)
  * [hf-serve](https://huggingface.co/spaces/m97j/PersonaChatEngine_hf-serve)
* **Model Card**:
  * [HF Hub](https://huggingface.co/m97j/npc_LoRA-fps)
* **Model Structure & Training & Inference Test**:
  * [colab notebook](https://colab.research.google.com/drive/1_-qH8kdoU2Jj58TdaSnswHex-BFefInq?usp=sharing)

---

## 🏁 Project Achievements
- NPC trust, relationship status, and quest events reflected in conversations
- Simultaneous processing of game state changes with Delta/Flag Head
- RAG-based context search for improved context-specific response quality
- Docker Hub + HF Spaces integrated deployment design

---

## 📁 Project Links

* **[FPS Game](https://github.com/m97j/fpsgame)**:
  * Client - Event testing and game loop integration
  * game_server - Generates payloads in the ai_server's ask/endpoint format, applies communication results to actual game data (Game_DB), and handles client communication
* **[Persona Chat Engine](https://github.com/m97j/persona-chat-engine)**: Multi-NPC, story/quest development pipeline
  * ​​These two projects integrate to enhance player experience design and AI NPC implementation capabilities.

---
