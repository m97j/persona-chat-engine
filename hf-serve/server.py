# hf-serve/server.py
import os, json, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

logger = logging.getLogger("hf-serve")
logging.basicConfig(level=logging.INFO)

# Config via ENV
MODEL_BASE = os.getenv("MODEL_BASE", "/models")   # 로컬 마운트 포인트 (optional)
HF_HUB_ID = os.getenv("HF_HUB_ID", None)          # 예: "username/model-with-adapter"
USE_HF_HUB = os.getenv("USE_HF_HUB", "0") == "1"
DEVICE = os.getenv("DEVICE", "cpu")               # "cpu" or "cuda"
LOAD_4BIT = os.getenv("LOAD_4BIT", "0") == "1"    # if you use bitsandbytes / transformers supports it
PEFT_ADAPTER = os.getenv("PEFT_ADAPTER_PATH", None)  # optional adapter name or local path

app = FastAPI(title="hf-serve")

# Simple request models
class PreprocessReq(BaseModel):
    session_id: str
    npc_id: str
    text: str
    context: Dict[str, Any] = {}

class MainReq(BaseModel):
    session_id: str
    npc_id: str
    prompt: str
    max_tokens: Optional[int] = 200

class PostReq(BaseModel):
    session_id: str
    npc_id: str
    text: str
    constraints: Dict[str, Any] = {}


# Model manager: lazy load & wrappers for generate/classify
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        logger.info("Loading model...")

        # import here to avoid heavy import on module load
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # prefer hub if configured
        model_source = None
        if USE_HF_HUB and HF_HUB_ID:
            model_source = HF_HUB_ID
            logger.info(f"Loading from HF hub: {HF_HUB_ID}")
        else:
            # fallback to local dir
            local_main = os.path.join(MODEL_BASE, "main_dialogue_model")
            if os.path.exists(local_main):
                model_source = local_main
                logger.info(f"Loading from local model dir: {local_main}")
            else:
                logger.warning("No model found on hub or local. Running stub responses.")
                self.loaded = False
                return

        # load tokenizer & model (simple - adapt as needed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
        # If using bitsandbytes / 4bit, you may configure load_in_4bit and device_map
        model_kwargs = {}
        if LOAD_4BIT:
            # this requires bitsandbytes + accelerate configured; leave for deploy tuning
            model_kwargs.update({"load_in_4bit": True, "device_map": "auto"})
        else:
            model_kwargs.update({"device_map": {"": DEVICE}} if DEVICE != "cpu" else {})

        self.model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

        # If adapter (LoRA/PEFT) exists, try to load and merge or use peft
        try:
            from peft import PeftModel, PeftConfig
            if PEFT_ADAPTER:
                logger.info(f"Applying PEFT adapter: {PEFT_ADAPTER}")
                self.model = PeftModel.from_pretrained(self.model, PEFT_ADAPTER)
        except Exception as e:
            logger.info("PEFT not applied or not installed: %s", str(e))

        self.loaded = True
        logger.info("Model loaded successfully.")

    def has_model(self, kind: str):
        # simple: if model loaded, return True for main; preprocess/postprocess depends on directories
        if kind == "main":
            return self.loaded
        path = os.path.join(MODEL_BASE, f"{kind}_model")
        return os.path.exists(path) and bool(os.listdir(path))

    def predict_preprocess(self, payload: dict):
        # Optionally implement a real classifier model. For now rule/fallback:
        text = payload.get("text", "")
        if any(x in text for x in ["아이", "죽", "사라졌", "잃"]):
            return {"trigger": "grief", "confidence": 0.95, "source": "local-rule"}
        return {"trigger": None, "confidence": 0.2, "source": "local-rule"}

    def predict_main(self, payload: dict):
        if not self.loaded:
            # fallback echo or simple template
            prompt = payload.get("prompt", "")
            text = "(DEMO) " + prompt[:400]
            return {"text": text}
        # real generation
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_tokens", 200))
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if DEVICE == "cuda":
            input_ids = input_ids.cuda()
            self.model.to("cuda")
        # generate (synchronous)
        gen_ids = self.model.generate(input_ids, max_new_tokens=max_tokens, do_sample=True, top_p=0.9, temperature=0.8)
        out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # return only generated suffix if you want; here we return entire text
        return {"text": out}

    def predict_postprocess(self, payload: dict):
        text = payload.get("text", "")
        constraints = payload.get("constraints", {})
        if constraints.get("no_violence") and ("죽여" in text or "kill" in text):
            cleaned = text.replace("죽여", "[삭제]").replace("kill", "[redacted]")
            return {"text": cleaned, "valid": False, "meta": {"cleaned": True}}
        return {"text": text, "valid": True, "meta": {}}


MM = ModelManager()
MM.load()  # load at startup (will be lazy if no model configured)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MM.loaded}


@app.get("/has_model/{kind}")
async def has_model(kind: str):
    return {"exists": MM.has_model(kind)}


@app.post("/predict_preprocess")
async def predict_preprocess(req: PreprocessReq):
    return MM.predict_preprocess(req.dict())


@app.post("/predict_main")
async def predict_main(req: MainReq):
    return MM.predict_main(req.dict())


@app.post("/predict_postprocess")
async def predict_postprocess(req: PostReq):
    return MM.predict_postprocess(req.dict())
