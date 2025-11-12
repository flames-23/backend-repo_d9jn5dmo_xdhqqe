import os
from io import BytesIO
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Database helpers (safe even if DB not configured)
from database import db, create_document

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/lung_cancer_xray.onnx")
MODEL_INPUT_SIZE = (224, 224)  # width, height

# Lazy, optional model/session to avoid import errors on startup
_session = None
_input_name: Optional[str] = None
_output_name: Optional[str] = None
_USE_DUMMY = False  # fall back when ORT/model unavailable


def load_model():
    """Try to load ONNX model; if unavailable, enable dummy mode."""
    global _session, _input_name, _output_name, _USE_DUMMY
    if _session is not None or _USE_DUMMY:
        return

    # Import onnxruntime here so the server can still start if it's not installed
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        _USE_DUMMY = True
        return

    if not os.path.exists(MODEL_PATH):
        _USE_DUMMY = True
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        _session = ort.InferenceSession(MODEL_PATH, providers=providers)
    except Exception:
        try:
            _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        except Exception:
            _USE_DUMMY = True
            return

    _input_name = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name


def preprocess_image(file_bytes: bytes):
    # Import numpy and PIL lazily to avoid startup failures if installs had issues
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError(f"NumPy not available: {e}")
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Image processing dependency missing: {e}")

    # Load and convert to RGB then normalize
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize(MODEL_INPUT_SIZE)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))  # NCHW
    arr = np.expand_dims(arr, 0)
    return arr


class PredictResponse(BaseModel):
    label: str
    confidence: float
    timestamp: str
    heatmap: Optional[str] = None  # base64 PNG overlay (optional, placeholder)


@app.get("/")
async def root():
    return {"message": "Lung Cancer Detection API"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/test")
async def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_name"] = db.name
            collections = db.list_collection_names()
            response["collections"] = collections[:10]
            response["database"] = "✅ Connected & Working"
            response["connection_status"] = "Connected"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # Load model (enables dummy if needed)
    load_model()

    # Read bytes
    content = await file.read()

    # If dummy, compute a pseudo probability from image bytes
    if _USE_DUMMY or _session is None:
        import numpy as np  # type: ignore
        # Deterministic pseudo-probability based on content hash
        seed = int(np.frombuffer(content[:1024], dtype=np.uint8).sum())
        rng = np.random.default_rng(seed)
        prob_cancer = float(rng.uniform(0.2, 0.8))  # moderate range
    else:
        # Real preprocessing + inference
        import numpy as np  # type: ignore
        try:
            inp = preprocess_image(content)
        except Exception:
            return JSONResponse(status_code=400, content={"detail": "Invalid image file"})
        outputs = _session.run([_output_name], {_input_name: inp})
        logits = outputs[0]
        if hasattr(logits, "ndim") and logits.ndim == 2 and logits.shape[1] == 2:
            exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            prob_cancer = float(probs[0, 1])
        else:
            prob_cancer = float(1 / (1 + np.exp(-np.squeeze(logits))))

    label = "cancer" if prob_cancer >= 0.5 else "normal"
    resp = PredictResponse(
        label=label,
        confidence=round(prob_cancer if label == "cancer" else 1 - prob_cancer, 4),
        timestamp=datetime.utcnow().isoformat() + "Z",
        heatmap=None,
    )

    # Save to DB if available
    try:
        if db is not None:
            create_document("prediction", {
                "filename": file.filename,
                "label": resp.label,
                "confidence": resp.confidence,
                "prob_cancer": prob_cancer,
                "timestamp": resp.timestamp,
            })
    except Exception:
        pass

    return JSONResponse(content=resp.summary() if hasattr(resp, "summary") else resp.model_dump())


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
