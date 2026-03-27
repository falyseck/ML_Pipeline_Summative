"""
prediction.py
-------------
Handles model inference for the Thyroid Cancer Classification pipeline.
Supports single image prediction, batch prediction, and confidence scoring.
Used directly by the FastAPI backend.

v2 fixes:
- Absolute paths so Docker container resolves models/ correctly
- Supports .keras format (version-agnostic) with .h5 fallback
- compile=False on load to avoid quantization_config version mismatch
"""

import io
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# ── Lazy imports ──────────────────────────────────────────────────────────────
_model      = None
_model_meta = None

# ── Absolute paths — Docker-safe ──────────────────────────────────────────────
# __file__ resolves to /app/src/prediction.py inside the container
# parent.parent = /app  →  /app/models/
MODELS_DIR  = Path(__file__).resolve().parent.parent / 'models'
MODEL_KERAS = MODELS_DIR / 'thyroid_efficientnet.keras'  # preferred
MODEL_H5    = MODELS_DIR / 'thyroid_efficientnet.h5'     # fallback
META_PATH   = MODELS_DIR / 'model_meta.pkl'

IMG_SIZE    = (224, 224)
CLASS_NAMES = ['benign', 'malignant']


def _get_model_path() -> Path:
    """Return the best available model file."""
    if MODEL_KERAS.exists():
        return MODEL_KERAS
    if MODEL_H5.exists():
        return MODEL_H5
    raise FileNotFoundError(
        f'No model found in {MODELS_DIR}. '
        f'Expected thyroid_efficientnet.keras or thyroid_efficientnet.h5. '
        f'Run the notebook to train and save the model first.'
    )


def _load_model():
    """
    Lazily load and cache the model (loaded once on first API call).

    Uses compile=False to avoid TF version mismatch errors such as
    the quantization_config unknown kwarg issue.
    """
    global _model
    if _model is None:
        from tensorflow import keras
        model_path = _get_model_path()
        print(f'Loading model from {model_path} ...')
        try:
            _model = keras.models.load_model(str(model_path))
        except Exception as e:
            print(f'Direct load failed ({e}), retrying with compile=False ...')
            _model = keras.models.load_model(str(model_path), compile=False)
            _model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')
                ]
            )
        print('✅ Model loaded successfully.')
    return _model


def _load_meta() -> Dict:
    """Lazily load and cache model metadata."""
    global _model_meta
    if _model_meta is None:
        if not META_PATH.exists():
            _model_meta = {
                'class_names'      : CLASS_NAMES,
                'img_size'         : IMG_SIZE,
                'optimal_threshold': 0.5,
                'model_version'    : 'v3',
                'metrics'          : {}
            }
        else:
            with open(str(META_PATH), 'rb') as f:
                _model_meta = pickle.load(f)
    return _model_meta


def _preprocess_pil(img: Image.Image) -> np.ndarray:
    """Preprocess a PIL Image for EfficientNetB0 inference."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict_from_path(image_path: str, threshold: Optional[float] = None) -> Dict:
    """Predict class for a single image given its file path."""
    model  = _load_model()
    meta   = _load_meta()
    thresh = threshold if threshold is not None else meta['optimal_threshold']
    img    = Image.open(image_path)
    arr    = _preprocess_pil(img)
    t0             = time.time()
    prob_malignant = float(model.predict(arr, verbose=0)[0][0])
    inference_ms   = round((time.time() - t0) * 1000, 2)
    return _build_result(prob_malignant, thresh, inference_ms)


def predict_from_bytes(image_bytes: bytes, threshold: Optional[float] = None) -> Dict:
    """
    Predict class for a single image given raw bytes.
    Used by the FastAPI /predict endpoint.
    """
    model  = _load_model()
    meta   = _load_meta()
    thresh = threshold if threshold is not None else meta['optimal_threshold']
    img    = Image.open(io.BytesIO(image_bytes))
    arr    = _preprocess_pil(img)
    t0             = time.time()
    prob_malignant = float(model.predict(arr, verbose=0)[0][0])
    inference_ms   = round((time.time() - t0) * 1000, 2)
    return _build_result(prob_malignant, thresh, inference_ms)


def predict_batch(image_paths: List[str], threshold: Optional[float] = None) -> List[Dict]:
    """Predict classes for a list of image paths efficiently."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    model  = _load_model()
    meta   = _load_meta()
    thresh = threshold if threshold is not None else meta['optimal_threshold']
    arrays = []
    for p in image_paths:
        img = Image.open(p).convert('RGB').resize(IMG_SIZE)
        arr = preprocess_input(np.array(img, dtype=np.float32))
        arrays.append(arr)
    batch    = np.stack(arrays, axis=0)
    t0       = time.time()
    probs    = model.predict(batch, verbose=0).ravel()
    total_ms = round((time.time() - t0) * 1000, 2)
    per_ms   = round(total_ms / len(image_paths), 2)
    return [_build_result(float(p), thresh, per_ms) for p in probs]


def _build_result(prob_malignant: float, threshold: float, inference_ms: float) -> Dict:
    """Build a standardised prediction result dictionary."""
    prob_benign = 1.0 - prob_malignant
    label       = 'Malignant' if prob_malignant >= threshold else 'Benign'
    confidence  = prob_malignant if label == 'Malignant' else prob_benign
    return {
        'label'            : label,
        'confidence'       : round(confidence * 100, 2),
        'prob_benign'      : round(prob_benign * 100, 2),
        'prob_malignant'   : round(prob_malignant * 100, 2),
        'threshold_used'   : round(threshold, 4),
        'inference_time_ms': inference_ms
    }


def get_model_info() -> Dict:
    """Return model metadata — called by /model-info and /health."""
    meta = _load_meta()
    try:
        model_path = str(_get_model_path())
    except FileNotFoundError:
        model_path = 'not found'
    return {
        'model_version'    : meta.get('model_version', 'v3'),
        'class_names'      : meta.get('class_names', CLASS_NAMES),
        'img_size'         : meta.get('img_size', IMG_SIZE),
        'optimal_threshold': meta.get('optimal_threshold', 0.5),
        'metrics'          : meta.get('metrics', {}),
        'model_loaded'     : _model is not None,
        'model_path'       : model_path
    }


def reload_model() -> None:
    """Force reload model from disk — called after retraining."""
    global _model, _model_meta
    _model = None
    _model_meta = None
    _load_model()
    _load_meta()
    print('✅ Model reloaded from disk.')


if __name__ == '__main__':
    import glob
    test_images = glob.glob('./data/test/benign/*.jpg')
    if test_images:
        result = predict_from_path(test_images[0])
        print('\n── Test Prediction ──────────────────────')
        for k, v in result.items():
            print(f'  {k:22s}: {v}')
    else:
        print('No test images found.')