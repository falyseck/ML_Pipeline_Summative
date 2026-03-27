"""
prediction.py
-------------
Handles model inference for the Thyroid Cancer Classification pipeline.
Supports single image prediction, batch prediction, and confidence scoring.
Used directly by the FastAPI backend.
"""

import io
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# ── Lazy imports to avoid loading TF until needed ─────────────────────────────
_model      = None
_model_meta = None

MODELS_DIR = Path('./models')
MODEL_H5   = MODELS_DIR / 'thyroid_efficientnet.h5'
META_PATH  = MODELS_DIR / 'model_meta.pkl'

IMG_SIZE   = (224, 224)
CLASS_NAMES = ['benign', 'malignant']


def _load_model():
    """Lazily load and cache the model (loaded once on first call)."""
    global _model
    if _model is None:
        import tensorflow as tf
        from tensorflow import keras
        if not MODEL_H5.exists():
            raise FileNotFoundError(
                f'Model not found at {MODEL_H5}. '
                'Train the model first using model.py'
            )
        print(f'Loading model from {MODEL_H5} ...')
        _model = keras.models.load_model(str(MODEL_H5), compile=False, safe_mode=False)
        print('✅ Model loaded.')
    return _model


def _load_meta() -> Dict:
    """Lazily load and cache model metadata."""
    global _model_meta
    if _model_meta is None:
        if not META_PATH.exists():
            # Fallback defaults if meta not saved yet
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
    """
    Preprocess a PIL Image for EfficientNetB0 inference.

    Parameters
    ----------
    img : PIL Image (any mode, any size)

    Returns
    -------
    numpy array of shape (1, 224, 224, 3)
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img  = img.convert('RGB').resize(IMG_SIZE)
    arr  = np.array(img, dtype=np.float32)
    arr  = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict_from_path(
    image_path: str,
    threshold: Optional[float] = None
) -> Dict:
    """
    Predict class for a single image given its file path.

    Parameters
    ----------
    image_path : path to image file (.jpg / .jpeg / .png)
    threshold  : decision threshold (uses optimal from meta if None)

    Returns
    -------
    dict with keys: label, confidence, prob_benign, prob_malignant,
                    threshold_used, inference_time_ms
    """
    model    = _load_model()
    meta     = _load_meta()
    thresh   = threshold if threshold is not None else meta['optimal_threshold']

    img      = Image.open(image_path)
    arr      = _preprocess_pil(img)

    t0               = time.time()
    prob_malignant   = float(model.predict(arr, verbose=0)[0][0])
    inference_ms     = round((time.time() - t0) * 1000, 2)

    return _build_result(prob_malignant, thresh, inference_ms)


def predict_from_bytes(
    image_bytes: bytes,
    threshold: Optional[float] = None
) -> Dict:
    """
    Predict class for a single image given raw bytes.
    Used by the FastAPI /predict endpoint.

    Parameters
    ----------
    image_bytes : raw bytes from file upload (UploadFile.read())
    threshold   : decision threshold (uses optimal from meta if None)

    Returns
    -------
    dict with keys: label, confidence, prob_benign, prob_malignant,
                    threshold_used, inference_time_ms
    """
    model    = _load_model()
    meta     = _load_meta()
    thresh   = threshold if threshold is not None else meta['optimal_threshold']

    img      = Image.open(io.BytesIO(image_bytes))
    arr      = _preprocess_pil(img)

    t0             = time.time()
    prob_malignant = float(model.predict(arr, verbose=0)[0][0])
    inference_ms   = round((time.time() - t0) * 1000, 2)

    return _build_result(prob_malignant, thresh, inference_ms)


def predict_batch(
    image_paths: List[str],
    threshold: Optional[float] = None
) -> List[Dict]:
    """
    Predict classes for a list of image paths (more efficient than looping).

    Parameters
    ----------
    image_paths : list of image file paths
    threshold   : decision threshold (uses optimal from meta if None)

    Returns
    -------
    list of result dicts (same format as predict_from_path)
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    model  = _load_model()
    meta   = _load_meta()
    thresh = threshold if threshold is not None else meta['optimal_threshold']

    # Build batch array
    arrays = []
    for p in image_paths:
        img = Image.open(p).convert('RGB').resize(IMG_SIZE)
        arr = preprocess_input(np.array(img, dtype=np.float32))
        arrays.append(arr)

    batch = np.stack(arrays, axis=0)

    t0     = time.time()
    probs  = model.predict(batch, verbose=0).ravel()
    total_ms = round((time.time() - t0) * 1000, 2)
    per_ms   = round(total_ms / len(image_paths), 2)

    results = []
    for prob in probs:
        results.append(_build_result(float(prob), thresh, per_ms))
    return results


def _build_result(
    prob_malignant: float,
    threshold: float,
    inference_ms: float
) -> Dict:
    """
    Build a standardised prediction result dictionary.

    Parameters
    ----------
    prob_malignant : raw sigmoid output (probability of malignant)
    threshold      : decision boundary
    inference_ms   : time taken for model.predict() in milliseconds

    Returns
    -------
    dict with label, confidence, probabilities, threshold, timing
    """
    prob_benign = 1.0 - prob_malignant
    label       = 'Malignant' if prob_malignant >= threshold else 'Benign'
    confidence  = prob_malignant if label == 'Malignant' else prob_benign

    return {
        'label'           : label,
        'confidence'      : round(confidence * 100, 2),
        'prob_benign'     : round(prob_benign * 100, 2),
        'prob_malignant'  : round(prob_malignant * 100, 2),
        'threshold_used'  : round(threshold, 4),
        'inference_time_ms': inference_ms
    }


def get_model_info() -> Dict:
    """
    Return model metadata and performance metrics.
    Called by the /model-info and /health API endpoints.

    Returns
    -------
    dict with version, metrics, class names, threshold
    """
    meta = _load_meta()
    return {
        'model_version'    : meta.get('model_version', 'v3'),
        'class_names'      : meta.get('class_names', CLASS_NAMES),
        'img_size'         : meta.get('img_size', IMG_SIZE),
        'optimal_threshold': meta.get('optimal_threshold', 0.5),
        'metrics'          : meta.get('metrics', {}),
        'model_loaded'     : _model is not None,
        'model_path'       : str(MODEL_H5)
    }


def reload_model() -> None:
    """
    Force reload the model from disk.
    Called after retraining to pick up the new weights without restarting.
    """
    global _model, _model_meta
    _model      = None
    _model_meta = None
    _load_model()
    _load_meta()
    print('✅ Model reloaded from disk.')


if __name__ == '__main__':
    # Quick test — predict on a sample image from test set
    import glob

    test_images = glob.glob('./data/test/benign/*.jpg')
    if test_images:
        result = predict_from_path(test_images[0])
        print('\n── Test Prediction ─────────────────────────')
        for k, v in result.items():
            print(f'  {k:22s}: {v}')
    else:
        print('No test images found. Run split_dataset() first.')