"""Model code for the OCT dashboard.
Rebuilds the SAME architecture as the Colab notebook, loads the trained weights,
and gives prediction (with optional flip TTA) and Grad-CAM."""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, MobileNetV2, VGG16
from tensorflow.keras.applications import densenet, efficientnet, mobilenet_v2, vgg16

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_oct_model.weights.h5")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

# ---- must stay identical to the notebook (same layers, same order) ----------
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.1),
], name="augmentation")

BACKBONES = {
    "MobileNetV2": (MobileNetV2, mobilenet_v2.preprocess_input),
    "VGG16": (VGG16, vgg16.preprocess_input),
    "DenseNet121": (DenseNet121, densenet.preprocess_input),
    "EfficientNetB0": (EfficientNetB0, efficientnet.preprocess_input),
}


def _add_head(x, num_classes):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(num_classes, activation="softmax", dtype="float32")(x)


def _build_custom_cnn(img_size, num_classes):
    inputs = layers.Input((img_size, img_size, 3))
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in [32, 64, 128, 256, 256]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.1)(x)
    return models.Model(inputs, _add_head(x, num_classes), name="Custom_CNN")


def _build_pretrained(name, img_size, num_classes):
    backbone, preprocess = BACKBONES[name]
    inputs = layers.Input((img_size, img_size, 3))
    x = augment(inputs)
    x = preprocess(x)
    base = backbone(include_top=False, weights=None, input_tensor=x)  # weights come from our file
    return models.Model(inputs, _add_head(base.output, num_classes), name=name)


def build_model(name, img_size, num_classes):
    if name == "Custom CNN":
        return _build_custom_cnn(img_size, num_classes)
    if name in BACKBONES:
        return _build_pretrained(name, img_size, num_classes)
    raise ValueError(f"Unknown model name in model_config.json: {name!r}")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_model():
    """Returns (model, config). Raises FileNotFoundError if the model files are missing."""
    for p in (CONFIG_PATH, WEIGHTS_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    cfg = load_config()
    model = build_model(cfg["model_name"], cfg["img_size"], len(cfg["class_names"]))
    model.load_weights(WEIGHTS_PATH)
    return model, cfg


# ---- prediction ------------------------------------------------------------
def preprocess(pil_image, img_size):
    """Same steps as the notebook's load_image(): RGB -> resize -> uint8 -> float32 (0..255)."""
    arr = np.array(pil_image.convert("RGB"))
    img = tf.image.resize(arr, (img_size, img_size))
    img = tf.cast(tf.cast(img, tf.uint8), tf.float32)
    return tf.expand_dims(img, 0).numpy()


def predict_probs(model, batch, use_tta=True):
    probs = model.predict(batch, verbose=0)
    if use_tta:  # same as the notebook: average the image and its mirrored copy
        flipped = tf.image.flip_left_right(batch).numpy()
        probs = (probs + model.predict(flipped, verbose=0)) / 2
    return probs[0]


# ---- Grad-CAM --------------------------------------------------------------
def _last_4d_layer(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None


def gradcam_overlay(model, batch, class_idx=None, alpha=0.4):
    """Returns an RGB uint8 image with the Grad-CAM heatmap on top, or None if not possible."""
    from matplotlib import cm
    name = _last_4d_layer(model)
    if name is None:
        return None
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(name).output, model.output])
    x = tf.convert_to_tensor(batch)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_out[0] @ weights[..., tf.newaxis])
    heat = tf.maximum(heat, 0)
    heat = (heat / (tf.reduce_max(heat) + 1e-8)).numpy()
    size = batch.shape[1]
    heat = tf.image.resize(heat[..., None], (size, size)).numpy().squeeze()
    colored = cm.jet(heat)[..., :3]
    overlay = np.clip((1 - alpha) * batch[0] / 255.0 + alpha * colored, 0, 1)
    return (overlay * 255).astype(np.uint8)
