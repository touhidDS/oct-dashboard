# ============================================================================
# Paste this into a NEW cell at the END of your Colab notebook and run it
# (after the whole notebook has finished, so `best_model` holds the final weights).
# It saves the 2 files the dashboard needs and downloads them.
# ============================================================================
import json, os

export_dir = f"{OUT_DIR}/dashboard_model"
os.makedirs(export_dir, exist_ok=True)

weights_file = f"{export_dir}/best_oct_model.weights.h5"
best_model.save_weights(weights_file)

config = {
    "model_name": best_name,          # e.g. "DenseNet121"  (the champion from stage 1)
    "class_names": list(class_names), # exact order used in training
    "img_size": IMG_SIZE,
}
config_file = f"{export_dir}/model_config.json"
json.dump(config, open(config_file, "w"), indent=2)

print("Saved:", config)
print(f"Weights size: {os.path.getsize(weights_file) / 1e6:.1f} MB")
print("TensorFlow version in Colab:", tf.__version__)   # use the same major.minor version locally

from google.colab import files
files.download(weights_file)
files.download(config_file)
