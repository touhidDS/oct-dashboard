# Retinal OCT Classifier – Streamlit Dashboard

Upload an OCT image and see the match percentage for every class, plus a Grad-CAM heatmap.

## Folder
```
oct_dashboard/
├── app.py                          # Streamlit web page (main file)
├── oct_model.py                    # model, prediction, Grad-CAM
├── requirements.txt
├── colab_export.py                 # paste into Colab to export the model
└── model/
    ├── best_oct_model.weights.h5   <- from Colab
    └── model_config.json           <- from Colab
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (the two model files must be included).
2. Go to https://share.streamlit.io -> Create app -> pick the repo, branch `main`, main file `app.py`.
3. Open "Advanced settings" and choose **Python 3.12**, then Deploy.

Research demo only. Not for clinical use.
