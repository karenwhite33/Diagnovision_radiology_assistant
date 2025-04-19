import gdown
import os

os.makedirs("/app/models", exist_ok=True)

files = {
    "vit_model_gpu.pth": "1-DXMLLmEPgie_1vwmtfM2BEEt560TPVw",
    "rf_models_cpu.pkl": "1LQbHBd2CSiSNTjYBgzaZ5Wkl10W37AYx",
    "vectorizer2_cpu.pkl": "1xUlGjKuiuROucSrWKrgReBrxGIgwSr6x"
}

for name, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    dest = f"/app/models/{name}"
    print(f"Downloading {name}...")
    gdown.download(url, dest, quiet=False)
