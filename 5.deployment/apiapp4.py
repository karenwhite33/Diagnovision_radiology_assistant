from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import torch
import joblib
import timm
import os
import torch.nn as nn
from PIL import Image
import io
import torchvision.transforms as T
from typing import Dict, Any

# Threshold for reporting a pathology as present
PRESENT_THRESHOLD = 0.4

# Pathology labels
PATHOLOGIES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture"
]

# Clinician‑mapped recommendations
ACTION_MAP = {
    'Enlarged Cardiomediastinum': 'Evaluate mediastinal contour with CT imaging',
    'Cardiomegaly': 'Obtain echocardiogram to assess cardiac size and function',
    'Lung Opacity': 'Perform contrast-enhanced chest CT to characterize opacity',
    'Lung Lesion': 'Schedule CT-guided biopsy for definitive diagnosis',
    'Edema': 'Assess for heart failure with BNP levels and echocardiogram',
    'Consolidation': 'Initiate antibiotic therapy for suspected infection',
    'Pneumonia': 'Obtain sputum culture and start empirical antibiotics',
    'Atelectasis': 'Recommend chest physiotherapy to re-expand lung segments',
    'Pneumothorax': 'Perform chest tube placement for lung re-expansion',
    'Pleural Effusion': 'Perform thoracentesis and analyze pleural fluid',
    'Pleural Other': 'Evaluate with pleural ultrasound and consider biopsy',
    'Fracture': 'Obtain orthopedic consultation and immobilization'
}

app = FastAPI()
class TextData(BaseModel):
    text: str

# Globals for models
vit_model = None
rf_model = None
vectorizer = None

@app.on_event("startup")
def load_models():
    global vit_model, rf_model, vectorizer

    repo = "karenwhiteg/diagnovision-app"
    # 1) download weights from HF Hub
    vit_p = hf_hub_download(repo_id=repo, filename="vit_model_gpu.pth")
    rf_p  = hf_hub_download(repo_id=repo, filename="rf_models_cpu.pkl")
    vec_p = hf_hub_download(repo_id=repo, filename="vectorizer2_cpu.pkl")

    # 2) load ViT
    vit = timm.create_model("vit_base_patch16_224", pretrained=False)
    vit.head = nn.Linear(vit.head.in_features, len(PATHOLOGIES))
    vit.load_state_dict(torch.load(vit_p, map_location="cpu"))
    vit_model = vit.eval()

    # 3) load RF & TF‑IDF
    rf_model   = joblib.load(rf_p)
    vectorizer = joblib.load(vec_p)

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    tfm = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        out = vit_model(x)
    probs = torch.sigmoid(out).numpy().flatten()
    return {
        "predictions": {
            p: float(round(probs[i],4))
            for i,p in enumerate(PATHOLOGIES)
        }
    }

@app.post("/upload_text/")
async def upload_text(data: TextData):
    tfidf = vectorizer.transform([data.text])
    preds = {
        p: float(round(rf_model[p].predict_proba(tfidf)[:,1].item(),4))
        for p in PATHOLOGIES
    }
    return {"predictions": preds}

@app.post("/generate_report/")
async def generate_report(texto1: Dict[str,Any], texto2: Dict[str,Any]):
    t1 = texto1.get("predictions", texto1)
    t2 = texto2.get("predictions", texto2)
    combined = {
        p: round((t1.get(p,0) + t2.get(p,0))/2, 4)
        for p in PATHOLOGIES
    }
    present = [(p,combined[p]) for p in PATHOLOGIES if combined[p] >= PRESENT_THRESHOLD]

    lines = []
    if not present:
        lines += [
            "Primary Findings: None above threshold",
            "Explanation: No pathology meets the reporting threshold.",
            "Medical Conclusion: Unremarkable study based on current models.",
            "Recommendations: 1. Continue routine clinical monitoring; 2. Re-evaluate if symptoms develop."
        ]
    else:
        conds = [f"{p} ({prob:.2f})" for p,prob in present]
        lines.append(f"Primary Findings: {', '.join(conds)}")
        for p,prob in present:
            lines.append(f"- {p}: probability {prob:.2f}, recommends {ACTION_MAP[p]}")
        main = [p for p,_ in present]
        lines.append(f"Medical Conclusion: Findings are consistent with {', '.join(main)}.")
        recs = [ACTION_MAP[p] for p,_ in present][:2]
        lines.append(f"Recommendations: 1. {recs[0]}; 2. {recs[1]}")
    report = "---------------------------------\n" + "\n".join(lines) + "\n---------------------------------"
    return {"report": report}
