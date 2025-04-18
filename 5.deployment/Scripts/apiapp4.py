from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import joblib
from PIL import Image
import io
import numpy as np
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import psutil
import gc
import os
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Threshold for reporting a pathology as present
PRESENT_THRESHOLD = 0.4  # probabilities ≥ this are considered present

# Pathology labels
PATHOLOGIES = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
    'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture'
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

# Memory usage helper
def print_memory_usage():
    p = psutil.Process()
    print(f"Memory usage: {p.memory_info().rss / (1024*1024):.1f} MB")

app = FastAPI()

class TextData(BaseModel):
    text: str

# Globals
vit_model = None
rf_model = None
vectorizer = None

# Load all models on startup
def get_models_dir():
    if os.path.exists('/app/models'):
        return '/app/models'
    base = os.path.dirname(__file__)
    cand = os.path.abspath(os.path.join(base, '..', 'models'))
    if os.path.exists(cand):
        return cand
    return r"D:\AI Bootcamp Github\Proyecto FInal\Diagnovision\models"

@app.on_event('startup')
def load_models():
    global vit_model, rf_model, vectorizer
    base = get_models_dir()
    # ViT image model
    vit_path = os.path.join(base, 'vit_model_gpu.pth')
    vit = timm.create_model('vit_base_patch16_224', pretrained=False)
    vit.head = nn.Linear(vit.head.in_features, len(PATHOLOGIES))
    vit.load_state_dict(torch.load(vit_path, map_location='cuda'))
    vit_model = vit.to('cuda').eval()
    # RandomForest & TFIDF
    rf_model = joblib.load(os.path.join(base, 'rf_models_cpu.pkl'))
    vectorizer = joblib.load(os.path.join(base, 'vectorizer2_cpu.pkl'))

@app.post('/upload_image/')
async def upload_image(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tensor = tfm(img).unsqueeze(0).to('cuda')
    with torch.no_grad():
        out = vit_model(tensor)
    probs = torch.sigmoid(out).cpu().numpy().flatten()
    return {'predictions': {PATHOLOGIES[i]: float(round(probs[i],4)) for i in range(len(PATHOLOGIES))}}

@app.post('/upload_text/')
async def upload_text(data: TextData):
    tfidf = vectorizer.transform([data.text])
    preds = {p: float(round(rf_model[p].predict_proba(tfidf)[:,1].item(),4)) for p in PATHOLOGIES}
    return {'predictions': preds}

@app.post('/generate_report/')
async def generate_report(texto1: Dict[str,Any], texto2: Dict[str,Any]):
    # Accept either nested 'predictions' or direct dict
    t1 = texto1.get('predictions') if 'predictions' in texto1 else texto1
    t2 = texto2.get('predictions') if 'predictions' in texto2 else texto2
    # average
    combined = {p: round((t1.get(p,0)+t2.get(p,0))/2,4) for p in PATHOLOGIES}
    # select present
    present = [(p,combined[p]) for p in combined if combined[p]>=PRESENT_THRESHOLD]
    lines = []
    if not present:
        lines.append('Primary Findings: None above threshold')
        lines.append('Explanation: No pathology meets the reporting threshold.')
        lines.append('Medical Conclusion: Unremarkable study based on current models.')
        lines.append('Recommendations: 1. Continue routine clinical monitoring; 2. Re-evaluate if symptoms develop.')
    else:
        cond_list = [f"{p} ({prob:.2f})" for p,prob in present]
        lines.append(f"Primary Findings: {', '.join(cond_list)}")
        for p,prob in present:
            lines.append(f"- {p}: probability {prob:.2f}, recommends {ACTION_MAP[p]}")
        main_conds = [p for p,_ in present]
        lines.append(f"Medical Conclusion: Findings are consistent with {', '.join(main_conds)}.")
        recs = [ACTION_MAP[p] for p,_ in present][:2]
        lines.append(f"Recommendations: 1. {recs[0]}; 2. {recs[1]}")
    report = '---------------------------------\n' + '\n'.join(lines) + '\n---------------------------------'
    return {'report': report}
