from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import joblib
from PIL import Image
import io
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import psutil
import gc
from typing import Dict, Any
from contextlib import asynccontextmanager

# Function to print memory usage
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")

# Global variables for models
vit_model = None
rf_model = None
vectorizer = None
fusion_model = None
fusion_tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vit_model, rf_model, vectorizer, fusion_model, fusion_tokenizer

    print("Loading models...")

    # ✅ Updated paths for Docker compatibility
    path_modelo_vit = "/app/models/vit_model_gpu.pth"
    rf_model_path = "/app/models/rf_models_cpu.pkl"
    vectorizer_path = "/app/models/vectorizer2_cpu.pkl"
    fusion_model_path = "/app/models/fusion_model_gpu"

    # Load ViT model
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=False)
    vit_model.head = nn.Linear(vit_model.head.in_features, 12)
    vit_model.load_state_dict(torch.load(path_modelo_vit, map_location="cuda"))
    vit_model.to("cuda")
    vit_model.eval()

    # Load Random Forest model and vectorizer
    rf_model = joblib.load(rf_model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Apply 4-bit quantization to the fusion model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load the fusion model with quantization
    fusion_model = AutoModelForCausalLM.from_pretrained(
        fusion_model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )

    fusion_tokenizer = AutoTokenizer.from_pretrained(
        fusion_model_path,
        padding_side="left",
        truncation=True
    )
    fusion_tokenizer.pad_token = fusion_tokenizer.eos_token

    print(f"✅ Models loaded successfully. Fusion Model running on: {fusion_model.device}")
    print_memory_usage()

    yield  # Startup complete, FastAPI runs

    print("Shutting down... Cleaning up models.")
    del vit_model, rf_model, vectorizer, fusion_model, fusion_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Model to receive clinical text
class TextData(BaseModel):
    text: str

# Function to preprocess images for ViT
def preprocesar_imagen(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0).to("cuda")
    return image

# Endpoint to process images using ViT
@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocesar_imagen(image)

    with torch.no_grad():
        salida = vit_model(image_tensor)

    # Convert logits to probabilities using Sigmoid
    probabilidades = F.sigmoid(salida).cpu().numpy().flatten()
    predicciones = (probabilidades > 0.4).astype(float)

    # Pathology labels
    patologia_cols = [
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
        'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
        'Pleural Effusion', 'Pleural Other', 'Fracture'
    ]

    # Convert to structured text output
    resultado_texto = "\n".join([
        f"{patologia}: Probability = {prob:.4f}, Prediction = {int(pred)}"
        for patologia, prob, pred in zip(patologia_cols, probabilidades, predicciones)
    ])

    return {"predictions": resultado_texto}

# Endpoint to process clinical text using Random Forest
@app.post("/upload_text/")
async def upload_text(data: TextData):
    texto1 = clasificar_patologias(data.text)
    return {"predictions": texto1}

# Function to classify pathologies with Random Forest
def clasificar_patologias(texto_clinico):
    texto_tfidf = vectorizer.transform([texto_clinico])
    patologia_cols = [
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
        'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
        'Pleural Effusion', 'Pleural Other', 'Fracture'
    ]

    resultados = {}
    for col in patologia_cols:
        modelo = rf_model[col]
        probabilidad = modelo.predict_proba(texto_tfidf)[:, 1].item()

        estado = "Positivo" if probabilidad >= 0.5 else "Incierto" if 0.35 <= probabilidad < 0.5 else "Negativo"

        resultados[col] = {
            "estado": estado,
            "valor": round(float(probabilidad), 4),
            "probabilidad": True
        }

    return resultados

# Endpoint to generate the medical report
@app.post("/generate_report/")
async def generate_report(texto1: Dict[str, Any], texto2: Dict[str, Any]):
    texto1_str = texto1["contenido"]
    texto2_str = texto2["predictions"]

    print("Generating medical report...")
    print_memory_usage()

    # Free memory before inference
    gc.collect()
    torch.cuda.empty_cache()

    report = razonamiento_medico_con_fusion(texto1_str, texto2_str)

    print_memory_usage()
    return {"report": report}

# Function for medical reasoning using the fusion model
def razonamiento_medico_con_fusion(texto1, texto2):
    prompt = f"""
    You have two sets of medical results for the same patient:

    ## **1️⃣ Text-based model results:**  
    {texto1}

    ## **2️⃣ Image-based model results:**  
    {texto2}

     ### **IMPORTANT:** Only analyze the listed pathologies. Do not introduce new diseases or medical hypotheses that are not explicitly mentioned in the given results.

     ### **Task:**  
    - Identify the most probable pathologies based strictly on the provided probabilities.  

     **Response format:**  
    ### Analysis
    - Summarize key findings from both text and image-based results.
    - Compare probability values and justify which pathology or pathologies are most likely.
    
    ### Conclusion
    - State concise and definitive the most probable pathology or pathologies with supporting probability values.
    
    **ONLY RETURN the response in the structured format above. DO NOT add explanations or unnecessary reasoning outside of the sections.**
    """

    inputs = fusion_tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=1024,
        truncation=True
    ).to("cuda")

    attention_mask = inputs.attention_mask

    with torch.no_grad():
        outputs = fusion_model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            temperature=0.4, 
            top_p=0.8,
            top_k=8,  
            do_sample=True,  
            num_return_sequences=1
        )

    response = fusion_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace("```markdown", "").replace("```", "").strip()

    if "## Analysis" not in response:
        response = "## Analysis\n- Unable to extract structured analysis.\n\n## Conclusion\n- The model did not provide a structured conclusion."

    return response
