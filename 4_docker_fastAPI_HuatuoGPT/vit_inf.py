
import torch
import timm
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

# Cargar el modelo ViT ya entrenado
path_modelo_vit = "D:/AI Bootcamp Github/Proyecto Final/Diagnovision/models/vit_model_gpu.pth"

modelo_vit = timm.create_model('vit_base_patch16_224', pretrained=False)
modelo_vit.head = nn.Linear(modelo_vit.head.in_features, 12)
modelo_vit.load_state_dict(torch.load(path_modelo_vit, map_location="cuda"))
modelo_vit.to("cuda")
modelo_vit.eval()

# Etiquetas de las 12 patologías
patologia_cols = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
    'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture'
]

# Función para preprocesar imágenes
def preprocesar_imagen(path_imagen):
    imagen = cv2.imread(path_imagen)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(imagen).unsqueeze(0).to("cuda")

# Ruta de la imagen a procesar
path_imagen = "D:/AI Bootcamp Github/Proyecto Final/CheXpert/dataset_extracted/train/patient21893/study1/view1_frontal.jpg"

# 1️⃣ Inferencia con ViT
imagen_tensor = preprocesar_imagen(path_imagen)

with torch.no_grad():
    salida = modelo_vit(imagen_tensor)

# 2️⃣ Convertir logits en probabilidades con Sigmoid
probabilidades = F.sigmoid(salida).cpu().numpy().flatten()

# 3️⃣ Obtener las predicciones (0 o 1) a partir de las probabilidades
predicciones = (probabilidades > 0.4).astype(float)

# 4️⃣ Imprimir las probabilidades con las patologías
for i in range(len(patologia_cols)):
    print(f"{patologia_cols[i]}: Probability = {probabilidades[i]:.4f}, Prediction = {predicciones[i]:.0f}")

