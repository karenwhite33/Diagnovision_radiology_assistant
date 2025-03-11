import joblib

# Cargar el modelo RF y el vectorizador
rf_model_path = 'D:/AI Bootcamp Github/Proyecto FInal/Diagnovision/models/rf_models_cpu.pkl'
vectorizer_path = 'D:/AI Bootcamp Github/Proyecto FInal/Diagnovision/models/vectorizer2_cpu.pkl'

# Cargar el diccionario de modelos RF y el vectorizador
models = joblib.load(rf_model_path)  # models es un diccionario
vectorizer = joblib.load(vectorizer_path)

# Patologías a evaluar
patologia_cols = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture'
]

# Función para realizar la predicción
def clasificar_patologias(texto_clinico, vectorizer=vectorizer, models=models, patologia_cols=patologia_cols, umbral=0.5):
    # Vectorizar el texto usando el vectorizador cargado
    texto_tfidf = vectorizer.transform([texto_clinico])

    # Realizar predicciones para cada patología
    resultados = {}

    for i, col in enumerate(patologia_cols):
        # Obtener el modelo para esta patología
        modelo = models[col]

        # Predecir probabilidad
        prob = modelo.predict_proba(texto_tfidf)

        # Obtener la probabilidad de la patología
        probabilidad = prob[:, 1].item()

        # Clasificar en "Positivo", "Incierto" o "Negativo"
        if probabilidad >= 0.5:
            estado = "Positivo"
        elif 0.35 <= probabilidad < 0.5:
            estado = "Incierto"
        else:
            estado = "Negativo"

        # Guardar los resultados
        resultados[col] = {
            "estado": estado,
            "valor": round(float(probabilidad), 4),
            "probabilidad": True
        }

    return resultados

# Ejemplo de uso
nuevo_caso = "stable tracheostomy tube and redemonstration of feeding tube coursing into the upper abdomen. surgical drains overlie the neck. new right pleural pigtail catheter. stable abnormal mediastinal contour. mild improvement in right pleural effusion. increased left pleural effusion. mildly improved right basilar opacities."
resultados = clasificar_patologias(nuevo_caso)

# Mostrar resultados
print("Texto radiólogo:", nuevo_caso)
print("\nResultados de la predicción:")
print("__________________________________________________________")
for patologia, info in resultados.items():
    print(f"{patologia}: {info['estado']} (Prob: {info['valor']:.4f})")
