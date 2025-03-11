import pandas as pd

# Cargar el dataset
file_path = "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/full_dataset.csv" 
df = pd.read_csv(file_path)

# Lista de columnas de patologías
pathology_columns = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture"
]

# Reemplazar los 0.5 por 0 en todas las patologías
df[pathology_columns] = df[pathology_columns].replace(0.5, 0)

# Generar un reporte de valores nulos después del cambio
nans_report = df.isnull().sum()

# Mostrar el shape final del dataset
print(f"Forma final del dataset después de la conversión: {df.shape}")

# Mostrar si se generaron valores nulos
print("\nReporte de valores nulos por columna:")
print(nans_report[nans_report > 0])

# Guardar el dataset limpio si es necesario
output_path = "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/full_df_incert.csv"  
df.to_csv(output_path, index=False)

print(f"\nDataset guardado en: {output_path}")
