import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el dataset limpio
file_path = "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/full_df_incert.csv"
df = pd.read_csv(file_path)

# Lista de columnas de patologías
pathology_columns = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture"
]

# Encontrar la patología más frecuente
most_frequent_pathology = df[pathology_columns].sum().idxmax()
print(f"Usando '{most_frequent_pathology}' para estratificar.")

# Hacer el primer split en Train (70%) y Temp (30%) usando la patología más frecuente
df_train, df_temp = train_test_split(
    df, test_size=0.30, stratify=df[most_frequent_pathology], random_state=42
)

# Ahora, dividir Temp en Val (15%) y Test (15%) con la misma estratificación
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, stratify=df_temp[most_frequent_pathology], random_state=42
)

# Guardar los datasets
df_train.to_csv("D:/AI Bootcamp Github/Proyecto FInal/CheXpert/train_df.csv", index=False)
df_val.to_csv("D:/AI Bootcamp Github/Proyecto FInal/CheXpert/val_df.csv", index=False)
df_test.to_csv("D:/AI Bootcamp Github/Proyecto FInal/CheXpert/test_df.csv", index=False)

# Mostrar los tamaños finales de cada conjunto
print(f"Train Set Shape: {df_train.shape}")
print(f"Validation Set Shape: {df_val.shape}")
print(f"Test Set Shape: {df_test.shape}")

print("\nLos conjuntos han sido guardados correctamente.")
