import os
import pandas as pd

# Definir las rutas de los datasets
base_path = "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/dataset_extracted"
csv_files = {
    "train": "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/train_df.csv",
    "val": "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/val_df.csv",
    "test": "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/test_df.csv"
}

# Verificar que todas las imágenes en los CSV existan en la carpeta
for split, csv_path in csv_files.items():
    df = pd.read_csv(csv_path)

    # Verificar que la columna con las rutas de imágenes exista
    if "path_to_image" not in df.columns:
        raise ValueError(f"El dataset {split} no tiene una columna llamada 'path_to_image'.")

    missing_images = [img for img in df["path_to_image"] if not os.path.exists(os.path.join(base_path, img))]

    print(f"{split}: {len(missing_images)} imágenes faltantes.")

    if missing_images:
        print(f"Ejemplo de imágenes faltantes en {split}: {missing_images[:5]}")
