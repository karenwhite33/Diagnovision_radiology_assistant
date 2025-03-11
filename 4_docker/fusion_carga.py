import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cargar el modelo Fusion (HuatuoGPT)
model_path = "D:/AI Bootcamp Github/Proyecto FInal/Diagnovision/models/fusion_model_gpu"
device = "cpu"  # Usar CPU

# Inicializar el modelo y el tokenizador
fusion_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
fusion_tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Modelo Fusion cargado exitosamente!")

def razonamiento_medico_con_fusion(texto1, texto2):
    prompt = f"""
    Dado los siguientes resultados médicos de un mismo paciente, determina cuál es la patología más probable considerando las probabilidades asociadas. Ten en cuenta que el primer texto viene de un modelo de texto y el segundo viene de un modelo de imagen.

    Resultados modelo de texto 1:
    {texto1}

    Resultados modelo de imagen 2:
    {texto2}

    Proporciona un razonamiento paso a paso para llegar a la conclusión. Y después dame un pequeño documento con las conclusiones médicas de la patología y los siguientes pasos que hacer con el paciente en función de su principal patología.
    """

    inputs = fusion_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)

    # Generar respuesta del modelo
    outputs = fusion_model.generate(
        inputs.input_ids,
        max_new_tokens=1024,
        temperature=0.4,
        top_p=0.8,
        top_k=8,
        do_sample=True,
        num_return_sequences=1
    )

    respuesta = fusion_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta
