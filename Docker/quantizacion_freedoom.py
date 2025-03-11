import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configurar quantization con bitsandbytes
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_compute_dtype=torch.float16,  
    llm_int8_enable_fp32_cpu_offload=True  
)

# Cargar el modelo con cuantización
modelo_path = "D:/AI Bootcamp Github/Proyecto FInal/Diagnovision/models/fusion_model_gpu"

try:
    model = AutoModelForCausalLM.from_pretrained(
        modelo_path,
        quantization_config=quantization_config,
        device_map="auto"
    )
    print("✅ El modelo soporta cuantización con bitsandbytes en 4-bit.")
except Exception as e:
    print("❌ El modelo NO soporta cuantización con bitsandbytes.")
    print("Error:", e)
