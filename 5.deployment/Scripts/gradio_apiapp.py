import gradio as gr
import uvicorn
import requests
import threading
from apiapp4 import app

TEXT_API_URL   = "http://0.0.0.0:8000/upload_text/"
IMAGE_API_URL  = "http://0.0.0.0:8000/upload_image/"
REPORT_API_URL = "http://0.0.0.0:8000/generate_report/"

def query_text_api(text):
    r = requests.post(TEXT_API_URL, json={"text": text})
    return r.json().get("predictions", "Error in text response.")

def query_image_api(image_path):
    if not image_path:
        return "No image uploaded."
    with open(image_path, "rb") as f:
        r = requests.post(IMAGE_API_URL, files={"file": f})
    return r.json().get("predictions", "Error in image response.")

def query_generate_report(text_data, image_data):
    payload = {
      "texto1": {"predictions": text_data},   # note: swapped to match API
      "texto2": {"predictions": image_data}
    }
    r = requests.post(REPORT_API_URL, json=payload)
    return r.json().get("report", "Error generating report.")

def gradio_interface(text_input, image_input):
    txt = query_text_api(text_input) if text_input else "No text provided."
    img = query_image_api(image_input)    if image_input else "No image provided."
    rpt = query_generate_report(txt, img)
    return txt, img, rpt

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Enter Medical Report"),
        gr.Image(type="filepath", label="Upload X-ray Image"),
    ],
    outputs=[
        gr.Textbox(label="Text Analysis Result"),
        gr.Textbox(label="Image Analysis Result"),
        gr.Textbox(label="Final Medical Report"),
    ],
    title="DiagnoVision Report",
    description="Upload an X-ray image and enter the doctor's report to generate a fused medical analysis."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
