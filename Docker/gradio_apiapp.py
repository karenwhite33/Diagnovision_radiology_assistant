import gradio as gr
import uvicorn
import requests
import threading
from apiapp import app

# FastAPI endpoints (dentro de Docker deben apuntar a 127.0.0.1)
TEXT_API_URL = "http://127.0.0.1:8000/upload_text/"
IMAGE_API_URL = "http://127.0.0.1:8000/upload_image/"

# Function to send text to FastAPI
def query_text_api(text):
    response = requests.post(TEXT_API_URL, json={"text": text})
    return response.json()["predictions"]

# Function to send images to FastAPI
def query_image_api(image):
    files = {"file": image}
    response = requests.post(IMAGE_API_URL, files=files)
    return response.json()["predictions"]

# Gradio UI function
def gradio_interface(text_input, image_input):
    text_result = query_text_api(text_input) if text_input else "No text provided."
    image_result = query_image_api(image_input) if image_input is not None else "No image uploaded."
    return text_result, image_result

# Define Gradio Interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Enter Medical Report"),
        gr.Image(type="filepath", label="Upload X-ray Image")
    ],
    outputs=[
        gr.Textbox(label="Text Analysis Result"),
        gr.Textbox(label="Image Analysis Result")
    ],
    title="DiagnoVision Report",
    description="Upload a X-ray image with the medical report to analyze possible conditions."
)

# Run FastAPI & Gradio together
def start_servers():
    # Run FastAPI in a separate thread
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    threading.Thread(target=run_fastapi, daemon=True).start()

    # Run Gradio UI
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

# Start both servers
if __name__ == "__main__":
    start_servers()
