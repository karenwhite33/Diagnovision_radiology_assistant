import gradio as gr
from fastapi.testclient import TestClient
from apiapp4 import app

client = TestClient(app)

def query_text_api(text):
    response = client.post("/upload_text/", json={"text": text})
    return response.json().get("predictions", "Error in text response.")

def query_image_api(image_path):
    if not image_path:
        return "No image uploaded."
    with open(image_path, "rb") as f:
        response = client.post("/upload_image/", files={"file": f})
    return response.json().get("predictions", "Error in image response.")

def query_generate_report(text_data, image_data):
    payload = {
        "texto1": {"predictions": text_data},
        "texto2": {"predictions": image_data}
    }
    response = client.post("/generate_report/", json=payload)
    return response.json().get("report", "Error generating report.")

def gradio_interface(text_input, image_input):
    txt = query_text_api(text_input) if text_input else "No text provided."
    img = query_image_api(image_input) if image_input else "No image provided."
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

def start_servers():
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    start_servers()
