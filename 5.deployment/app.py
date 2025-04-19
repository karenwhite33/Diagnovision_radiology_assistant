import gradio as gr
from fastapi.testclient import TestClient
import apiapp4
from apiapp4 import app, upload_text, upload_image, generate_report

# Must call startup loader manually,
# because TestClient does NOT trigger FastAPI startup events:
apiapp4.load_models()

client = TestClient(app)

def txt_fn(txt):
    return client.post("/upload_text/", json={"text": txt}) \
                 .json().get("predictions", {})

def img_fn(img_path):
    with open(img_path, "rb") as f:
        return client.post("/upload_image/", files={"file": f}) \
                     .json().get("predictions", {})

def rpt_fn(t_dict, i_dict):
    payload = {
        "texto1": {"predictions": t_dict},
        "texto2": {"predictions": i_dict}
    }
    return client.post("/generate_report/", json=payload) \
                 .json().get("report", "")

iface = gr.Interface(
    fn=lambda t,i: (txt_fn(t), img_fn(i), rpt_fn(txt_fn(t), img_fn(i))),
    inputs=[
        gr.Textbox(label="Enter Medical Report"),
        gr.Image(type="filepath", label="Upload X‑ray Image"),
    ],
    outputs=[
        gr.Textbox(label="Text Analysis Result"),
        gr.Textbox(label="Image Analysis Result"),
        gr.Textbox(label="Final Medical Report"),
    ],
    title="DiagnoVision Multimodal Radiology Assistant",
    description="Upload an X‑ray image and paste the report text to get a fused analysis."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
