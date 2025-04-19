import gradio as gr
from fastapi.testclient import TestClient
from apiapp4 import app, upload_text, upload_image, generate_report

client = TestClient(app)

def txt_fn(txt):
  return client.post("/upload_text/", json={"text": txt}).json()["predictions"]

def img_fn(img_path):
  with open(img_path,"rb") as f:
    return client.post("/upload_image/", files={"file": f}).json()["predictions"]

def rpt_fn(t, i):
  return client.post("/generate_report/",
                     json={"texto1":{"predictions":t},
                           "texto2":{"predictions":i}}).json()["report"]

iface = gr.Interface(
  fn=lambda t,i: (txt_fn(t), img_fn(i), rpt_fn(txt_fn(t),img_fn(i))),
  inputs=[gr.Textbox(), gr.Image(type="filepath")],
  outputs=[gr.Textbox(), gr.Textbox(), gr.Textbox()],
  title="DiagnoVision"
)

if __name__=="__main__":
  iface.launch()
