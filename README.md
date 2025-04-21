# 💻 **DiagnoVision: Multimodal Assistant for Chest X-Ray Analysis & Report Generation**

![image](https://github.com/user-attachments/assets/8d77e55b-4e63-499f-bb71-826058f9f5c8)


## ⁉️ **Pain Point**

Delays in diagnoses and appropriate treatments due to a lack of technological implementation and continuous training to streamline the workflow of radiologists and physicians when analyzing medical images such as X-rays.

### **Challenges in radiology workflow**

1️⃣ Image Quality Issues

Artifacts, patient movement, or foreign objects that hinder interpretation.
Low-resolution images, making it difficult to identify crucial details.
Overexposed or underexposed images that obscure or distort anatomical structures.

2️⃣ Workload Overload

High volume of studies to review, leading to fatigue and diagnostic errors.
Limited time per case, potentially compromising analysis quality.
3️⃣ Interpretation Variability

Subjectivity: Different radiologists may interpret the same image differently.
Ambiguous Findings: Lesions or abnormalities that are not clearly benign or malignant, requiring additional tests and time.
4️⃣ Case Complexity

Rare pathologies requiring specialized expertise.
Complex patient anatomy that makes interpretation more challenging.
5️⃣ Technical Limitations

Visualization tools lacking advanced features.
Difficulty accessing prior studies due to non-integrated systems.
6️⃣ Communication Gaps

Incomplete reports and challenges in discussing complex cases with other specialists.
7️⃣ Human Errors

Mistakes due to distraction and cognitive bias (e.g., confirmation bias in diagnosis).
Lower-quality images in patients with a high body mass index (BMI).
8️⃣ Regulations and Documentation

Time-consuming need to document and justify every finding.
Compliance with radiation protection and ethical protocols.
These pain points highlight the need for AI-powered solutions like DiagnoVision to enhance efficiency, reduce errors, and support radiologists in their diagnostic workflow.

--------------------------------------------------------------------------------------------------------------------------------------------

# ✅ Our Solution

## **DIAGNOVISION**

AI-powered radiology assistant capable of automatically analyzing chest X-rays using a multimodal approach, combining computer vision and natural language processing (NLP) models.

The tool enables critical case prioritization, structured report generation, and decision support for radiologists, enhancing diagnostic efficiency and accuracy.

DEMO: ⏯️📽️ https://drive.google.com/file/d/1tu_tkyFKcYj-S9EgVjA2kqWz4Jjrzvuo/view?usp=drive_link

![image](https://github.com/user-attachments/assets/58e940f1-9e79-45ba-acec-20b7704a1198)

TRY IT HERE ➡️ https://huggingface.co/spaces/karenwhiteg/diagnovision-app



------------------------------------------------------------------------------------------------------------------------------------------

# 🛠 **Technology Stack**

**Vision Transformer (ViT):** google/vit-base-patch16-224 for automated chest X-ray analysis adjusted for 12 features (pathologies)

![image](https://github.com/user-attachments/assets/64024ee1-31f8-4501-af6f-acdfd4caa112)


**RandomForestClassifier:** Applied to enhance text diagnostic accuracy by classifying pathologies in text with probabilities.

![image](https://github.com/user-attachments/assets/135a5792-4913-4448-bb5a-612e77f8f662)

![image](https://github.com/user-attachments/assets/9080f401-fd35-414d-bd4a-51fb73f93b65)



LLM for Report Generation: FreedomIntelligence/HuatuoGPT-o1-7B for structured, coherent radiology reports.

![image](https://github.com/user-attachments/assets/5db1979f-82ba-4580-bdc8-2fbde9ee6b3a)


📊 Data Sources

CheXpert & CheXpertSmall – Large-scale labeled datasets for chest radiography interpretation.


🚀 Results

✅ High diagnostic accuracy in pathology classification.

✅ Coherent and detailed AI-generated radiology reports.

✅ Improved interpretability of radiological findings.


🌍 Impact

Reduced analysis time, streamlining radiology workflows.

More precise and faster diagnoses, improving patient outcomes.

Decision support for medical professionals, reducing cognitive load.


🔜 Next Steps

🔹 Model optimization for enhanced performance.

🔹 Deployment in clinical environments to assist radiologists.


----------------------------------------------------------------------------------------------------

# **DIAGNOVISION PHASES**

1️⃣ Data Preparation

Chest X-ray preprocessing: Resized to 224x224, normalized pixel values.
Text preprocessing: Tokenized and structured radiology reports for NLP classification.

2️⃣ Model Architecture

Vision Transformer (ViT - google/vit-base-patch16-224): Classified 12 pathologies from X-ray images (89.85% accuracy).

Random Forest NLP classifier: Predicted pathology labels from reports (F1-score: 0.98).

LLM (HuatuoGPT-7B): Generated structured radiology reports based on model outputs.

3️⃣ Model Fusion & Optimization

Combined ViT image predictions with Random Forest probability outputs to improve pathology classification reliability.

4️⃣ Evaluation & Metrics

ViT Model: 89.85% test accuracy.

Random Forest Text Classifier: Precision, recall, and F1-score up to 0.98.

5️⃣ FastAPI Deployment on Google Cloud Run

Deployed the system via FastAPI, exposing endpoints for image upload, pathology classification, and report generation.

Integrated asynchronous processing for real-time inference and response handling.

Hosting on Google Cloud Run for a globally accessible API link with auto-scaling.






