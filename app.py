import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Neurological Diagnostic System",
    layout="wide"
)

st.title("🧠 AI Neurological Diagnostic System")
st.caption("Deep Transfer Ensemble | Explainable AI | Clinical Decision Support")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():

    vgg = tf.keras.models.load_model(
        "models/vgg16_savedmodel",
        compile=False
    )

    resnet = tf.keras.models.load_model(
        "models/resnet50_savedmodel",
        compile=False
    )

    return vgg, resnet


vgg16_model, resnet50_model = load_models()

CLASS_NAMES = [
    "Cognitively Normal",
    "Mild Cognitive Impairment",
    "Alzheimer’s Disease"
]

IMG_SIZE = 224

# -------------------------------------------------
# PREPROCESS
# -------------------------------------------------
def preprocess_image(image):
    image = np.array(image.convert("L"))
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.GaussianBlur(image, (5,5), 0)
    image = image / 255.0
    image = np.stack([image]*3, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image


# -------------------------------------------------
# ENSEMBLE
# -------------------------------------------------
def ensemble_predict(img_tensor):
    pred_vgg = vgg16_model.predict(img_tensor, verbose=0)
    pred_resnet = resnet50_model.predict(img_tensor, verbose=0)

    ensemble_prob = (0.6 * pred_vgg + 0.4 * pred_resnet)
    pred_class = np.argmax(ensemble_prob)

    return pred_class, ensemble_prob[0]


# -------------------------------------------------
# CONFIDENCE INTERPRETATION
# -------------------------------------------------
def confidence_level(conf):

    if conf < 0.45:
        return "Low Confidence", "Recommend further clinical testing."

    elif conf < 0.70:
        return "Moderate Confidence", "Clinical correlation advised."

    else:
        return "High Confidence", "Strong imaging biomarkers detected."


# -------------------------------------------------
# MEDICAL ADVICE
# -------------------------------------------------
def medical_advice(pred_class):

    if pred_class == 0:
        return """
**Result:** No significant neurodegenerative patterns detected.

**Advice:**
• Maintain physical exercise  
• Stay cognitively active  
• Schedule periodic health evaluations  
• Ensure proper sleep and nutrition  
"""

    elif pred_class == 1:
        return """
**Result:** Findings suggest Mild Cognitive Impairment (MCI).

**Advice:**
• Seek neurological consultation  
• Early intervention may delay progression  
• Perform periodic cognitive screening  
• Stay socially engaged  
"""

    else:
        return """
**Result:** Patterns associated with Alzheimer’s Disease detected.

⚠️ This AI system is NOT a medical diagnosis.

**Advice:**
• Consult a neurologist immediately  
• Begin early treatment planning  
• Arrange caregiver awareness  
• Consider follow-up neuroimaging  
"""


# -------------------------------------------------
# GRADCAM
# -------------------------------------------------
def compute_gradcam(model, img_tensor, layer_name="block5_conv3"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


def overlay_heatmap(original_img, heatmap):

    original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return overlay


# -------------------------------------------------
# CONFIDENCE GAUGE
# -------------------------------------------------
def risk_meter(probability):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Model Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00c6ff"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"},
            ],
        }
    ))

    fig.update_layout(height=250)
    return fig


# -------------------------------------------------
# PDF GENERATOR
# -------------------------------------------------
def generate_pdf(prediction, probs, advice, image):

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_pdf.name, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height-40, "AI Neurological Diagnostic Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, height-60, f"Generated: {datetime.now()}")

    c.line(50, height-70, 550, height-70)

    # Save MRI temporarily
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_img.name)

    c.drawImage(temp_img.name, 50, height-300, width=200, height=200)

    # Prediction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(300, height-100, "Prediction:")
    c.setFont("Helvetica", 12)
    c.drawString(300, height-120, prediction)

    # Probabilities
    y = height-160
    c.setFont("Helvetica-Bold", 12)
    c.drawString(300, y, "Class Probabilities:")

    y -= 20
    c.setFont("Helvetica", 11)

    for cls, prob in zip(CLASS_NAMES, probs):
        c.drawString(300, y, f"{cls}: {prob:.2f}")
        y -= 18

    # Advice
    y = height-330
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Clinical Advice:")

    y -= 20
    c.setFont("Helvetica", 11)

    for line in advice.split("\n"):
        if line.strip():
            c.drawString(60, y, line.strip())
            y -= 15

    c.save()

    return temp_pdf.name


# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload Brain MRI", type=["png","jpg","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, width=260, caption="Uploaded MRI")

    if st.button("🔬 Run AI Diagnosis"):

        img_tensor = preprocess_image(image)

        pred_class, probs = ensemble_predict(img_tensor)

        prediction_text = CLASS_NAMES[pred_class]
        confidence = probs[pred_class]

        conf_label, note = confidence_level(confidence)

        st.divider()

        col1, col2, col3 = st.columns(3)

        # MRI
        with col1:
            st.subheader("MRI Scan")
            st.image(image, use_container_width=True)

        # GradCAM
        heatmap = compute_gradcam(vgg16_model, img_tensor)
        overlay = overlay_heatmap(
            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
            heatmap
        )

        with col2:
            st.subheader("AI Attention Map")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                     use_container_width=True)

        # Gauge + probabilities stacked
        with col3:

            st.plotly_chart(
                risk_meter(confidence),
                use_container_width=True
            )

            prob_fig = go.Figure(data=[
                go.Bar(
                    x=CLASS_NAMES,
                    y=probs
                )
            ])

            prob_fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=30, b=10),
                title="Class Probabilities"
            )

            st.plotly_chart(prob_fig, use_container_width=True)

        st.divider()

        st.header(f"Diagnostic Outcome: {prediction_text}")

        st.metric("Confidence Level", conf_label)
        st.info(note)

        advice = medical_advice(pred_class)
        st.markdown(advice)

        st.warning(
            "This AI system supports clinical decision-making and does NOT replace professional diagnosis."
        )

        pdf_path = generate_pdf(
            prediction_text,
            probs,
            advice,
            image
        )

        with open(pdf_path, "rb") as file:
            st.download_button(
                "Download Neurological Report",
                file,
                file_name="AI_Diagnostic_Report.pdf",
                mime="application/pdf"
            )
