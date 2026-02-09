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
    page_title="AI Alzheimer's Diagnostic System",
    layout="wide"
)

st.title("🧠 AI Alzheimer's Diagnostic System")
st.caption("VGG16 + Xception Weighted Ensemble | 78.17% Accuracy | 100% AD Sensitivity")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    """Load VGG16 and Xception models from SavedModel format"""
    try:
        # Load VGG16 from vgg16_new_savedmodel folder
        vgg = tf.keras.models.load_model(
            "models/vgg16_new_savedmodel",
            compile=False
        )
        st.sidebar.success("✅ VGG16 loaded")
    except Exception as e:
        st.sidebar.error(f"❌ VGG16 load failed: {e}")
        return None, None
    
    try:
        # Load Xception from xception_model_savedmodel folder
        xception = tf.keras.models.load_model(
            "models/xception_model_savedmodel",
            compile=False
        )
        st.sidebar.success("✅ Xception loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Xception load failed: {e}")
        return None, None
    
    return vgg, xception


# Load models
vgg16_model, xception_model = load_models()

# Check if models loaded successfully
if vgg16_model is None or xception_model is None:
    st.error("⚠️ Failed to load models. Please check model paths and try again.")
    st.stop()

# Class names and configuration
CLASS_NAMES = [
    "Cognitively Normal (CN)",
    "Mild Cognitive Impairment (MCI)",
    "Alzheimer's Disease (AD)"
]

CLASS_NAMES_SHORT = ["CN", "MCI", "AD"]

IMG_SIZE = 224

# Ensemble weights (from your validation results)
VGG_WEIGHT = 0.477
XCEPTION_WEIGHT = 0.523

# -------------------------------------------------
# PREPROCESS
# -------------------------------------------------
def preprocess_image(image):
    """
    Preprocess MRI image for model input
    Matches training preprocessing
    """
    # Convert to grayscale
    image = np.array(image.convert("L"))
    
    # Resize to 224x224
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Apply Gaussian blur (denoising)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Normalize to [0, 1]
    image = image / 255.0
    
    # Convert to 3-channel (RGB) for pretrained models
    image = np.stack([image] * 3, axis=-1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


# -------------------------------------------------
# WEIGHTED ENSEMBLE PREDICTION
# -------------------------------------------------
def ensemble_predict(img_tensor):
    """
    Weighted ensemble: 47.7% VGG16 + 52.3% Xception
    Achieves 78.17% accuracy, 100% AD recall, 50% AD precision
    """
    # Get predictions from both models
    pred_vgg = vgg16_model.predict(img_tensor, verbose=0)
    pred_xception = xception_model.predict(img_tensor, verbose=0)
    
    # Weighted ensemble (based on validation performance)
    ensemble_prob = (VGG_WEIGHT * pred_vgg) + (XCEPTION_WEIGHT * pred_xception)
    
    # Get predicted class
    pred_class = np.argmax(ensemble_prob)
    
    return pred_class, ensemble_prob[0], pred_vgg[0], pred_xception[0]


# -------------------------------------------------
# CONFIDENCE INTERPRETATION
# -------------------------------------------------
def confidence_level(conf):
    """Interpret model confidence level"""
    if conf < 0.50:
        return "Low Confidence", "⚠️ Recommend additional clinical testing and expert review."
    elif conf < 0.75:
        return "Moderate Confidence", "⚡ Clinical correlation advised. Consider follow-up imaging."
    else:
        return "High Confidence", "✅ Strong imaging biomarkers detected. Consult specialist."


# -------------------------------------------------
# MEDICAL ADVICE
# -------------------------------------------------
def medical_advice(pred_class, confidence):
    """Generate medical advice based on prediction"""
    
    if pred_class == 0:  # Cognitively Normal
        return f"""
### 🟢 Result: Cognitively Normal (CN)
**Confidence:** {confidence*100:.1f}%

**Interpretation:**
No significant neurodegenerative patterns detected in the MRI scan.

**Recommendations:**
- ✅ Continue routine health monitoring
- 🏃 Maintain regular physical exercise (150 min/week)
- 🧠 Stay cognitively active (reading, puzzles, learning)
- 🥗 Follow Mediterranean or DASH diet
- 😴 Ensure 7-8 hours quality sleep
- 📅 Schedule follow-up in 1-2 years if age 65+

**Prevention:**
- Manage cardiovascular risk factors (BP, cholesterol, diabetes)
- Stay socially engaged
- Avoid smoking and excessive alcohol
"""
    
    elif pred_class == 1:  # MCI
        return f"""
### 🟡 Result: Mild Cognitive Impairment (MCI)
**Confidence:** {confidence*100:.1f}%

**Interpretation:**
Imaging findings suggest Mild Cognitive Impairment. MCI represents an intermediate stage between normal aging and dementia.

**Important:**
- ~15% of MCI patients progress to Alzheimer's per year
- Early intervention may help delay progression
- Not all MCI cases progress to dementia

**Immediate Actions:**
- 👨‍⚕️ **Consult neurologist** for comprehensive evaluation
- 🧪 Perform cognitive screening (MMSE, MoCA)
- 🔬 Consider biomarker testing (CSF, PET scan)
- 💊 Discuss treatment options (cholinesterase inhibitors)

**Lifestyle Interventions:**
- Intensive cognitive training
- Aerobic exercise (30+ min daily)
- Social engagement
- Stress management
- Monitor progression every 6 months
"""
    
    else:  # Alzheimer's Disease
        return f"""
### 🔴 Result: Alzheimer's Disease (AD)
**Confidence:** {confidence*100:.1f}%

⚠️ **This AI system provides decision support and is NOT a definitive diagnosis.**

**Interpretation:**
Imaging patterns strongly associated with Alzheimer's Disease detected. The ensemble model has **100% sensitivity** for AD detection.

**Critical Next Steps:**
- 🚨 **Urgent neurologist consultation** required
- 🏥 Comprehensive clinical assessment needed
- 🧬 Consider genetic counseling (APOE4 testing)
- 💉 Discuss FDA-approved treatments:
  - Cholinesterase inhibitors (Donepezil, Rivastigmine)
  - NMDA antagonist (Memantine)
  - Anti-amyloid antibodies (Lecanemab, Donanemab)

**Care Planning:**
- 👨‍👩‍👧‍👦 Involve family/caregivers in care decisions
- 📋 Legal planning (advance directives, power of attorney)
- 🏠 Home safety assessment
- 💰 Financial planning for long-term care

**Support Resources:**
- Alzheimer's Association: 800-272-3900
- Clinical trials: ClinicalTrials.gov
- Support groups for patients and caregivers

**Prognosis:**
Early detection allows for:
- Better treatment response
- Clinical trial eligibility
- Advanced care planning
- Quality of life optimization
"""


# -------------------------------------------------
# GRAD-CAM (Explainable AI)
# -------------------------------------------------
def compute_gradcam(model, img_tensor, layer_name=None):
    """
    Compute Grad-CAM heatmap for explainability
    Shows which brain regions influenced the prediction
    """
    # Auto-detect last conv layer if not specified
    if layer_name is None:
        # Try VGG16 layer first
        try:
            layer_name = "block5_conv3"  # VGG16
            model.get_layer(layer_name)
        except:
            # Find last conv layer for other architectures
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
    
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= (tf.reduce_max(heatmap) + 1e-8)

        
        return heatmap.numpy()
    
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None


def overlay_heatmap(original_img, heatmap):
    """Overlay Grad-CAM heatmap on original image"""
    
    if heatmap is None:
        return original_img
    
    # Resize original image
    original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
    
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    return overlay


# -------------------------------------------------
# CONFIDENCE GAUGE
# -------------------------------------------------
def risk_meter(probability, pred_class):
    """Interactive confidence gauge"""
    
    # Color based on prediction
    colors = {
        0: "#4CAF50",  # Green for CN
        1: "#FF9800",  # Orange for MCI
        2: "#F44336"   # Red for AD
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Model Confidence", 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': colors[pred_class]},
            'steps': [
                {'range': [0, 50], 'color': "#E8F5E9"},
                {'range': [50, 75], 'color': "#FFF3E0"},
                {'range': [75, 100], 'color': "#FFEBEE"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# -------------------------------------------------
# PROBABILITY BAR CHART
# -------------------------------------------------
def probability_chart(probs, vgg_probs, xception_probs):
    """Visualize ensemble and individual model probabilities"""
    
    fig = go.Figure()
    
    # Ensemble probabilities
    fig.add_trace(go.Bar(
        name='Ensemble (Weighted)',
        x=CLASS_NAMES_SHORT,
        y=probs,
        marker_color=['#4CAF50', '#FF9800', '#F44336'],
        text=[f'{p:.1%}' for p in probs],
        textposition='auto',
    ))
    
    # VGG16 probabilities
    fig.add_trace(go.Bar(
        name=f'VGG16 ({VGG_WEIGHT:.1%})',
        x=CLASS_NAMES_SHORT,
        y=vgg_probs,
        marker_color='lightblue',
        text=[f'{p:.1%}' for p in vgg_probs],
        textposition='auto',
        opacity=0.6
    ))
    
    # Xception probabilities
    fig.add_trace(go.Bar(
        name=f'Xception ({XCEPTION_WEIGHT:.1%})',
        x=CLASS_NAMES_SHORT,
        y=xception_probs,
        marker_color='lightcoral',
        text=[f'{p:.1%}' for p in xception_probs],
        textposition='auto',
        opacity=0.6
    ))
    
    fig.update_layout(
        title="Class Probabilities (Ensemble & Individual Models)",
        xaxis_title="Diagnosis",
        yaxis_title="Probability",
        barmode='group',
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=80, b=10)
    )
    
    return fig


# -------------------------------------------------
# PDF REPORT GENERATOR
# -------------------------------------------------
def generate_pdf(prediction, probs, confidence, advice, image):
    """Generate comprehensive PDF diagnostic report"""
    
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_pdf.name, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "AI Alzheimer's Diagnostic Report")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 85, "Weighted Ensemble Model | 78.17% Validation Accuracy")
    
    c.line(50, height - 95, width - 50, height - 95)
    
    # Save MRI image temporarily
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_img.name)
    
    # Draw MRI image
    c.drawImage(temp_img.name, 50, height - 320, width=200, height=200)
    
    # Prediction box
    c.setFont("Helvetica-Bold", 16)
    c.drawString(280, height - 120, "Diagnostic Result:")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(280, height - 145, prediction)
    
    # Confidence
    c.setFont("Helvetica", 12)
    c.drawString(280, height - 170, f"Confidence: {confidence*100:.1f}%")
    
    # Probabilities
    y = height - 210
    c.setFont("Helvetica-Bold", 12)
    c.drawString(280, y, "Class Probabilities:")
    
    y -= 25
    c.setFont("Helvetica", 11)
    for cls, prob in zip(CLASS_NAMES, probs):
        c.drawString(290, y, f"{cls}: {prob*100:.1f}%")
        y -= 18
    
    # Model info
    y = height - 340
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Model Architecture:")
    c.setFont("Helvetica", 10)
    y -= 18
    c.drawString(60, y, f"• VGG16 Weight: {VGG_WEIGHT:.1%}")
    y -= 15
    c.drawString(60, y, f"• Xception Weight: {XCEPTION_WEIGHT:.1%}")
    y -= 15
    c.drawString(60, y, "• Validation Accuracy: 78.17%")
    y -= 15
    c.drawString(60, y, "• AD Sensitivity: 100%")
    y -= 15
    c.drawString(60, y, "• AD Precision: 50%")
    
    # Clinical advice (new page)
    c.showPage()
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Clinical Recommendations")
    
    c.line(50, height - 60, width - 50, height - 60)
    
    y = height - 90
    c.setFont("Helvetica", 11)
    
    # Parse and write advice
    for line in advice.split("\n"):
        line = line.strip()
        if not line:
            y -= 10
            continue
        
        # Handle different formatting
        if line.startswith("###"):
            c.setFont("Helvetica-Bold", 13)
            line = line.replace("###", "").replace("🟢", "").replace("🟡", "").replace("🔴", "").strip()
        elif line.startswith("**") and line.endswith("**"):
            c.setFont("Helvetica-Bold", 11)
            line = line.replace("**", "")
        else:
            c.setFont("Helvetica", 10)
            line = line.replace("•", "  •").replace("✅", "").replace("🚨", "").replace("⚠️", "")
        
        # Word wrap if needed
        if len(line) > 85:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line + word) < 85:
                    current_line += word + " "
                else:
                    c.drawString(60, y, current_line.strip())
                    y -= 15
                    current_line = word + " "
                    if y < 50:
                        c.showPage()
                        y = height - 50
            if current_line:
                c.drawString(60, y, current_line.strip())
                y -= 15
        else:
            c.drawString(60, y, line)
            y -= 15
        
        if y < 50:
            c.showPage()
            y = height - 50
    
    # Disclaimer
    y -= 20
    if y < 150:
        c.showPage()
        y = height - 50
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "IMPORTANT DISCLAIMER:")
    y -= 20
    c.setFont("Helvetica", 9)
    disclaimer = [
        "This AI diagnostic system is intended for clinical decision support only.",
        "It does NOT replace professional medical diagnosis or clinical judgment.",
        "All results must be confirmed by qualified healthcare professionals.",
        "The model was validated on a specific dataset and may not generalize to all populations.",
        "Consult a licensed neurologist or physician for definitive diagnosis and treatment planning."
    ]
    for line in disclaimer:
        c.drawString(60, y, line)
        y -= 14
    
    c.save()
    
    return temp_pdf.name


# -------------------------------------------------
# MAIN UI
# -------------------------------------------------

# Sidebar information
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.metric("Validation Accuracy", "78.17%")
    st.metric("AD Sensitivity", "100%")
    st.metric("AD Precision", "50%")
    
    st.divider()
    
    st.subheader("📊 Ensemble Configuration")
    st.write(f"**VGG16:** {VGG_WEIGHT:.1%}")
    st.write(f"**Xception:** {XCEPTION_WEIGHT:.1%}")
    
    st.divider()
    
    st.subheader("🎯 Performance Metrics")
    st.write("**CN Recall:** 77.3%")
    st.write("**MCI Recall:** 78.3%")
    st.write("**AD Recall:** 100%")
    
    st.divider()
    
    st.info("""
    **How to use:**
    1. Upload brain MRI scan
    2. Click 'Run AI Diagnosis'
    3. Review results & advice
    4. Download PDF report
    """)

# Main content
st.markdown("---")
st.subheader("📤 Upload MRI Scan")

uploaded_file = st.file_uploader(
    "Choose a brain MRI image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    
    image = Image.open(uploaded_file)
    
    col_upload1, col_upload2 = st.columns([1, 2])
    
    with col_upload1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)
    
    with col_upload2:
        st.info("""
        **MRI Requirements:**
        - Brain scan (axial, sagittal, or coronal view)
        - Grayscale or RGB format
        - Clear visualization of brain structures
        - Minimal artifacts or noise
        
        **Best Results:**
        T1-weighted MRI scans with good contrast between gray/white matter.
        """)
    
    st.markdown("---")
    
    if st.button("🔬 Run AI Diagnosis", type="primary", use_container_width=True):
        
        with st.spinner("🧠 Analyzing MRI scan with ensemble model..."):
            
            # Preprocess image
            img_tensor = preprocess_image(image)
            
            # Get ensemble prediction
            pred_class, ensemble_probs, vgg_probs, xception_probs = ensemble_predict(img_tensor)
            
            prediction_text = CLASS_NAMES[pred_class]
            confidence = ensemble_probs[pred_class]
            
            conf_label, conf_note = confidence_level(confidence)
        
        st.success("✅ Analysis Complete!")
        
        st.markdown("---")
        st.header("📊 Diagnostic Results")
        
        # Create 3-column layout
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Column 1: Original MRI
        with col1:
            st.subheader("🖼️ Original MRI")
            st.image(image, use_container_width=True, caption="Input Scan")
        
        # Column 2: Grad-CAM Visualization
        with col2:
            st.subheader("🔍 AI Attention Map (Grad-CAM)")
            
            with st.spinner("Generating explainability heatmap..."):
                heatmap = compute_gradcam(vgg16_model, img_tensor)
                
                if heatmap is not None:
                    overlay = overlay_heatmap(
                        cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
                        heatmap
                    )
                    st.image(
                        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                        caption="Regions influencing prediction"
                    )
                else:
                    st.warning("Grad-CAM visualization unavailable")
        
        # Column 3: Confidence & Probabilities
        with col3:
            st.subheader("📈 Model Confidence")
            st.plotly_chart(
                risk_meter(confidence, pred_class),
                use_container_width=True
            )
        
        # Full-width probability chart
        st.markdown("---")
        st.plotly_chart(
            probability_chart(ensemble_probs, vgg_probs, xception_probs),
            use_container_width=True
        )
        
        # Diagnostic Outcome
        st.markdown("---")
        st.header(f"🎯 Diagnostic Outcome: **{prediction_text}**")
        
        # Confidence display
        col_conf1, col_conf2 = st.columns(2)
        with col_conf1:
            st.metric("Confidence Level", conf_label, f"{confidence*100:.1f}%")
        with col_conf2:
            if pred_class == 0:
                st.success(conf_note)
            elif pred_class == 1:
                st.warning(conf_note)
            else:
                st.error(conf_note)
        
        # Medical Advice
        st.markdown("---")
        st.subheader("💡 Clinical Recommendations")
        
        advice = medical_advice(pred_class, confidence)
        st.markdown(advice)
        
        # Important disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **IMPORTANT MEDICAL DISCLAIMER:**
        
        This AI system provides **clinical decision support** and is NOT a substitute for professional medical diagnosis.
        
        - All results must be **confirmed by a qualified healthcare professional**
        - The model achieved 78.17% accuracy on validation data
        - Individual patient outcomes may vary
        - Consult a **licensed neurologist** for definitive diagnosis and treatment
        
        **Emergency:** If experiencing severe symptoms, seek immediate medical attention.
        """)
        
        # Generate PDF Report
        st.markdown("---")
        st.subheader("📄 Download Diagnostic Report")
        
        with st.spinner("Generating PDF report..."):
            pdf_path = generate_pdf(
                prediction_text,
                ensemble_probs,
                confidence,
                advice,
                image
            )
        
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="📥 Download Full Diagnostic Report (PDF)",
                data=file,
                file_name=f"Alzheimer_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        
        st.success("✅ Report ready for download!")

else:
    # Welcome screen when no file uploaded
    st.info("""
    ### 👋 Welcome to the AI Alzheimer's Diagnostic System
    
    This system uses a **weighted ensemble** of deep learning models (VGG16 + Xception) to analyze brain MRI scans 
    and assist in the diagnosis of Alzheimer's Disease and Mild Cognitive Impairment.
    
    **Model Performance:**
    - ✅ 78.17% overall accuracy
    - ✅ 100% Alzheimer's Disease sensitivity (no missed cases)
    - ✅ 50% AD precision (reduced false positives)
    
    **Get Started:**
    Upload a brain MRI scan using the file uploader above to begin analysis.
    """)
    
    # Example images or instructions
    st.markdown("---")
    st.subheader("📝 Tips for Best Results")
    
    col_tip1, col_tip2, col_tip3 = st.columns(3)
    
    with col_tip1:
        st.markdown("""
        **Image Quality**
        - High resolution preferred
        - Clear brain structures
        - Minimal noise/artifacts
        """)
    
    with col_tip2:
        st.markdown("""
        **Scan Type**
        - T1-weighted MRI ideal
        - Axial, sagittal, or coronal
        - Grayscale or RGB accepted
        """)
    
    with col_tip3:
        st.markdown("""
        **Output**
        - Detailed prediction
        - Explainable AI heatmap
        - Downloadable PDF report
        """)

# Footer
st.markdown("---")
st.caption("""
**Developed for Clinical Decision Support | Not FDA Approved**  
Model: VGG16 + Xception Weighted Ensemble (47.7% / 52.3%)  
Performance: 78.17% Accuracy | 100% AD Sensitivity | Research Use Only
""")
