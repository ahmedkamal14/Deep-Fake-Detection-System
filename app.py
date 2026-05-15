import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from timm import create_model
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
import numpy as np
import plotly.graph_objects as go


# ========================================
# MODEL DEFINITION
# ========================================

class ClassifierMixin(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = BinaryAccuracy()
        self.auroc = BinaryAUROC()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class ResNetClassifier(ClassifierMixin):
    def __init__(self, model_name='resnet50', num_classes=2, learning_rate=1e-4):
        ClassifierMixin.__init__(self) 
        self.learning_rate = learning_rate
        self.backbone = create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ========================================
# LOAD MODEL (Cached)
# ========================================

@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load('final_deepfake_detector.pth', map_location='cpu')
        model = ResNetClassifier(model_name='resnet50')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        return model, device, checkpoint.get('test_results', None)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


# ========================================
# PREPROCESSING
# ========================================

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])


# ========================================
# PREDICTION FUNCTION
# ========================================

def predict_image(image, model, device):
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return {
        'prediction': 'REAL' if pred_class == 0 else 'FAKE',
        'confidence': confidence,
        'real_prob': probs[0, 0].item(),
        'fake_prob': probs[0, 1].item()
    }


# ========================================
# STREAMLIT UI
# ========================================

# Page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .real-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .fake-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔍 Deepfake Detection System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.write("""
    This application uses a fine-tuned ResNet50 model to detect 
    whether a face image is real or AI-generated/manipulated.
    """)
    
    st.header("📊 Model Info")
    model, device, test_results = load_model()
    
    if model is not None:
        st.success("✅ Model loaded successfully")
        st.info(f"🖥️ Device: {device.upper()}")
        
        if test_results:
            st.header("🎯 Model Performance")
            if 'test_acc' in test_results:
                st.metric("Test Accuracy", f"{test_results['test_acc']:.2%}")
            if 'test_auroc' in test_results:
                st.metric("Test AUROC", f"{test_results['test_auroc']:.4f}")
            if 'test_loss' in test_results:
                st.metric("Test Loss", f"{test_results['test_loss']:.4f}")
    else:
        st.error("❌ Failed to load model")
    
    st.header("ℹ️ How to Use")
    st.write("""
    1. Upload a face image
    2. Wait for analysis
    3. View the prediction and confidence scores
    """)
    
    st.header("⚠️ Note")
    st.warning("""
    This model is trained for educational purposes. 
    Results should be validated by human experts for critical applications.
    """)

# Main content
if model is None:
    st.error("⚠️ Model not loaded. Please ensure 'final_deepfake_detector.pth' is in the same directory.")
    st.stop()

# File uploader
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a face image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear face image for detection"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                result = predict_image(image, model, device)
                
                st.session_state['result'] = result

with col2:
    st.header("📊 Analysis Results")
    
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        if result['prediction'] == 'REAL':
            st.markdown(f"""
                <div class="result-box real-box">
                    <h1 style="color: #28a745; margin: 0;">✓ REAL</h1>
                    <h3 style="margin-top: 1rem;">Confidence: {result['confidence']:.2%}</h3>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box fake-box">
                    <h1 style="color: #dc3545; margin: 0;">⚠ FAKE</h1>
                    <h3 style="margin-top: 1rem;">Confidence: {result['confidence']:.2%}</h3>
                </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📈 Detailed Probabilities")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Real Probability",
                f"{result['real_prob']:.2%}",
                delta=None
            )
        with col_b:
            st.metric(
                "Fake Probability", 
                f"{result['fake_prob']:.2%}",
                delta=None
            )
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Real', 'Fake'],
                y=[result['real_prob'], result['fake_prob']],
                marker_color=['#28a745', '#dc3545'],
                text=[f"{result['real_prob']:.2%}", f"{result['fake_prob']:.2%}"],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Probability Distribution",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            height=300,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("💡 Interpretation")
        
        confidence = result['confidence']
        if confidence >= 0.9:
            st.success("🟢 **Very High Confidence** - The model is highly certain about this prediction.")
        elif confidence >= 0.75:
            st.info("🔵 **High Confidence** - The model is reasonably confident about this prediction.")
        elif confidence >= 0.6:
            st.warning("🟡 **Moderate Confidence** - The prediction should be verified by other methods.")
        else:
            st.error("🔴 **Low Confidence** - The model is uncertain. Manual verification recommended.")
        
        with st.expander("📋 Detailed Analysis"):
            st.write("**Classification Details:**")
            st.write(f"- Predicted Class: **{result['prediction']}**")
            st.write(f"- Confidence Score: **{result['confidence']:.4f}**")
            st.write(f"- Real Score: **{result['real_prob']:.4f}**")
            st.write(f"- Fake Score: **{result['fake_prob']:.4f}**")
            st.write(f"- Difference: **{abs(result['real_prob'] - result['fake_prob']):.4f}**")
            
    else:
        st.info("👆 Upload an image and click 'Analyze' to see results")