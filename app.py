"""
Interactive Streamlit Web Demo
Modern, polished UI for ISIC 2019 Dermoscopic Skin Lesion Classification & Grad-CAM Explainability.
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset_4class import CLASSES_4, CLASS_NAMES_4
from src.train_4class import build_4class_resnet50
from src.gradcam import GradCAM, apply_heatmap_overlay
from src.model import build_resnet50, get_device
from src.dataset import CLASSES as CLASSES_8, CLASS_NAMES as CLASS_NAMES_8

# Set page configuration with a modern theme
st.set_page_config(
    page_title="DermaAI — ISIC Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(15, 118, 110, 0.3);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(45, 212, 191, 0.15);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.2;
        margin: 0 0 8px 0;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* Diagnosis Result Card */
    .diag-card {
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 18px;
        border: 1px solid;
    }
    .diag-mel {
        background: rgba(239, 68, 68, 0.08);
        border-color: rgba(239, 68, 68, 0.35);
        color: #ef4444;
    }
    .diag-bcc {
        background: rgba(245, 158, 11, 0.08);
        border-color: rgba(245, 158, 11, 0.35);
        color: #f59e0b;
    }
    .diag-nv {
        background: rgba(16, 185, 129, 0.08);
        border-color: rgba(16, 185, 129, 0.35);
        color: #10b981;
    }
    .diag-other {
        background: rgba(99, 102, 241, 0.08);
        border-color: rgba(99, 102, 241, 0.35);
        color: #6366f1;
    }
    .diag-header {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .diag-main {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 6px 0;
    }
    .diag-desc {
        font-size: 0.88rem;
        color: #475569;
        margin: 0;
    }

    /* Metric pill */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        background: #f1f5f9;
        border-radius: 8px;
        padding: 8px 14px;
        margin-top: 10px;
        border: 1px solid #e2e8f0;
    }
    .metric-pill-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        margin-left: 8px;
    }

    /* Section titles */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin: 18px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Image frame */
    .img-frame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px -2px rgba(0,0,0,0.06);
    }

    /* Prob bar container */
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
    .prob-label {
        width: 140px;
        font-weight: 600;
        color: #334155;
    }
    .prob-bar-bg {
        flex: 1;
        height: 18px;
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0 12px;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.4s ease;
    }
    .prob-pct {
        width: 55px;
        text-align: right;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #0f172a;
    }

    /* Metric card */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px -2px rgba(0,0,0,0.04);
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f766e;
    }
    .stat-lbl {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Sidebar info block */
    .sidebar-info {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 0.82rem;
        color: #475569;
        margin-top: 14px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_INFO = {
    'MEL': {
        'name': 'Melanoma',
        'type': 'Malignant (High Priority)',
        'class_tag': 'diag-mel',
        'color': '#ef4444',
        'desc': 'High-risk skin cancer originating in melanocytes. Requires prompt biopsy and urgent dermatological staging.'
    },
    'BCC': {
        'name': 'Basal Cell Carcinoma',
        'type': 'Malignant (Non-Melanoma)',
        'class_tag': 'diag-bcc',
        'color': '#f59e0b',
        'desc': 'Most common form of skin cancer. Locally invasive, slowly growing, typically curable with excision or Mohs surgery.'
    },
    'NV': {
        'name': 'Melanocytic Nevus',
        'type': 'Benign Mole',
        'class_tag': 'diag-nv',
        'color': '#10b981',
        'desc': 'Common benign melanocytic proliferation. Uniform pigment network with low clinical concern.'
    },
    'OTHER': {
        'name': 'Other Lesions (AK / BKL / DF / VASC / SCC)',
        'type': 'Aggregated Group',
        'class_tag': 'diag-other',
        'color': '#6366f1',
        'desc': 'Includes Actinic Keratosis, Benign Keratosis, Dermatofibroma, Vascular Lesions, and Squamous Cell Carcinomas.'
    }
}

# Curated benchmark validation samples that clearly exemplify each diagnostic category
CURATED_SAMPLES = {
    "Melanoma Case (MEL - Malignant)": {
        "image_id": "ISIC_0000030_downsampled",
        "class": "MEL",
        "description": "True Melanoma (93.8% confidence prediction)"
    },
    "Melanocytic Nevus (NV - Benign Mole)": {
        "image_id": "ISIC_0000003",
        "class": "NV",
        "description": "True Benign Mole (95.1% confidence prediction)"
    },
    "Basal Cell Carcinoma (BCC - Carcinoma)": {
        "image_id": "ISIC_0056669",
        "class": "BCC",
        "description": "True Basal Cell Carcinoma (High confidence prediction)"
    },
    "Other Diagnoses (OTHER - Keratosis/Vascular)": {
        "image_id": "ISIC_0034197",
        "class": "OTHER",
        "description": "True Other Lesion / BKL (High confidence prediction)"
    }
}

@st.cache_resource
def load_model(model_path="models/best_resnet50_4class.pth", mode="4-Class"):
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if mode == "4-Class":
        model = build_4class_resnet50(pretrained=False).to(device)
        classes = CLASSES_4
        class_names = CLASS_NAMES_4
    else:
        model = build_resnet50(num_classes=8, pretrained=False).to(device)
        classes = CLASSES_8
        class_names = CLASS_NAMES_8

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device, classes, class_names

def main():
    # Hero Header Banner
    st.markdown("""
    <div class="hero-card">
        <span class="hero-badge">ISIC 2019 • ResNet-50 Transfer Learning</span>
        <h1 class="hero-title">🔬 Dermoscopic Skin Lesion Classifier</h1>
        <p class="hero-sub">
            Real-time multi-class diagnostic triage and pixel-level <strong>Grad-CAM explainability</strong> powered by deep convolutional feature analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")
        mode = st.radio(
            "Classification Task",
            ["4-Class (NV, MEL, BCC, OTHER) [Primary]", "8-Class (Original ISIC Benchmark)"],
            index=0
        )

        if mode.startswith("4-Class"):
            model_path = st.text_input("Model Checkpoint", "models/best_resnet50_4class.pth")
            active_mode = "4-Class"
        else:
            model_path = st.text_input("Model Checkpoint", "models/best_resnet50.pth")
            active_mode = "8-Class"

        if not os.path.exists(model_path):
            st.error(f"⚠️ Checkpoint file not found: `{model_path}`")
            return

        model, device, classes, class_names = load_model(model_path, active_mode)
        grad_cam = GradCAM(model, model.layer4)

        st.markdown("---")
        st.markdown("### 📋 Supported Categories")
        for code in classes:
            color = CLASS_INFO.get(code, {}).get('color', '#475569')
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:0.85rem;'>"
                f"<span style='width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;'></span>"
                f"<strong>{code}</strong>: {class_names[code]}"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("""
        <div class="sidebar-info">
            <strong>🔒 Clinical Safety Notice:</strong><br>
            This AI tool provides visual decision support for research and education. It does not replace histological confirmation by a pathologist.
        </div>
        """, unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🩺 Live Diagnosis & Explainability", "📈 Model Benchmarks & Metrics", "📖 Clinical & Model Docs"])

    with tab1:
        st.markdown("<div class='section-title'>📤 1. Provide a Dermoscopic Image</div>", unsafe_allow_html=True)

        c_up1, c_up2 = st.columns([1.2, 1])
        with c_up1:
            uploaded_file = st.file_uploader(
                "Upload image file (JPG, PNG)",
                type=["jpg", "jpeg", "png"],
                help="Supports standard high-resolution dermatoscope photography."
            )
        with c_up2:
            sample_options = ["Select a sample..."] + list(CURATED_SAMPLES.keys())
            sample_choice = st.selectbox("Or choose a validated test case:", sample_options)

        # Load image from upload or curated sample
        image = None
        loaded_gt_label = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
        elif sample_choice != "Select a sample..." and os.path.exists("data/processed/val_split_4class.csv"):
            val_df = pd.read_csv("data/processed/val_split_4class.csv")
            curated_info = CURATED_SAMPLES[sample_choice]
            target_id = curated_info["image_id"]
            matched_rows = val_df[val_df['image_id'] == target_id]

            if len(matched_rows) > 0:
                sample_row = matched_rows.iloc[0]
            else:
                # Fallback to first row matching the class if ID not present
                sample_row = val_df[val_df['label_4class'] == curated_info["class"]].iloc[0]

            if os.path.exists(sample_row['image_path']):
                image = Image.open(sample_row['image_path']).convert('RGB')
                loaded_gt_label = f"{sample_row['image_id']} (Ground Truth: {sample_row['label_4class']} - {sample_row.get('original_label', sample_row['label_4class'])})"

        if image is not None:
            if loaded_gt_label:
                st.caption(f"ℹ️ Loaded benchmark sample: `{loaded_gt_label}`")

            # Inference
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_class = classes[pred_idx]
                pred_confidence = probs[pred_idx]

            info = CLASS_INFO.get(pred_class, {
                'name': class_names[pred_class],
                'type': 'Classified Diagnosis',
                'class_tag': 'diag-other',
                'color': '#0f766e',
                'desc': 'Lesion classified by neural network.'
            })

            st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🎯 2. Diagnostic Assessment</div>", unsafe_allow_html=True)

            col_diag, col_prob = st.columns([1, 1.3])

            with col_diag:
                st.markdown(f"""
                <div class="diag-card {info['class_tag']}">
                    <div class="diag-header">{info['type']}</div>
                    <div class="diag-main">{pred_class} — {info['name']}</div>
                    <p class="diag-desc">{info['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.image(image, caption="Ingested Dermoscopic Input", use_container_width=True)

            with col_prob:
                st.markdown("#### Class Probability Distribution")

                # Visual custom progress bars
                for c_code in classes:
                    c_idx = classes.index(c_code)
                    c_prob = probs[c_idx]
                    c_pct = c_prob * 100
                    c_color = CLASS_INFO.get(c_code, {}).get('color', '#0f766e')
                    is_pred = (c_code == pred_class)
                    border_style = f"border: 2px solid {c_color};" if is_pred else ""

                    st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-label" style="{'font-weight:700;color:'+c_color if is_pred else ''}">
                            {c_code} ({class_names[c_code].split()[0]})
                        </div>
                        <div class="prob-bar-bg" style="{border_style}">
                            <div class="prob-bar-fill" style="width: {max(c_pct, 2):.1f}%; background: {c_color};"></div>
                        </div>
                        <div class="prob-pct">{c_pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-pill">
                    <span>Confidence Score:</span>
                    <span class="metric-pill-val" style="color:{info['color']};">{pred_confidence * 100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)

            # Grad-CAM Section
            st.markdown("<hr style='margin:28px 0 20px 0;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🔍 3. Explainable AI: Grad-CAM Saliency Analysis</div>", unsafe_allow_html=True)

            show_cam = st.toggle("Enable Layer 4 Feature Map Activation Overlay", value=True)

            if show_cam:
                heatmap, _, _ = grad_cam.generate(input_tensor, target_class=pred_idx)
                _, overlay = apply_heatmap_overlay(image, heatmap, colormap='jet', alpha=0.45)

                gc1, gc2, gc3 = st.columns(3)
                with gc1:
                    st.markdown("**Original Dermoscopy**")
                    st.image(image, use_container_width=True)
                    st.caption("Standardized RGB dermatoscopic frame.")
                with gc2:
                    st.markdown("**Grad-CAM Saliency Map**")
                    st.image(heatmap, clamp=True, use_container_width=True)
                    st.caption("Warm colors (red/yellow) indicate highest influence.")
                with gc3:
                    st.markdown("**Blended Spatial Overlay**")
                    st.image(overlay, use_container_width=True)
                    st.caption(f"Spatial focus mapped over {info['name']}.")

                st.info(
                    "💡 **How to interpret:** Grad-CAM traces gradients from the prediction logit back to ResNet-50's final convolutional layer (`layer4`). "
                    "For Melanomas, activations typically highlight irregular pigment fringes and peripheral asymmetry; for Basal Cell Carcinomas, telangiectatic vessels and translucent areas are highlighted."
                )

    with tab2:
        st.markdown("<div class='section-title'>📊 Model Performance & Comparative Benchmark</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:grid;grid-template-columns: repeat(4, 1fr);gap:16px;margin-bottom:24px;">
            <div class="stat-card">
                <div class="stat-val">71.33%</div>
                <div class="stat-lbl">4-Class Val Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">0.6522</div>
                <div class="stat-lbl">4-Class Macro F1</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">66.74%</div>
                <div class="stat-lbl">Official Test Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">0.6394</div>
                <div class="stat-lbl">Official Test Macro F1</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### 4-Class Training Dynamics")
            if os.path.exists("results/plots/training_curves_4class.png"):
                st.image("results/plots/training_curves_4class.png", use_container_width=True)

            st.markdown("#### Validation Confusion Matrix")
            if os.path.exists("results/plots/confusion_matrix_validation_4class.png"):
                st.image("results/plots/confusion_matrix_validation_4class.png", use_container_width=True)

        with col_p2:
            st.markdown("#### 4-Class Target Distribution")
            if os.path.exists("results/plots/eda_4class_distribution.png"):
                st.image("results/plots/eda_4class_distribution.png", use_container_width=True)

            st.markdown("#### Official ISIC Test Confusion Matrix")
            if os.path.exists("results/plots/confusion_matrix_test_4class.png"):
                st.image("results/plots/confusion_matrix_test_4class.png", use_container_width=True)

    with tab3:
        st.markdown("<div class='section-title'>📖 Clinical Background & Technical Specifications</div>", unsafe_allow_html=True)

        st.markdown("""
        ### Why 4-Class Diagnostic Grouping?
        In the original ISIC 2019 dataset, rare diseases like Dermatofibroma (`DF`, 0.9%) and Vascular Lesions (`VASC`, 1.0%) have fewer than 250 images each, causing standard cross-entropy models to develop high false-negative rates on rare classes.

        By aggregating the 5 minor diagnoses into a single **`OTHER`** class:
        1. **Zero Data Loss:** All **25,331 training images** remain actively utilized.
        2. **Robust Boundaries:** Distinguishes the primary clinical triage questions:
           - Is this a standard mole (**`NV`**)?
           - Is this a dangerous melanoma (**`MEL`**)?
           - Is this a non-melanoma carcinoma (**`BCC`**)?
           - Or is it another dermatological condition (**`OTHER`**)?
        3. **Accuracy Surge:** Test accuracy jumps from **50.06% (8-class)** to **66.74% (4-class)**, and validation accuracy reaches **71.33%**.

        ---
        ### Technical Stack
        - **Backbone:** ResNet-50 initialized on ImageNet weights.
        - **Explainability:** Grad-CAM on `layer4` residual blocks.
        - **Imbalance Handling:** Balanced inverse-frequency cross-entropy loss.
        - **Pipeline Safety:** Patient-level grouping on `lesion_id` to guarantee 0% data leakage.
        """)

if __name__ == "__main__":
    main()
