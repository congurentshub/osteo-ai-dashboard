import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Import model architecture
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_dual_cbam_vit_optimized import DualBranchOsteoporosisFusion

# Set Page Config
st.set_page_config(
    page_title="Osteoporosis Diagnosis & Admin Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
# Dynamic Base Directory for Cloud & Local Deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Checkpoint Path (supports both deployment and local workspace folders)
if os.path.exists(os.path.join(BASE_DIR, "checkpoints", "best_model.pth")):
    LATEST_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
else:
    LATEST_CHECKPOINT = os.path.join(BASE_DIR, "outputs_dual_cbam_vit_optimized", "checkpoints", "best_model.pth")

PATIENT_LOGS_CSV  = os.path.join(BASE_DIR, "patient_records.csv")

CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]
IMAGE_SIZE  = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dashboard Dark Glass Styling (Inspired by sample.py, strictly NO emojis)
DASHBOARD_CSS = """
<style>
    .stat-card-futuristic {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(102, 126, 234, 0.35);
        border-radius: 16px;
        padding: 22px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }
    
    .stat-card-futuristic:hover {
        transform: translateY(-6px);
        border-color: rgba(102, 126, 234, 0.7);
        box-shadow: 0 15px 45px rgba(102, 126, 234, 0.4);
    }
    
    .stat-value {
        font-size: 34px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 6px 0;
        letter-spacing: 0.5px;
    }
    
    .stat-label {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.75);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .stat-subtext {
        font-size: 11px;
        color: #10b981;
        margin-top: 4px;
        font-weight: 500;
    }

    .chart-card-futuristic {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(102, 126, 234, 0.35);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .chart-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .futuristic-table-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(102, 126, 234, 0.35);
        border-radius: 16px;
        padding: 24px;
        margin-top: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        overflow-x: auto;
    }

    .futuristic-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 6px;
    }

    .futuristic-table th {
        background: rgba(102, 126, 234, 0.25);
        padding: 12px 14px;
        text-align: left;
        color: #ffffff;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 1px;
        border: none;
    }

    .futuristic-table td {
        padding: 12px 14px;
        color: rgba(255, 255, 255, 0.9);
        border: none;
        font-size: 13px;
        background: rgba(255, 255, 255, 0.02);
    }

    .futuristic-table tr:hover td {
        background: rgba(102, 126, 234, 0.12);
    }

    .badge-normal {
        background: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-osteopenia {
        background: #f59e0b;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-osteoporosis {
        background: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
    }
</style>
"""

# CLAHE Preprocessing Setup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(img_pil: Image.Image) -> Image.Image:
    img_pil = img_pil.convert("RGB")
    if img_pil.size != (512, 512):
        img_pil = img_pil.resize((512, 512), Image.Resampling.LANCZOS)
    img_np = np.array(img_pil).astype(np.uint8)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    result_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result_rgb)

# Normalization Transform
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize Patient Records CSV File
def init_patient_logs():
    if not os.path.exists(PATIENT_LOGS_CSV):
        df_init = pd.DataFrame(columns=[
            "patient_id", "patient_name", "patient_age", "patient_gender",
            "scan_date", "diagnosis_result", "confidence_pct", "prob_normal",
            "prob_osteopenia", "prob_osteoporosis", "timestamp"
        ])
        sample_logs = [
            ["PAT-1001", "John Doe", 62, "Male", "2026-09-10", "Osteoporosis", 91.45, 0.02, 0.06, 0.91, "2026-09-10 10:30:00"],
            ["PAT-1002", "Jane Smith", 45, "Female", "2026-09-11", "Osteopenia", 84.20, 0.10, 0.84, 0.06, "2026-09-11 11:15:00"],
            ["PAT-1003", "Robert Johnson", 38, "Male", "2026-09-12", "Normal", 95.10, 0.95, 0.04, 0.01, "2026-09-12 14:00:00"],
            ["PAT-1004", "Emily Davis", 58, "Female", "2026-09-13", "Osteopenia", 78.90, 0.15, 0.79, 0.06, "2026-09-13 09:45:00"],
            ["PAT-1005", "Michael Brown", 67, "Male", "2026-09-14", "Osteoporosis", 88.75, 0.05, 0.06, 0.89, "2026-09-14 16:20:00"],
            ["PAT-1006", "Sarah Wilson", 52, "Female", "2026-09-15", "Normal", 92.30, 0.92, 0.06, 0.02, "2026-09-15 08:30:00"]
        ]
        df_sample = pd.DataFrame(sample_logs, columns=df_init.columns)
        df_sample.to_csv(PATIENT_LOGS_CSV, index=False)

init_patient_logs()

def load_patient_logs() -> pd.DataFrame:
    if os.path.exists(PATIENT_LOGS_CSV):
        return pd.read_csv(PATIENT_LOGS_CSV)
    return pd.DataFrame()

def save_patient_log(record: dict):
    df = load_patient_logs()
    df_new = pd.DataFrame([record])
    df_updated = pd.concat([df, df_new], ignore_index=True)
    df_updated.to_csv(PATIENT_LOGS_CSV, index=False)

@st.cache_resource
def load_latest_model():
    if not os.path.exists(LATEST_CHECKPOINT):
        return None, f"Model file not found at: {LATEST_CHECKPOINT}"
    try:
        model = DualBranchOsteoporosisFusion(num_classes=len(CLASS_NAMES)).to(DEVICE)
        checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Load Latest Model
model, model_error = load_latest_model()

# ==============================================================================
# GRAD-CAM HEATMAP GENERATOR
# ==============================================================================
def generate_gradcam(model, input_tensor, target_class=None):
    """
    Generates Grad-CAM activation heatmap overlay for knee X-ray radiograph.
    Target layer: model.resnet_conv[-1] (Layer 4 of ResNet-50 stem).
    """
    model.eval()
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.resnet_conv[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        logits = model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward()

    h1.remove()
    h2.remove()

    if not gradients or not feature_maps:
        return None

    grads = gradients[0].cpu().data.numpy()[0]
    f_map = feature_maps[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(f_map.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * f_map[i, :, :]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def overlay_heatmap(orig_pil: Image.Image, heatmap_np: np.ndarray, alpha: float = 0.45) -> Image.Image:
    orig_np = np.array(orig_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    overlay = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_np, alpha, 0)
    return Image.fromarray(overlay)

# ==============================================================================
# SIDEBAR LOGIN & NAVIGATION
# ==============================================================================
st.sidebar.title("OsteoAI Portal")

# Administrator Authentication
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

st.sidebar.subheader("Administrator Login")
if not st.session_state.admin_logged_in:
    admin_user = st.sidebar.text_input("Username")
    admin_pass = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login as Administrator"):
        if admin_user == "admin" and admin_pass == "admin123":
            st.session_state.admin_logged_in = True
            st.sidebar.success("Logged in as Administrator")
            st.rerun()
        else:
            st.sidebar.error("Invalid Username or Password")
else:
    st.sidebar.info("Status: Administrator Logged In")
    if st.sidebar.button("Logout"):
        st.session_state.admin_logged_in = False
        st.rerun()

st.sidebar.markdown("---")
use_clahe = st.sidebar.checkbox("Apply CLAHE Enhancement", value=True)

# Main Navigation Tabs
if st.session_state.admin_logged_in:
    nav_mode = st.radio("Navigation", ["Patient Diagnosis & Scan Input", "Administrator Analytics Dashboard", "Patient Master Records"], horizontal=True)
else:
    nav_mode = "Patient Diagnosis & Scan Input"

st.markdown("---")

# ==============================================================================
# TAB 1: PATIENT DIAGNOSIS & SCAN INPUT
# ==============================================================================
if nav_mode == "Patient Diagnosis & Scan Input":
    st.header("Patient Scan & Clinical Diagnosis Input")

    if model_error:
        st.error(model_error)
    else:
        with st.form("patient_form"):
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                p_name = st.text_input("Patient Full Name", value="Patient Test")
            with col_b:
                p_age = st.number_input("Patient Age", min_value=1, max_value=120, value=50)
            with col_c:
                p_gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            with col_d:
                p_date = st.date_input("Scan Date", value=datetime.date.today())

            uploaded_file = st.file_uploader("Choose Knee X-Ray Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
            submit_btn = st.form_submit_button("Run Diagnostic Scan & Save Log")

        if submit_btn and uploaded_file is not None:
            raw_img = Image.open(uploaded_file)
            
            if use_clahe:
                proc_img = apply_clahe(raw_img)
            else:
                proc_img = raw_img.convert("RGB")

            tensor_in = eval_transform(proc_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    logits = model(tensor_in)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            conf_pct = float(probs[pred_idx] * 100)

            # Generate Grad-CAM Heatmap Overlay
            heatmap_np = generate_gradcam(model, tensor_in, target_class=pred_idx)
            if heatmap_np is not None:
                gradcam_img = overlay_heatmap(raw_img, heatmap_np)
            else:
                gradcam_img = raw_img.convert("RGB")

            c_img1, c_img2, c_img3 = st.columns([1, 1, 1])
            with c_img1:
                st.markdown("**Uploaded Knee X-Ray**")
                st.image(raw_img, use_container_width=True)

            with c_img2:
                st.markdown("**Grad-CAM AI Attention Heatmap**")
                st.image(gradcam_img, use_container_width=True)

            with c_img3:
                st.markdown("### Diagnosis Result")
                if pred_class == "Normal":
                    st.success(f"Diagnosis: **{pred_class}** ({conf_pct:.2f}% confidence)")
                elif pred_class == "Osteopenia":
                    st.warning(f"Diagnosis: **{pred_class}** ({conf_pct:.2f}% confidence)")
                else:
                    st.error(f"Diagnosis: **{pred_class}** ({conf_pct:.2f}% confidence)")

                st.markdown("**Class Probabilities:**")
                for c_name, p_val in zip(CLASS_NAMES, probs):
                    st.text(f"{c_name:<15}: {p_val*100:.2f}%")
                    st.progress(float(p_val))

            log_id = f"PAT-{int(time.time()) % 10000}"
            new_record = {
                "patient_id": log_id,
                "patient_name": p_name,
                "patient_age": int(p_age),
                "patient_gender": p_gender,
                "scan_date": str(p_date),
                "diagnosis_result": pred_class,
                "confidence_pct": round(conf_pct, 2),
                "prob_normal": round(float(probs[0]), 4),
                "prob_osteopenia": round(float(probs[1]), 4),
                "prob_osteoporosis": round(float(probs[2]), 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_patient_log(new_record)
            st.info(f"Log Record Successfully Saved! Assigned Patient ID: **{log_id}**")

# ==============================================================================
# TAB 2: ADMINISTRATOR ANALYTICS DASHBOARD (INSPIRED BY SAMPLE.PY, ZERO EMOJIS)
# ==============================================================================
elif nav_mode == "Administrator Analytics Dashboard":
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.header("Administrator Analytics & Clinical Dashboard")

    df_logs = load_patient_logs()
    if df_logs.empty:
        st.warning("No patient log records found in system.")
    else:
        total_scans = len(df_logs)
        n_normal = len(df_logs[df_logs["diagnosis_result"] == "Normal"])
        n_penia  = len(df_logs[df_logs["diagnosis_result"] == "Osteopenia"])
        n_porosis = len(df_logs[df_logs["diagnosis_result"] == "Osteoporosis"])

        pct_normal = f"{100 * n_normal / total_scans:.1f}%" if total_scans > 0 else "0.0%"
        pct_penia  = f"{100 * n_penia / total_scans:.1f}%" if total_scans > 0 else "0.0%"
        pct_porosis = f"{100 * n_porosis / total_scans:.1f}%" if total_scans > 0 else "0.0%"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-card-futuristic">
                <div class="stat-label">Total Scans Conducted</div>
                <div class="stat-value">{total_scans}</div>
                <div class="stat-subtext">Active Patient Logs</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card-futuristic">
                <div class="stat-label">Normal Scans</div>
                <div class="stat-value">{n_normal}</div>
                <div class="stat-subtext">{pct_normal} of total</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card-futuristic">
                <div class="stat-label">Osteopenia Cases</div>
                <div class="stat-value">{n_penia}</div>
                <div class="stat-subtext">{pct_penia} of total</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="stat-card-futuristic">
                <div class="stat-label">Osteoporosis Cases</div>
                <div class="stat-value">{n_porosis}</div>
                <div class="stat-subtext">{pct_porosis} severe cases</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            st.markdown('<div class="chart-card-futuristic">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Daily Scan Volume Trajectory</div>', unsafe_allow_html=True)
            df_trend = df_logs.groupby("scan_date").size().reset_index(name="scan_count")
            
            fig_trend = go.Figure(data=[go.Scatter(
                x=df_trend["scan_date"],
                y=df_trend["scan_count"],
                mode='lines+markers',
                line=dict(color='#38bdf8', width=3, shape='spline'),
                marker=dict(size=8, color='#6366f1', line=dict(color='#38bdf8', width=2)),
                fill='tozeroy',
                fillcolor='rgba(56, 189, 248, 0.15)',
                hovertemplate='<b>Date:</b> %{x}<br><b>Scans:</b> %{y}<extra></extra>'
            )])
            fig_trend.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Scan Date"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Scans Conducted"),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_chart2:
            st.markdown('<div class="chart-card-futuristic">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Prevalence Breakdown of Diagnoses</div>', unsafe_allow_html=True)
            df_disease = df_logs["diagnosis_result"].value_counts()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(df_disease.index),
                values=list(df_disease.values),
                hole=0.45,
                marker=dict(
                    colors=['#10b981' if c == 'Normal' else '#f59e0b' if c == 'Osteopenia' else '#ef4444' for c in df_disease.index],
                    line=dict(color='#0f172a', width=2)
                ),
                textinfo='percent+label',
                textfont=dict(size=12, color='white')
            )])
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        c_chart3, c_chart4 = st.columns(2)

        with c_chart3:
            st.markdown('<div class="chart-card-futuristic">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Disease Severity Across Age Groups</div>', unsafe_allow_html=True)
            bins = [0, 40, 60, 120]
            labels_age = ["<40", "40-60", ">60"]
            df_logs["age_bracket"] = pd.cut(df_logs["patient_age"], bins=bins, labels=labels_age)
            df_age_diag = df_logs.groupby(["age_bracket", "diagnosis_result"], observed=False).size().reset_index(name="count")

            fig_age = px.bar(
                df_age_diag,
                x="age_bracket",
                y="count",
                color="diagnosis_result",
                color_discrete_map={"Normal": "#10b981", "Osteopenia": "#f59e0b", "Osteoporosis": "#ef4444"},
                barmode="stack",
                labels={"age_bracket": "Age Group", "count": "Patient Count", "diagnosis_result": "Diagnosis"}
            )
            fig_age.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_age, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_chart4:
            st.markdown('<div class="chart-card-futuristic">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Diagnoses Split by Patient Gender</div>', unsafe_allow_html=True)
            df_gender_diag = df_logs.groupby(["patient_gender", "diagnosis_result"], observed=False).size().reset_index(name="count")

            fig_gender = px.bar(
                df_gender_diag,
                x="patient_gender",
                y="count",
                color="diagnosis_result",
                color_discrete_map={"Normal": "#10b981", "Osteopenia": "#f59e0b", "Osteoporosis": "#ef4444"},
                barmode="group",
                labels={"patient_gender": "Gender", "count": "Patient Count", "diagnosis_result": "Diagnosis"}
            )
            fig_gender.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_gender, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="futuristic-table-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Recent Clinical Scan Activity</div>', unsafe_allow_html=True)
        
        recent_records = df_logs.tail(10).iloc[::-1]
        table_html = '<table class="futuristic-table"><thead><tr>'
        table_html += '<th>Patient ID</th><th>Patient Name</th><th>Age</th><th>Gender</th><th>Scan Date</th><th>Diagnosis Result</th><th>Confidence</th>'
        table_html += '</tr></thead><tbody>'
        
        for _, r in recent_records.iterrows():
            diag = r['diagnosis_result']
            badge_class = "badge-normal" if diag == "Normal" else "badge-osteopenia" if diag == "Osteopenia" else "badge-osteoporosis"
            table_html += '<tr>'
            table_html += f'<td><strong>{r["patient_id"]}</strong></td>'
            table_html += f'<td>{r["patient_name"]}</td>'
            table_html += f'<td>{r["patient_age"]}</td>'
            table_html += f'<td>{r["patient_gender"]}</td>'
            table_html += f'<td>{r["scan_date"]}</td>'
            table_html += f'<td><span class="{badge_class}">{diag}</span></td>'
            table_html += f'<td>{r["confidence_pct"]}%</td>'
            table_html += '</tr>'
        
        table_html += '</tbody></table>'
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 3: PATIENT MASTER RECORDS
# ==============================================================================
elif nav_mode == "Patient Master Records":
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.header("Patient Master Records Database")

    df_logs = load_patient_logs()
    if not df_logs.empty:
        search_query = st.text_input("Search Patient by Name or Patient ID")
        if search_query:
            df_logs = df_logs[df_logs["patient_name"].str.contains(search_query, case=False, na=False) | 
                              df_logs["patient_id"].str.contains(search_query, case=False, na=False)]

        st.dataframe(df_logs, use_container_width=True)

        csv_data = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Complete Patient Logs CSV",
            data=csv_data,
            file_name=f"patient_records_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
