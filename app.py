# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from src.decision import get_decision

# ── Load Model ──────────────────────────────────────────────
model = tf.keras.models.load_model("model/vehicle_model.keras")
class_names = ["Ambulance", "Bike", "Bus", "Car", "Truck"]

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="RoadSense AI",
    page_icon="Car",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Master CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Rajdhani', sans-serif !important;
    background-color: #06020e !important;
    color: #f5f0ff !important;
}

/* ── BACKGROUND ── */
.stApp {
    background:
        linear-gradient(rgba(124,58,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,255,0.04) 1px, transparent 1px),
        radial-gradient(ellipse at 20% 20%, rgba(80,20,160,0.2) 0%, transparent 60%),
        #06020e !important;
    background-size: 60px 60px, 60px 60px, 100% 100%, 100% 100% !important;
}

/* ── HIDE STREAMLIT DEFAULTS ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(14, 5, 32, 0.97) !important;
    border-right: 1px solid rgba(124,58,255,0.25) !important;
}
[data-testid="stSidebar"] * { color: #c4b0e8 !important; }

/* ── TOP NAV BAR ── */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 40px;
    height: 64px;
    background: rgba(6,2,14,0.92);
    border-bottom: 1px solid rgba(124,58,255,0.22);
    backdrop-filter: blur(20px);
    position: sticky;
    top: 0;
    z-index: 999;
    margin: 0 -2rem 2rem -2rem;
}

.topnav-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #e040fb;
    text-shadow: 0 0 20px rgba(224,64,251,0.5);
    display: flex;
    align-items: center;
    gap: 10px;
}

.topnav-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #e040fb;
    box-shadow: 0 0 10px #e040fb;
    animation: blink 2s infinite;
}

@keyframes blink {
    0%,100% { opacity:1; }
    50% { opacity:0.3; }
}

.topnav-links {
    display: flex;
    gap: 6px;
}

.nav-btn {
    padding: 7px 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #c4b0e8;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.25s;
    text-decoration: none;
}

.nav-btn:hover, .nav-btn.active {
    color: #e040fb;
    border-color: rgba(224,64,251,0.4);
    background: rgba(224,64,251,0.07);
    text-shadow: 0 0 10px rgba(224,64,251,0.4);
}

/* ── SECTION BADGE ── */
.section-tag {
    display: inline-block;
    padding: 4px 14px;
    border: 1px solid rgba(224,64,251,0.4);
    border-radius: 100px;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #e040fb;
    margin-bottom: 12px;
}

/* ── HERO ── */
.hero-wrap {
    text-align: center;
    padding: 60px 20px 40px;
}

.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2.5rem, 6vw, 5rem);
    font-weight: 900;
    letter-spacing: 0.05em;
    line-height: 1.1;
    margin: 0 0 12px;
    color: #f5f0ff;
}

.hero-title .accent {
    color: #e040fb;
    text-shadow: 0 0 40px rgba(224,64,251,0.55);
}

.hero-title .accent2 { color: #9d6bff; }

.hero-sub {
    font-size: 1rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #c4b0e8;
    margin-bottom: 20px;
}

.hero-desc {
    max-width: 620px;
    margin: 0 auto 40px;
    font-size: 1.05rem;
    color: #c4b0e8;
    line-height: 1.7;
}

/* ── STATS ROW ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    border: 1px solid rgba(124,58,255,0.22);
    border-radius: 12px;
    overflow: hidden;
    margin: 0 0 60px;
    background: rgba(124,58,255,0.08);
}

.stat-item {
    padding: 28px 20px;
    text-align: center;
    border-right: 1px solid rgba(124,58,255,0.15);
}
.stat-item:last-child { border-right: none; }

.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e040fb;
    text-shadow: 0 0 20px rgba(224,64,251,0.4);
    display: block;
    margin-bottom: 6px;
}

.stat-label {
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6e5a90;
}

/* ── CARDS ── */
.pcard {
    background: rgba(20,8,50,0.65);
    border: 1px solid rgba(124,58,255,0.22);
    border-radius: 14px;
    padding: 28px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
    transition: border-color 0.3s;
}

.pcard:hover { border-color: rgba(224,64,251,0.4); }

.pcard-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e040fb;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.pcard-title::before {
    content: '';
    display: block;
    width: 4px; height: 16px;
    background: #e040fb;
    border-radius: 2px;
    box-shadow: 0 0 8px #e040fb;
    flex-shrink: 0;
}

/* ── FEATURE GRID ── */
.feat-grid {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 20px;
    margin-bottom: 50px;
}

.feat-card {
    background: rgba(20,8,50,0.6);
    border: 1px solid rgba(124,58,255,0.2);
    border-radius: 12px;
    padding: 28px 24px;
    transition: all 0.3s;
}

.feat-card:hover {
    border-color: rgba(224,64,251,0.4);
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.feat-icon { font-size: 1.8rem; margin-bottom: 14px; display: block; }

.feat-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: #f5f0ff;
    margin-bottom: 10px;
    letter-spacing: 0.04em;
}

.feat-text { font-size: 0.9rem; color: #c4b0e8; line-height: 1.6; }

/* ── DECISION CARDS ── */
.dec-grid {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 20px;
    margin-bottom: 50px;
}

.dec-card {
    border-radius: 12px;
    padding: 30px 24px;
    text-align: center;
    border: 1px solid;
}

.dec-card.high { background: rgba(0,230,118,0.07); border-color: rgba(0,230,118,0.3); }
.dec-card.mid  { background: rgba(255,179,0,0.07);  border-color: rgba(255,179,0,0.3); }
.dec-card.low  { background: rgba(255,82,82,0.07);  border-color: rgba(255,82,82,0.3); }

.dec-icon  { font-size: 2rem; margin-bottom: 12px; display: block; }
.dec-label { font-family:'Orbitron',monospace; font-size:0.8rem; font-weight:700; letter-spacing:0.08em; margin-bottom:8px; }
.dec-card.high .dec-label { color: #00e676; }
.dec-card.mid  .dec-label { color: #ffb300; }
.dec-card.low  .dec-label { color: #ff5252; }
.dec-thresh { font-family:'Orbitron',monospace; font-size:1.5rem; font-weight:700; color:#f5f0ff; margin-bottom:8px; }
.dec-desc   { font-size:0.88rem; color:#c4b0e8; line-height:1.6; }

/* ── UPLOAD ZONE ── */
.upload-hint {
    border: 2px dashed rgba(124,58,255,0.4);
    border-radius: 14px;
    padding: 40px 20px;
    text-align: center;
    color: #c4b0e8;
    margin-bottom: 12px;
    font-size: 0.95rem;
}

/* ── RESULT SECTION ── */
.result-class-box {
    background: rgba(124,58,255,0.1);
    border: 1px solid rgba(124,58,255,0.3);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-bottom: 20px;
}

.result-class-box.ambulance {
    background: rgba(255,82,82,0.1);
    border-color: rgba(255,82,82,0.4);
}

.result-name {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    color: #f5f0ff;
    letter-spacing: 0.08em;
    display: block;
    margin-bottom: 4px;
}

.result-sublabel {
    font-size: 0.78rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6e5a90;
}

.conf-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.conf-label { font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase; color:#6e5a90; font-weight:600; }
.conf-value { font-family:'Orbitron',monospace; font-size:1rem; font-weight:700; color:#e040fb; }

.dec-badge {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 20px;
    border-radius: 10px;
    border: 1px solid;
    margin-bottom: 20px;
}

.dec-badge.high { background:rgba(0,230,118,0.07); border-color:rgba(0,230,118,0.3); }
.dec-badge.mid  { background:rgba(255,179,0,0.07);  border-color:rgba(255,179,0,0.3); }
.dec-badge.low  { background:rgba(255,82,82,0.07);  border-color:rgba(255,82,82,0.3); }
.dec-badge.emergency { background:rgba(255,82,82,0.12); border-color:rgba(255,82,82,0.5); animation: pulse-red 1.5s infinite; }

@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 rgba(255,82,82,0); }
    50% { box-shadow: 0 0 18px rgba(255,82,82,0.4); }
}

.dec-badge-icon { font-size:1.8rem; flex-shrink:0; }
.dec-badge-title { font-family:'Orbitron',monospace; font-size:0.82rem; font-weight:700; letter-spacing:0.08em; display:block; margin-bottom:3px; }
.dec-badge.high .dec-badge-title { color:#00e676; }
.dec-badge.mid  .dec-badge-title { color:#ffb300; }
.dec-badge.low  .dec-badge-title { color:#ff5252; }
.dec-badge.emergency .dec-badge-title { color:#ff5252; }
.dec-badge-desc { font-size:0.85rem; color:#c4b0e8; }

/* ── SCORE BARS ── */
.score-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.score-name { font-size:0.78rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; color:#c4b0e8; width:80px; flex-shrink:0; }
.score-track { flex:1; height:6px; background:rgba(255,255,255,0.06); border-radius:100px; overflow:hidden; }
.score-fill  { height:100%; border-radius:100px; background: linear-gradient(90deg,#7c3aff,#e040fb); }
.score-pct   { font-family:'Orbitron',monospace; font-size:0.72rem; color:#6e5a90; width:38px; text-align:right; }

/* ── INFO PAGE ── */
.info-table { width:100%; border-collapse:collapse; }
.info-table tr { border-bottom: 1px solid rgba(124,58,255,0.15); }
.info-table tr:last-child { border-bottom:none; }
.info-table td { padding:11px 0; font-size:0.9rem; vertical-align:top; }
.info-table td:first-child { color:#6e5a90; font-size:0.8rem; letter-spacing:0.05em; text-transform:uppercase; width:42%; font-weight:500; }
.info-table td:last-child  { color:#f5f0ff; font-weight:500; }

.tag { display:inline-block; padding:3px 10px; border-radius:4px; font-size:0.75rem; font-weight:600; letter-spacing:0.05em; margin:2px; }
.tag-p { background:rgba(124,58,255,0.2); color:#9d6bff; border:1px solid rgba(124,58,255,0.3); }
.tag-m { background:rgba(224,64,251,0.1); color:#e040fb; border:1px solid rgba(224,64,251,0.25); }
.tag-r { background:rgba(255,82,82,0.1);  color:#ff5252; border:1px solid rgba(255,82,82,0.25); }

.bar-row { display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.bar-name { font-size:0.78rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; color:#c4b0e8; width:80px; flex-shrink:0; }
.bar-track { flex:1; height:7px; background:rgba(255,255,255,0.06); border-radius:100px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:100px; background:linear-gradient(90deg,#7c3aff,#e040fb); }
.bar-fill.red { background:linear-gradient(90deg,#c62828,#ff5252); }
.bar-num   { font-family:'Orbitron',monospace; font-size:0.72rem; color:#6e5a90; width:38px; text-align:right; }

.step-item { display:flex; gap:14px; align-items:flex-start; margin-bottom:14px; font-size:0.9rem; color:#c4b0e8; line-height:1.6; }
.step-num  { font-family:'Orbitron',monospace; font-size:0.72rem; font-weight:700; color:#e040fb; background:rgba(224,64,251,0.1); border:1px solid rgba(224,64,251,0.25); border-radius:4px; padding:3px 7px; flex-shrink:0; margin-top:2px; }

/* ── VEHICLE CHIPS ── */
.chip-row { display:flex; gap:10px; flex-wrap:wrap; }
.chip {
    text-align:center;
    padding:12px 16px;
    border:1px solid rgba(124,58,255,0.25);
    border-radius:10px;
    font-size:0.78rem;
    letter-spacing:0.06em;
    text-transform:uppercase;
    font-weight:600;
    color:#c4b0e8;
    background:rgba(20,8,50,0.5);
    flex:1;
    min-width:80px;
}
.chip.amb { border-color:rgba(255,82,82,0.4); color:#ff5252; }
.chip-icon { font-size:1.4rem; display:block; margin-bottom:6px; }

/* ── STREAMLIT OVERRIDES ── */
.stFileUploader > div {
    background: rgba(20,8,50,0.5) !important;
    border: 2px dashed rgba(124,58,255,0.4) !important;
    border-radius: 12px !important;
    color: #c4b0e8 !important;
}

.stFileUploader label { color: #c4b0e8 !important; font-family:'Rajdhani',sans-serif !important; }

div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #7c3aff, #e040fb) !important;
}

.stSpinner > div { border-top-color: #e040fb !important; }

/* Override st.success / warning / error */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
}

/* sidebar nav buttons */
.sidebar-nav-btn {
    display: block;
    width: 100%;
    padding: 10px 16px;
    margin-bottom: 6px;
    background: rgba(124,58,255,0.08);
    border: 1px solid rgba(124,58,255,0.2);
    border-radius: 8px;
    color: #c4b0e8;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    text-align: left;
    transition: all 0.2s;
}

.sidebar-nav-btn.active, .sidebar-nav-btn:hover {
    background: rgba(224,64,251,0.12);
    border-color: rgba(224,64,251,0.4);
    color: #e040fb;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid rgba(124,58,255,0.3);
}

h1,h2,h3,h4,h5 { font-family:'Rajdhani',sans-serif !important; color:#f5f0ff !important; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE for page ───────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"


# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:24px 0 20px;">
        <div style="font-family:'Orbitron',monospace; font-size:1.1rem; font-weight:700;
                    color:#e040fb; text-shadow:0 0 16px rgba(224,64,251,0.5);
                    letter-spacing:0.12em;">
            &#9679; RoadSenseAI
        </div>
        <div style="font-size:0.75rem; letter-spacing:0.15em; text-transform:uppercase;
                    color:#6e5a90; margin-top:4px;">
            Vehicle Classification
        </div>
    </div>
    <hr style="border-color:rgba(124,58,255,0.2); margin:0 0 20px;">
    """, unsafe_allow_html=True)

    pages = {"Home": "Home", "Classify": "Classify", "About": "About"}
    for label, key in pages.items():
        active = "active" if st.session_state.page == key else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("""
    <hr style="border-color:rgba(124,58,255,0.2); margin:20px 0;">
    <div style="font-family:'Orbitron',monospace; font-size:0.72rem; color:#e040fb;
                letter-spacing:0.1em; margin-bottom:12px;">VEHICLE CLASSES</div>
    """, unsafe_allow_html=True)

    chips = [("AMB","Ambulance","#ff5252"), ("BIKE","Bike","#9d6bff"),
             ("BUS","Bus","#9d6bff"), ("CAR","Car","#9d6bff"), ("TRK","Truck","#9d6bff")]
    for icon, name, color in chips:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding:8px 12px;
                    border:1px solid rgba(124,58,255,0.18); border-radius:8px;
                    margin-bottom:6px; background:rgba(20,8,50,0.4);">
            <span style="font-size:1.1rem;">{icon}</span>
            <span style="font-size:0.85rem; font-weight:600; color:{color};
                         letter-spacing:0.06em; text-transform:uppercase;">{name}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="border-color:rgba(124,58,255,0.2); margin:16px 0;">
    <div style="font-size:0.75rem; color:#6e5a90; text-align:center; line-height:1.6;">
        Powered by MobileNetV2<br>Trained on 5-class vehicle dataset
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: HOME
# ════════════════════════════════════════════════════════════
if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero-wrap">
        <div class="section-tag">Deep Learning &middot; Computer Vision &middot; Real-Time</div>
        <h1 class="hero-title">
            Road<span class="accent">Sense</span><span class="accent2">AI</span>
        </h1>
        <p class="hero-sub">Vehicle Type Classification System</p>
        <p class="hero-desc">
            An intelligent deep learning system that classifies vehicle images into 5 categories
            with confidence-based decision logic &mdash; including priority detection of ambulances
            for emergency response.
        </p>
    </div>

    <div class="stats-row">
        <div class="stat-item">
            <span class="stat-value">5</span>
            <span class="stat-label">Vehicle Classes</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">MobileNetV2</span>
            <span class="stat-label">Model Architecture</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">&ge;85%</span>
            <span class="stat-label">High Confidence Threshold</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">224px</span>
            <span class="stat-label">Input Resolution</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features
    st.markdown('<div class="section-tag">Core Capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Orbitron,monospace; font-size:1.4rem; font-weight:700; margin-bottom:24px;">What RoadSenseAI Does</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card">
            <span class="feat-icon">&#129504;</span>
            <div class="feat-title">Transfer Learning CNN</div>
            <p class="feat-text">MobileNetV2 backbone fine-tuned on a curated vehicle dataset with partial layer unfreezing for optimal accuracy.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">&#128680;</span>
            <div class="feat-title">Ambulance Priority Detection</div>
            <p class="feat-text">Special classification logic for ambulance detection with a dedicated emergency decision tier at &ge;75% confidence.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">&#128202;</span>
            <div class="feat-title">Confidence Decision Layer</div>
            <p class="feat-text">Three-tier confidence system: High Confidence, Needs Review, and Uncertain &mdash; with justifiable thresholds.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">&#128300;</span>
            <div class="feat-title">Image Preprocessing</div>
            <p class="feat-text">Automated resize to 224&times;224, pixel normalization, and augmentation (rotation, zoom, flip) applied during training.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">&#128200;</span>
            <div class="feat-title">Class Imbalance Handling</div>
            <p class="feat-text">Automatic class weight computation ensures balanced learning across all vehicle categories during training.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">&#9889;</span>
            <div class="feat-title">Offline Ready</div>
            <p class="feat-text">Runs entirely without internet connection. No cloud dependencies &mdash; full inference on local compute.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Decision system
    st.markdown('<div class="section-tag">Intelligent Layer</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Orbitron,monospace; font-size:1.4rem; font-weight:700; margin-bottom:24px;">Confidence Decision System</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="dec-grid">
        <div class="dec-card high">
            <span class="dec-icon">&#9989;</span>
            <div class="dec-label">High Confidence</div>
            <div class="dec-thresh">&ge; 85%</div>
            <p class="dec-desc">System is highly certain. Classification result is trusted and reliable for real-world use.</p>
        </div>
        <div class="dec-card mid">
            <span class="dec-icon">&#10067;</span>
            <div class="dec-label">Needs Review</div>
            <div class="dec-thresh">65% &ndash; 84%</div>
            <p class="dec-desc">Ambiguous prediction. Human verification recommended before acting on the result.</p>
        </div>
        <div class="dec-card low">
            <span class="dec-icon">&#9888;</span>
            <div class="dec-label">Uncertain</div>
            <div class="dec-thresh">&lt; 65%</div>
            <p class="dec-desc">Low model confidence. Image may be unclear, occluded, or outside training distribution.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: CLASSIFY
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "Classify":

    st.markdown("""
    <div style="padding:30px 0 24px;">
        <div class="section-tag">AI Engine</div>
        <div style="font-family:Orbitron,monospace; font-size:1.6rem; font-weight:700; margin-top:10px;">
            Vehicle Classification
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="pcard"><div class="pcard-title">Upload Image</div>', unsafe_allow_html=True)
        file = st.file_uploader("Drop a vehicle image or click to browse",
                                type=["jpg", "jpeg", "png", "avif"],
                                label_visibility="collapsed")

        if file:
            img = Image.open(file).convert("RGB")
            st.image(img, use_column_width=True)
        else:
            st.markdown("""
            <div class="upload-hint">
                <div style="font-size:2.5rem; margin-bottom:12px;">&#128193;</div>
                <div style="font-weight:600; margin-bottom:6px;">Drop an image here or click to browse</div>
                <div style="font-size:0.82rem; color:#6e5a90;">Supports JPG, PNG, AVIF</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Vehicle class chips
        st.markdown("""
        <div class="pcard">
            <div class="pcard-title">Vehicle Classes</div>
            <div class="chip-row">
                <div class="chip"><span class="chip-icon">&#128663;</span>Car</div>
                <div class="chip"><span class="chip-icon">&#128667;</span>Truck</div>
                <div class="chip"><span class="chip-icon">&#127949;️</span>Bike</div>
                <div class="chip"><span class="chip-icon">&#128652;</span>Bus</div>
                <div class="chip amb"><span class="chip-icon">&#128657;</span>Ambulance</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="pcard" style="min-height:500px;"><div class="pcard-title">Classification Results</div>', unsafe_allow_html=True)

        if not file:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#6e5a90;">
                <div style="font-size:3rem; margin-bottom:16px; opacity:0.4;">&#128269;</div>
                <div style="font-size:0.95rem; margin-bottom:8px; color:#c4b0e8;">No image uploaded yet</div>
                <div style="font-size:0.85rem;">Upload a vehicle image on the left<br>to see classification results here</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            with st.spinner("Analysing image..."):
                time.sleep(1)
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array, verbose=0)

            confidence = float(np.max(pred))
            label = class_names[np.argmax(pred)]
            decision = get_decision(label, confidence)
            pct = int(confidence * 100)

            # Result class box
            amb_class = "ambulance" if label == "Ambulance" else ""
            st.markdown(f"""
            <div class="result-class-box {amb_class}">
                <span class="result-name">{'&#128657; ' if label == 'Ambulance' else ''}{label}</span>
                <span class="result-sublabel">Predicted Vehicle Class</span>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            if confidence >= 0.85:
                bar_color = "linear-gradient(90deg,#00c853,#00e676)"
            elif confidence >= 0.65:
                bar_color = "linear-gradient(90deg,#ff8f00,#ffb300)"
            else:
                bar_color = "linear-gradient(90deg,#c62828,#ff5252)"

            st.markdown(f"""
            <div style="margin-bottom:20px;">
                <div class="conf-label-row">
                    <span class="conf-label">Confidence Score</span>
                    <span class="conf-value">{pct}%</span>
                </div>
                <div style="height:8px; background:rgba(255,255,255,0.07); border-radius:100px; overflow:hidden;">
                    <div style="height:100%; width:{pct}%; background:{bar_color}; border-radius:100px; transition:width 1s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Decision badge
            if label == "Ambulance" and confidence >= 0.75:
                st.markdown(f"""
                <div class="dec-badge emergency">
                    <span class="dec-badge-icon">&#128680;</span>
                    <div>
                        <span class="dec-badge-title">HIGH PRIORITY EMERGENCY</span>
                        <span class="dec-badge-desc">Ambulance detected with high confidence &mdash; priority vehicle</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif confidence >= 0.85:
                st.markdown(f"""
                <div class="dec-badge high">
                    <span class="dec-badge-icon">&#9989;</span>
                    <div>
                        <span class="dec-badge-title">HIGH CONFIDENCE</span>
                        <span class="dec-badge-desc">Classification trusted. Result is reliable for real-world use.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif confidence >= 0.65:
                st.markdown(f"""
                <div class="dec-badge mid">
                    <span class="dec-badge-icon">&#10067;</span>
                    <div>
                        <span class="dec-badge-title">AMBIGUOUS &mdash; NEEDS REVIEW</span>
                        <span class="dec-badge-desc">Human verification recommended before acting on this result.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="dec-badge low">
                    <span class="dec-badge-icon">&#9888;</span>
                    <div>
                        <span class="dec-badge-title">UNCERTAIN PREDICTION</span>
                        <span class="dec-badge-desc">Low confidence. Image may be unclear or out of distribution.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # All class scores
            st.markdown('<div style="font-family:Orbitron,monospace; font-size:0.78rem; color:#e040fb; letter-spacing:0.1em; margin-bottom:14px;">ALL CLASS SCORES</div>', unsafe_allow_html=True)
            probs = pred[0]
            for i, prob in enumerate(probs):
                p = int(prob * 100)
                fill_color = "linear-gradient(90deg,#c62828,#ff5252)" if class_names[i] == "Ambulance" else "linear-gradient(90deg,#7c3aff,#e040fb)"
                st.markdown(f"""
                <div class="score-row">
                    <span class="score-name">{class_names[i]}</span>
                    <div class="score-track">
                        <div class="score-fill" style="width:{p}%; background:{fill_color};"></div>
                    </div>
                    <span class="score-pct">{p}%</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "About":

    st.markdown("""
    <div style="padding:30px 0 24px;">
        <div class="section-tag">Technical Documentation</div>
        <div style="font-family:Orbitron,monospace; font-size:1.6rem; font-weight:700; margin-top:10px;">
            Dataset &middot; Model &middot; Architecture
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Dataset Info
        st.markdown("""
        <div class="pcard">
            <div class="pcard-title">Dataset Information</div>
            <table class="info-table">
                <tr><td>Source</td><td>Publicly available vehicle image dataset</td></tr>
                <tr><td>Total Classes</td><td><span class="tag tag-m">5 Classes</span></td></tr>
                <tr><td>Vehicle Types</td><td>
                    <span class="tag tag-p">Car</span>
                    <span class="tag tag-p">Truck</span>
                    <span class="tag tag-p">Bike</span>
                    <span class="tag tag-p">Bus</span>
                    <span class="tag tag-r">Ambulance ★#9733;</span>
                </td></tr>
                <tr><td>Image Format</td><td>JPEG / PNG</td></tr>
                <tr><td>Input Resolution</td><td>224 &times; 224 px (resized)</td></tr>
                <tr><td>Color Space</td><td>RGB (normalized 0&ndash;1)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Class Distribution
        st.markdown("""
        <div class="pcard">
            <div class="pcard-title">Class Distribution</div>
            <div class="bar-row"><span class="bar-name">Car</span>
                <div class="bar-track"><div class="bar-fill" style="width:90%"></div></div>
                <span class="bar-num">~900</span></div>
            <div class="bar-row"><span class="bar-name">Truck</span>
                <div class="bar-track"><div class="bar-fill" style="width:85%"></div></div>
                <span class="bar-num">~850</span></div>
            <div class="bar-row"><span class="bar-name">Bike</span>
                <div class="bar-track"><div class="bar-fill" style="width:83%"></div></div>
                <span class="bar-num">~830</span></div>
            <div class="bar-row"><span class="bar-name">Bus</span>
                <div class="bar-track"><div class="bar-fill" style="width:81%"></div></div>
                <span class="bar-num">~810</span></div>
            <div class="bar-row"><span class="bar-name">Ambulance</span>
                <div class="bar-track"><div class="bar-fill red" style="width:88%"></div></div>
                <span class="bar-num">~880</span></div>
            <p style="margin-top:16px; font-size:0.82rem; color:#6e5a90;">
                ★#9733; Ambulance is prioritised as a special detection category with dedicated decision logic.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Model Architecture
        st.markdown("""
        <div class="pcard">
            <div class="pcard-title">Model Architecture</div>
            <table class="info-table">
                <tr><td>Base Model</td><td>MobileNetV2 (ImageNet weights)</td></tr>
                <tr><td>Fine-tuning</td><td>Last 30 layers unfrozen</td></tr>
                <tr><td>Layers Added</td><td>GlobalAvgPool → BatchNorm → Dense(128) → Dropout(0.5) → Softmax</td></tr>
                <tr><td>Optimizer</td><td>Adam (lr = 0.0001)</td></tr>
                <tr><td>Loss</td><td>Categorical Cross-Entropy</td></tr>
                <tr><td>Epochs</td><td>10</td></tr>
                <tr><td>Output Classes</td><td>5</td></tr>
                <tr><td>Evaluation</td><td>Accuracy &middot; Precision &middot; Recall</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Preprocessing
        st.markdown("""
        <div class="pcard">
            <div class="pcard-title">Preprocessing Pipeline</div>
            <div class="step-item"><span class="step-num">01</span>Resize all images to <strong>224 &times; 224</strong> pixels</div>
            <div class="step-item"><span class="step-num">02</span>Convert to RGB, discard alpha channels</div>
            <div class="step-item"><span class="step-num">03</span>Normalize pixel values from [0&ndash;255] to <strong>[0.0&ndash;1.0]</strong></div>
            <div class="step-item"><span class="step-num">04</span>Augmentation: random horizontal flip, ±25° rotation, 20% zoom</div>
            <div class="step-item"><span class="step-num">05</span>Train / Validation split: <strong>80% / 20%</strong></div>
            <div class="step-item"><span class="step-num">06</span>Class weight balancing applied to handle imbalance</div>
        </div>
        """, unsafe_allow_html=True)

    # Confidence thresholds full width
    st.markdown("""
    <div class="pcard">
        <div class="pcard-title">Confidence Decision Rules</div>
        <table class="info-table">
            <tr>
                <td>Ambulance &ge; 0.75</td>
                <td><span style="color:#ff5252; font-weight:700;">&#128680; High Priority Emergency</span> &mdash; special ambulance tier</td>
            </tr>
            <tr>
                <td>&ge; 0.85 (85%)</td>
                <td><span style="color:#00e676; font-weight:700;">&#9989; High Confidence Classification</span> &mdash; trusted result</td>
            </tr>
            <tr>
                <td>0.65 &ndash; 0.84</td>
                <td><span style="color:#ffb300; font-weight:700;">&#10067; Ambiguous / Needs Review</span> &mdash; verify manually</td>
            </tr>
            <tr>
                <td>&lt; 0.65 (65%)</td>
                <td><span style="color:#ff5252; font-weight:700;">&#9888; Uncertain Prediction</span> &mdash; low confidence</td>
            </tr>
        </table>
        <p style="margin-top:18px; font-size:0.85rem; color:#6e5a90;">
            Thresholds chosen based on validation experiments to balance precision and recall,
            particularly for ambulance detection sensitivity.
        </p>
    </div>
    """, unsafe_allow_html=True)