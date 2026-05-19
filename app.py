"""
YouTube Spam Detector â€” Red Neuronal para DetecciÃ³n de Spam
Proyecto I: IntroducciÃ³n a la IA
Universitat PolitÃ¨cnica de ValÃ¨ncia

Modelo: Red Neuronal Superficial (MLP) con TF-IDF + Feature Engineering
Dataset: YouTube Comments (5 artistas: PSY, Katy Perry, LMFAO, Eminem, Shakira)
"""

import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="YouTube Spam Detector IA",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CSS â€” TEMA OSCURO CON NEON
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("""
<style>
:root {
  --bg0: #080812; --bg1: #0f0f23; --bg2: #1a1a35; --bg3: #1e2040;
  --cyan:   #00d4ff; --green: #00ff88; --red: #ff4455;
  --purple: #a855f7; --orange: #ff8c42;
  --txt: #e2e8f0; --txt2: #94a3b8;
  --border: rgba(0,212,255,0.18);
}

/* ── Fondo ─────────────────────────────────────────────────────────────── */
.stApp, .stApp > header {
  background: linear-gradient(135deg,var(--bg0) 0%,var(--bg1) 50%,var(--bg0) 100%) !important;
}
section[data-testid="stSidebar"] {
  background: rgba(8,8,18,0.97) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Hero ───────────────────────────────────────────────────────────────── */
.hero {
  background: linear-gradient(135deg,#0f0f23,#1a0a2e,#0a1628);
  border: 1px solid var(--border); border-radius: 24px;
  padding: 2.5rem 3rem 2rem; text-align: center;
  margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero::before {
  content:''; position:absolute; inset:0;
  background: radial-gradient(ellipse 80% 60% at 50% 0%,rgba(0,212,255,0.10) 0%,transparent 70%);
  pointer-events:none;
}
.hero-title {
  font-size: 3.2rem; font-weight: 900;
  background: linear-gradient(90deg,#00d4ff 0%,#a855f7 50%,#00ff88 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text; margin:0 0 0.4rem; letter-spacing:-0.02em; line-height:1.1;
}
.hero-sub { color:var(--txt2); font-size:1.05rem; margin:0; }
.hero-badge {
  display:inline-block;
  background:rgba(0,212,255,0.12); border:1px solid rgba(0,212,255,0.35);
  color:var(--cyan); padding:0.3rem 1rem; border-radius:50px;
  font-size:0.8rem; font-weight:700; letter-spacing:0.08em;
  margin-top:1rem; text-transform:uppercase;
}

/* ── Cards ──────────────────────────────────────────────────────────────── */
.card {
  background:rgba(26,26,53,0.7); border:1px solid var(--border);
  border-radius:16px; padding:1.5rem; backdrop-filter:blur(8px);
  transition:border-color 0.25s,transform 0.2s; margin-bottom:1rem;
}
.card:hover { border-color:rgba(0,212,255,0.4); transform:translateY(-1px); }

/* ── Veredicto ──────────────────────────────────────────────────────────── */
.verdict {
  text-align:center; padding:2rem 1rem 1.5rem;
  border-radius:20px; margin:1rem 0;
}
.verdict.spam {
  background:rgba(255,68,85,0.08); border:2px solid var(--red);
  box-shadow:0 0 40px rgba(255,68,85,0.15);
}
.verdict.safe {
  background:rgba(0,255,136,0.08); border:2px solid var(--green);
  box-shadow:0 0 40px rgba(0,255,136,0.15);
}
.verdict-icon { font-size:4rem; display:block; margin-bottom:0.5rem; }
.verdict-label {
  font-size:2rem; font-weight:900; letter-spacing:0.08em;
  text-transform:uppercase; display:block; margin-bottom:0.3rem;
}
.verdict.spam .verdict-label { color:var(--red); }
.verdict.safe .verdict-label { color:var(--green); }
.verdict-prob { font-size:1rem; color:var(--txt2); display:block; }

/* ── Feature pills ──────────────────────────────────────────────────────── */
.pills { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.6rem; }
.pill {
  display:inline-block; padding:0.3rem 0.85rem;
  border-radius:30px; font-size:0.78rem; font-weight:700;
}
.pill-on  { background:rgba(255,68,85,0.15);   border:1px solid var(--red);             color:var(--red);   }
.pill-off { background:rgba(148,163,184,0.07); border:1px solid rgba(148,163,184,0.2); color:var(--txt2); }

/* ── Sidebar stats ──────────────────────────────────────────────────────── */
.stat-box {
  background:rgba(26,26,53,0.5); border:1px solid var(--border);
  border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.6rem; text-align:center;
}
.stat-val { font-size:1.5rem; font-weight:800; color:var(--cyan); }
.stat-lbl { font-size:0.75rem; color:var(--txt2); text-transform:uppercase; letter-spacing:0.05em; }

/* ── Overrides Streamlit ────────────────────────────────────────────────── */
.stTextArea textarea {
  background:rgba(26,26,53,0.9) !important; border:1px solid rgba(0,212,255,0.3) !important;
  color:var(--txt) !important; border-radius:12px !important; font-size:0.95rem !important;
}
.stTextArea textarea:focus {
  border-color:var(--cyan) !important;
  box-shadow:0 0 0 2px rgba(0,212,255,0.15) !important;
}
.stButton > button {
  background:linear-gradient(135deg,#00d4ff 0%,#a855f7 100%) !important;
  border:none !important; color:white !important; font-weight:800 !important;
  border-radius:12px !important; padding:0.8rem 2rem !important;
  font-size:1rem !important; letter-spacing:0.03em !important;
  transition:all 0.25s !important; width:100% !important;
}
.stButton > button:hover {
  transform:translateY(-2px) !important;
  box-shadow:0 8px 25px rgba(0,212,255,0.35) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background:rgba(26,26,53,0.6) !important; border-radius:12px !important;
  padding:4px !important; border:1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"]       { border-radius:9px !important; color:var(--txt2) !important; }
.stTabs [aria-selected="true"]     { background:rgba(0,212,255,0.15) !important; color:var(--cyan) !important; }
[data-testid="stMetricValue"]      { color:var(--cyan) !important; font-size:1.6rem !important; font-weight:800 !important; }
[data-testid="stMetricLabel"]      { color:var(--txt2) !important; font-size:0.8rem !important; }
h1,h2,h3 { color:var(--txt) !important; }
p, li    { color:var(--txt2) !important; }
hr       { border-color:var(--border) !important; }
.stProgress > div > div {
  background:linear-gradient(90deg,var(--cyan),var(--purple)) !important;
  border-radius:4px !important;
}
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CONSTANTES Y HELPERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
SPAM_WORDS = re.compile(
    r'subscribe|suscrib|check.?out|my.?channel|free|gratis|click|win|gana|'
    r'giveaway|sorteo|crypto|bitcoin|investment|earn|profit|dm.?me|'
    r'followers|views|likes|watch.?now|promo|discount|descuento|link',
    re.I,
)

EJEMPLOS_SPAM = [
    "Subscribe to my channel NOW! FREE music + giveaway: http://bit.ly/freemusic",
    "WIN $1000 FREE! Click here and subscribe â†’ www.spam.com/win",
    "Check out my channel!! 1000 subs giveaway! Subscribe NOW!!!!!",
    "earn money fast click here dm me for info crypto investment profit guaranteed",
]
EJEMPLOS_REAL = [
    "This is one of my favorite songs of all time. Pure nostalgia.",
    "The guitar solo at 2:47 gives me chills every single time I listen to this.",
    "I showed this song to my dad and now he loves it too!",
    "Honestly this takes me back to 2012, what a time to be alive.",
]

MODEL_DIR = "model"


def engineer_features(texts) -> np.ndarray:
    """5 features numéricas derivadas del EDA (Sprint 2)."""
    s = pd.Series(list(texts)).astype(str)
    n_chars  = s.str.len().fillna(0).values.astype(np.float32)
    ratio_up = (s.str.count(r"[A-Z]") / pd.Series(n_chars).clip(lower=1)).fillna(0).values.astype(np.float32)
    has_url  = s.str.contains(r"https?://|www\.|bit\.ly", case=False, na=False).astype(int).values.astype(np.float32)
    log_excl = np.log1p(s.str.count(r"!").fillna(0).values).astype(np.float32)
    spam_kw  = s.apply(lambda x: 1 if SPAM_WORDS.search(x) else 0).values.astype(np.float32)
    return np.column_stack([n_chars, ratio_up, has_url, log_excl, spam_kw])


def features_activadas(text: str):
    s = str(text)
    n_chars = len(s)
    return [
        ("🔗 Contiene URL",          bool(re.search(r"https?://|www\.|bit\.ly", s, re.I)), "Enlace externo detectado"),
        ("📣 Palabras clave spam",   bool(SPAM_WORDS.search(s)),                           "Términos típicos de spam"),
        ("📏 Comentario muy largo",  n_chars > 150,                                         "Más de 150 caracteres"),
        ("🔠 Exceso de mayúsculas",  (len(re.findall(r"[A-Z]", s)) / max(n_chars, 1)) > 0.25, "Ratio mayúsculas > 25%"),
        ("❗ Muchas exclamaciones",  s.count("!") > 2,                                     "Más de 2 signos !"),
    ]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CARGA DE DATOS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    path = "Youtube-Spam-Dataset.csv"
    if not os.path.exists(path):
        st.error("❌ No se encontró 'Youtube-Spam-Dataset.csv'. Ejecuta la app desde la carpeta raíz del proyecto.")
        st.stop()
    df = pd.read_csv(path)
    df = df[["CONTENT", "CLASS"]].rename(columns={"CONTENT": "text", "CLASS": "label"})
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df[df["text"].str.len() > 2].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODELO — carga pre-entrenado o entrena y guarda
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    model_path   = os.path.join(MODEL_DIR, "spam_model.pkl")
    tfidf_path   = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")

    # ── Intentar cargar modelo guardado ──────────────────────────────────────
    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        try:
            mlp   = joblib.load(model_path)
            tfidf = joblib.load(tfidf_path)
            metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
            return mlp, tfidf, metrics
        except Exception:
            pass  # Si falla la carga, entrenar de nuevo

    # ── Entrenar modelo ───────────────────────────────────────────────────────
    df = load_data()

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"],
        test_size=0.20, random_state=42, stratify=df["label"]
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=2,
    )
    tfidf.fit(X_tr)

    def build_X(texts):
        Xt = tfidf.transform(texts).toarray().astype(np.float32)
        Xe = engineer_features(texts.values if hasattr(texts, "values") else list(texts))
        return np.hstack([Xt, Xe])

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        verbose=False,
    )
    mlp.fit(build_X(X_tr), y_tr)

    y_pred = mlp.predict(build_X(X_te))
    y_prob = mlp.predict_proba(build_X(X_te))[:, 1]

    metrics = {
        "accuracy":  float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_te, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_te, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_te, y_prob)),
        "model_type": "MLP Clasificador (Red Neuronal Superficial)",
        "architecture": "5005 → 256 → 128 → 64 → 1",
        "tfidf_features": 5000,
        "eng_features": 5,
        "y_test": y_te.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }

    # ── Guardar artefactos ────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(mlp,   model_path)
    joblib.dump(tfidf, tfidf_path)
    metrics_save = {k: v for k, v in metrics.items() if k not in ("y_test", "y_pred", "y_prob")}
    with open(metrics_path, "w") as f:
        json.dump(metrics_save, f, indent=2)

    return mlp, tfidf, metrics


def predict_single(text: str, model, tfidf) -> float:
    Xt = tfidf.transform([text]).toarray().astype(np.float32)
    Xe = engineer_features([text])
    return float(model.predict_proba(np.hstack([Xt, Xe]))[0][1])


def predict_batch(texts, model, tfidf) -> np.ndarray:
    Xt = tfidf.transform(list(texts)).toarray().astype(np.float32)
    Xe = engineer_features(list(texts))
    return model.predict_proba(np.hstack([Xt, Xe]))[:, 1]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# COMPONENTES VISUALES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def gauge_chart(prob: float) -> go.Figure:
    color = "#ff4455" if prob >= 0.5 else "#00ff88"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": "#ff4455"}, "decreasing": {"color": "#00ff88"}},
        number={"suffix": "%", "font": {"size": 52, "color": color}},
        title={"text": "Probabilidad de SPAM", "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                     "tickfont": {"color": "#64748b", "size": 11}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(30,32,64,0.6)",
            "borderwidth": 1, "bordercolor": "rgba(0,212,255,0.2)",
            "steps": [
                {"range": [0,  30], "color": "rgba(0,255,136,0.07)"},
                {"range": [30, 70], "color": "rgba(255,140,66,0.05)"},
                {"range": [70, 100], "color": "rgba(255,68,85,0.07)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.85, "value": 50},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"}, height=260,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def bar_features_chart(features_info) -> go.Figure:
    names  = [f[0] for f in features_info]
    vals   = [1 if f[1] else 0 for f in features_info]
    colors = ["#ff4455" if v else "rgba(148,163,184,0.3)" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=["DETECTADO" if v else "no detectado" for v in vals],
        textposition="inside", textfont={"size": 11, "color": "white"},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,32,64,0.4)",
        font={"color": "#94a3b8"}, height=210,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis={"visible": False, "range": [0, 1.2]},
        yaxis={"tickfont": {"size": 12, "color": "#e2e8f0"}},
    )
    return fig


def confusion_heatmap(cm: np.ndarray) -> go.Figure:
    labels = ["No Spam", "Spam"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}", textfont={"size": 24, "color": "white"},
        colorscale=[[0, "#1a1a35"], [0.5, "#004d80"], [1, "#00d4ff"]],
        showscale=False,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8", "size": 13}, height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis={"title": "Predicción", "titlefont": {"color": "#94a3b8"}},
        yaxis={"title": "Real",       "titlefont": {"color": "#94a3b8"}},
    )
    return fig


def roc_chart(y_test, y_prob, auc: float) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, fill="tozeroy", fillcolor="rgba(0,212,255,0.1)",
        line={"color": "#00d4ff", "width": 2.5},
        name=f"Modelo (AUC = {auc:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line={"color": "#475569", "dash": "dash", "width": 1.5},
        name="Azar (AUC = 0.500)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,32,64,0.4)",
        font={"color": "#94a3b8"}, height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        legend={"bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(0,212,255,0.2)", "borderwidth": 1},
        xaxis={"title": "Falsos Positivos",       "gridcolor": "rgba(255,255,255,0.06)"},
        yaxis={"title": "Verdaderos Positivos",   "gridcolor": "rgba(255,255,255,0.06)"},
    )
    return fig


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(metrics: dict, df: pd.DataFrame):
    with st.sidebar:
        st.markdown("### 🛡️ Spam Detector IA")
        st.markdown("---")

        st.markdown("""
        <div class="stat-box">
          <div class="stat-lbl">Arquitectura</div>
          <div class="stat-val" style="font-size:0.95rem">256 → 128 → 64 → 1</div>
        </div>
        <div class="stat-box">
          <div class="stat-lbl">Features de entrada</div>
          <div class="stat-val" style="font-size:0.95rem">TF-IDF (5k) + EDA (5)</div>
        </div>
        """, unsafe_allow_html=True)

        if metrics:
            acc = metrics.get("accuracy", 0)
            f1  = metrics.get("f1", 0)
            auc = metrics.get("auc_roc", 0)
            st.markdown(f"""
            <div class="stat-box"><div class="stat-lbl">Accuracy</div>
              <div class="stat-val">{acc:.1%}</div></div>
            <div class="stat-box"><div class="stat-lbl">F1-Score</div>
              <div class="stat-val">{f1:.3f}</div></div>
            <div class="stat-box"><div class="stat-lbl">AUC-ROC</div>
              <div class="stat-val">{auc:.3f}</div></div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        n_spam   = int(df["label"].sum())
        n_nospam = len(df) - n_spam
        st.markdown(f"""
        <div class="stat-box"><div class="stat-lbl">Comentarios totales</div>
          <div class="stat-val">{len(df):,}</div></div>
        <div class="stat-box"><div class="stat-lbl">Spam / No Spam</div>
          <div class="stat-val" style="font-size:1rem">
            <span style="color:#ff4455">{n_spam:,}</span> /
            <span style="color:#00ff88">{n_nospam:,}</span>
          </div></div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem;color:#64748b;line-height:1.7;">
        <b style="color:#94a3b8">Dataset de entrenamiento:</b><br>
        • PSY · Katy Perry · LMFAO<br>
        • Eminem · Shakira<br>
        • 1 956 comentarios reales<br><br>
        <b style="color:#94a3b8">UPV — Proyecto I: Intro IA</b><br>
        Ángel · Samuel · Artur · Pablo
        </div>
        """, unsafe_allow_html=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def main():
    with st.spinner("📊 Cargando dataset de YouTube..."):
        df = load_data()

    with st.spinner("🧠 Inicializando red neuronal... (solo la primera vez)"):
        model, tfidf, metrics = load_or_train_model()

    render_sidebar(metrics, df)

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1 class="hero-title">YouTube Spam Detector</h1>
      <p class="hero-sub">Red Neuronal Superficial · TF-IDF + Feature Engineering · Clasificación Automática</p>
      <span class="hero-badge">🎓 Proyecto IA · UPV · 2026</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "ðŸ”  Analizar Comentario",
        "ðŸ“Š  AnÃ¡lisis Masivo (CSV)",
        "ðŸ“ˆ  Rendimiento del Modelo",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — ANÁLISIS INDIVIDUAL
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            st.markdown("#### ✍️ Introduce un comentario")

            # Manejo de ejemplos rápidos con session_state
            if "texto_demo" not in st.session_state:
                st.session_state.texto_demo = ""

            comentario = st.text_area(
                label="",
                value=st.session_state.texto_demo,
                placeholder="Escribe o pega aquí un comentario de YouTube...",
                height=160,
                key="texto_input",
            )

            st.markdown("<p style='font-size:0.82rem;margin-bottom:0.3rem;color:#94a3b8'>Ejemplos rápidos:</p>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🚨 Cargar SPAM", use_container_width=True):
                    st.session_state.texto_demo = np.random.choice(EJEMPLOS_SPAM)
                    st.rerun()
            with c2:
                if st.button("✅ Cargar real", use_container_width=True):
                    st.session_state.texto_demo = np.random.choice(EJEMPLOS_REAL)
                    st.rerun()

            analizar = st.button("ðŸ” Analizar Comentario", use_container_width=True)

        with col_out:
            texto_a_analizar = comentario.strip()

            if analizar and texto_a_analizar:
                prob    = predict_single(texto_a_analizar, model, tfidf)
                is_spam = prob >= 0.5
                clase   = "spam" if is_spam else "safe"
                icono   = "🚨" if is_spam else "✅"
                label   = "SPAM" if is_spam else "NO ES SPAM"

                st.markdown(f"""
                <div class="verdict {clase}">
                  <span class="verdict-icon">{icono}</span>
                  <span class="verdict-label">{label}</span>
                  <span class="verdict-prob">Confianza: {prob:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(gauge_chart(prob), use_container_width=True,
                                config={"displayModeBar": False})

                feats  = features_activadas(texto_a_analizar)
                n_act  = sum(1 for f in feats if f[1])
                st.markdown(f"**Indicadores de spam detectados: {n_act} / {len(feats)}**")
                st.plotly_chart(bar_features_chart(feats), use_container_width=True,
                                config={"displayModeBar": False})

                pills = "".join(
                    f'<span class="pill {"pill-on" if f[1] else "pill-off"}">{f[0]}</span>'
                    for f in feats
                )
                st.markdown(f'<div class="pills">{pills}</div>', unsafe_allow_html=True)

                # Tooltip descripción
                if n_act > 0:
                    activos_desc = " · ".join(f[2] for f in feats if f[1])
                    st.markdown(f"""
                    <div class="card" style="margin-top:1rem;padding:1rem;">
                      <span style="color:#94a3b8;font-size:0.82rem;">
                      <b style="color:#ff4455">Señales detectadas:</b> {activos_desc}
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

            elif analizar:
                st.warning("âš ï¸ Por favor, introduce un comentario antes de analizar.")
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:3rem 1rem;">
                  <div style="font-size:3rem;margin-bottom:1rem;">👈</div>
                  <p>Escribe un comentario y pulsa <b style="color:#00d4ff">Analizar</b>
                  para obtener el resultado de la red neuronal.</p>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — ANÁLISIS MASIVO
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### ðŸ“¤ Sube un archivo CSV para anÃ¡lisis masivo")
        st.markdown("""
        <div class="card">
          <p>El CSV debe contener una columna llamada <code>CONTENT</code>, <code>comment_text</code>
          o <code>text</code> con los comentarios a analizar.<br>
          La columna <code>CLASS</code> es opcional — si existe, se muestra para comparar con las predicciones.</p>
        </div>
        """, unsafe_allow_html=True)

        archivo = st.file_uploader("Arrastra o selecciona tu CSV", type=["csv"])

        if archivo:
            try:
                df_up = pd.read_csv(archivo)
                col_txt = next(
                    (c for c in df_up.columns if c.upper() in ["CONTENT", "COMMENT_TEXT", "TEXT"]),
                    None,
                )
                if col_txt is None:
                    st.error("âŒ No se encontrÃ³ una columna 'CONTENT', 'comment_text' o 'text'.")
                else:
                    df_up["text"] = df_up[col_txt].astype(str).str.strip()
                    df_up = df_up[df_up["text"].str.len() > 2].reset_index(drop=True)
                    st.info(f"📋 **{len(df_up):,} comentarios cargados.** Procesando con la red neuronal...")

                    barra = st.progress(0, text="Analizando...")
                    probs, BATCH = [], 256
                    for i in range(0, len(df_up), BATCH):
                        batch_texts = df_up["text"].iloc[i:i + BATCH].tolist()
                        probs.extend(predict_batch(batch_texts, model, tfidf).tolist())
                        barra.progress(min((i + BATCH) / len(df_up), 1.0),
                                       text=f"Procesando {min(i+BATCH, len(df_up))}/{len(df_up)}...")
                    barra.empty()

                    df_up["spam_prob"]  = probs
                    df_up["prediccion"] = (df_up["spam_prob"] >= 0.5).map({True: "🚨 SPAM", False: "✅ No Spam"})

                    n_spam = (df_up["spam_prob"] >= 0.5).sum()
                    n_real = len(df_up) - n_spam

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total", f"{len(df_up):,}")
                    k2.metric("🚨 Spam", f"{n_spam:,}")
                    k3.metric("✅ No Spam", f"{n_real:,}")
                    k4.metric("% Spam", f"{n_spam/len(df_up):.1%}")

                    c_pie, c_hist = st.columns(2)
                    with c_pie:
                        fig_pie = go.Figure(go.Pie(
                            labels=["ðŸš¨ SPAM", "âœ… No Spam"],
                            values=[n_spam, n_real],
                            marker_colors=["#ff4455", "#00ff88"],
                            hole=0.5,
                            textfont={"color": "white", "size": 13},
                        ))
                        fig_pie.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", height=250,
                            margin=dict(l=10, r=10, t=30, b=10),
                            legend={"font": {"color": "#94a3b8"}},
                            title={"text": "Distribución SPAM / No Spam",
                                   "font": {"color": "white", "size": 13}},
                        )
                        st.plotly_chart(fig_pie, use_container_width=True,
                                        config={"displayModeBar": False})

                    with c_hist:
                        fig_h = go.Figure(go.Histogram(
                            x=df_up["spam_prob"], nbinsx=40,
                            marker_color="#00d4ff", opacity=0.8,
                        ))
                        fig_h.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,32,64,0.4)",
                            height=250, margin=dict(l=10, r=10, t=30, b=10),
                            title={"text": "Distribución de probabilidades",
                                   "font": {"color": "white", "size": 13}},
                            font={"color": "#94a3b8"},
                            xaxis={"title": "P(spam)", "gridcolor": "rgba(255,255,255,0.05)"},
                            yaxis={"title": "Frecuencia", "gridcolor": "rgba(255,255,255,0.05)"},
                            shapes=[{"type": "line", "x0": 0.5, "x1": 0.5,
                                     "y0": 0, "y1": 1, "yref": "paper",
                                     "line": {"color": "#ff4455", "width": 2, "dash": "dash"}}],
                        )
                        st.plotly_chart(fig_h, use_container_width=True,
                                        config={"displayModeBar": False})

                    cols_show = ["text", "spam_prob", "prediccion"]
                    if "CLASS" in df_up.columns:
                        cols_show.insert(0, "CLASS")

                    st.dataframe(
                        df_up[cols_show].rename(columns={
                            "text": "Comentario", "spam_prob": "P(spam)", "prediccion": "Predicción"
                        }),
                        use_container_width=True, height=320,
                    )

                    csv_out = df_up[cols_show].to_csv(index=False)
                    st.download_button("⬇️ Descargar resultados CSV", csv_out,
                                       "resultados_spam.csv", "text/csv", use_container_width=True)

            except Exception as e:
                st.error(f"âŒ Error al procesar el archivo: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — MÉTRICAS
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### 📈 Rendimiento en el conjunto de test (20% datos)")

        if not metrics:
            st.info("ℹ️ Métricas no disponibles todavía.")
        else:
            acc  = metrics.get("accuracy",  0)
            prec = metrics.get("precision", 0)
            rec  = metrics.get("recall",    0)
            f1   = metrics.get("f1",        0)
            auc  = metrics.get("auc_roc",   0)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",  f"{acc:.2%}")
            m2.metric("Precision", f"{prec:.3f}")
            m3.metric("Recall",    f"{rec:.3f}")
            m4.metric("F1-Score",  f"{f1:.3f}")
            m5.metric("AUC-ROC",   f"{auc:.3f}")

            st.markdown("---")
            c_cm, c_roc = st.columns(2, gap="large")

            y_test_lst = metrics.get("y_test", [])
            y_pred_lst = metrics.get("y_pred", [])
            y_prob_lst = metrics.get("y_prob", [])

            if y_test_lst:
                y_ta = np.array(y_test_lst)
                y_pa = np.array(y_pred_lst)
                y_pr = np.array(y_prob_lst)

                with c_cm:
                    st.markdown("**Matriz de Confusión**")
                    cm = confusion_matrix(y_ta, y_pa)
                    st.plotly_chart(confusion_heatmap(cm), use_container_width=True,
                                    config={"displayModeBar": False})
                    tn, fp, fn, tp = cm.ravel()
                    c1, c2 = st.columns(2)
                    c1.metric("Verdaderos Positivos (TP)", f"{tp}")
                    c1.metric("Verdaderos Negativos (TN)", f"{tn}")
                    c2.metric("Falsos Positivos (FP)", f"{fp}")
                    c2.metric("Falsos Negativos (FN)", f"{fn}")

                with c_roc:
                    st.markdown("**Curva ROC**")
                    st.plotly_chart(roc_chart(y_ta, y_pr, auc), use_container_width=True,
                                    config={"displayModeBar": False})

                st.markdown("---")
                st.markdown("**Informe completo de clasificación**")
                report = classification_report(y_ta, y_pa,
                                               target_names=["No Spam", "Spam"],
                                               output_dict=True)
                st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

        st.markdown("---")
        st.markdown("**Arquitectura del Modelo**")
        arch_col, info_col = st.columns(2)

        with arch_col:
            st.markdown("""
            <div class="card">
              <b style="color:#00d4ff">Red Neuronal Superficial (MLP)</b><br><br>
              <code style="color:#a855f7">Input → 5 005 features</code><br>&emsp;↓<br>
              <code style="color:#00d4ff">Dense(256, ReLU) — capa 1</code><br>&emsp;↓<br>
              <code style="color:#00d4ff">Dense(128, ReLU) — capa 2</code><br>&emsp;↓<br>
              <code style="color:#00d4ff">Dense(64,  ReLU) — capa 3</code><br>&emsp;↓<br>
              <code style="color:#00ff88">Dense(1, Sigmoid) — salida</code>
            </div>
            """, unsafe_allow_html=True)

        with info_col:
            st.markdown("""
            <div class="card">
              <b style="color:#00d4ff">Features de Entrada (5 005 dim.)</b><br><br>
              <span style="color:#e2e8f0">• TF-IDF bigramas</span>
              <span style="color:#a855f7"> → 5 000 features</span><br>
              <span style="color:#e2e8f0">• Longitud en caracteres</span>
              <span style="color:#a855f7"> → 1 feature</span><br>
              <span style="color:#e2e8f0">• Ratio de mayúsculas</span>
              <span style="color:#a855f7"> → 1 feature</span><br>
              <span style="color:#e2e8f0">• Presencia de URL</span>
              <span style="color:#a855f7"> → 1 feature</span><br>
              <span style="color:#e2e8f0">• log(exclamaciones + 1)</span>
              <span style="color:#a855f7"> → 1 feature</span><br>
              <span style="color:#e2e8f0">• Palabras clave spam</span>
              <span style="color:#a855f7"> → 1 feature</span><br><br>
              <b>Regularización:</b>
              <span style="color:#94a3b8">L2 (α=1e-4) + Early Stopping</span>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

