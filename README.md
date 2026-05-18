# 🧠 Detector de Spam en YouTube — Red Neuronal Superficial

> **Proyecto I — Introducción a la IA**  
> Universitat Politècnica de València · Curso 2025–2026  
> **Equipo:** Ángel del Campo · Samuel Litago · Artur Mas · Pablo

---

## 📋 Descripción

Sistema de detección automática de spam en comentarios de YouTube mediante una **Red Neuronal Superficial** (Shallow Neural Network). El modelo combina vectorización **TF-IDF** con features de ingeniería derivadas del análisis exploratorio de datos (Sprint 2).

---

## 🏗️ Arquitectura del modelo

```
Entrada: TF-IDF (5 000 features) + 5 features EDA (RobustScaler) = 5 005 dimensiones
         │
         ▼
    Dense(256, ReLU) → BatchNorm → Dropout(0.30)   [L2=1e-4]
         │
         ▼
    Dense(128, ReLU) → BatchNorm → Dropout(0.20)   [L2=1e-4]
         │
         ▼
       Dense(64, ReLU)
         │
         ▼
    Dense(1, Sigmoid) → Probabilidad de spam [0, 1]
```

| Parámetro | Valor |
|-----------|-------|
| Optimizador | Adam (lr=0.001) |
| Loss | Binary Crossentropy |
| Regularización | L2 (λ=1e-4) + Dropout + EarlyStopping (paciencia=7) |
| Preprocesado texto | EDA features → RobustScaler + TF-IDF word bigramas |
| División | 70% train / 15% val / 15% test (estratificada) |

---

## 📁 Estructura del repositorio

```
Red-Neuronal-Entrega/
│
├── 📓 notebooks/
│   └── 01_Entrenamiento_Red_Neuronal.ipynb   ← Pipeline completo Keras (Sprint 3)
│
├── 📓 02_Analisis_Univariante_Artur.ipynb    ← EDA y análisis univariante (Sprint 2)
│
├── 🤖 app.py                                 ← App Streamlit interactiva (3 pestañas)
│
├── 📊 [Datos]
│   ├── Youtube-Spam-Dataset.csv              ← 1 956 comentarios (PSY, LMFAO, Eminem,
│   │                                            Katy Perry, Shakira)
│   ├── Youtube-Spam-Dataset equilibrado.csv  ← Versión balanceada
│   └── YouTube Comments Dataset (45K).csv   ← Dataset extendido (45 005 comentarios)
│
├── 🗂️ model/                                 ← Artefactos del modelo (generados al entrenar)
│   ├── spam_model.keras                      ← Modelo Keras
│   ├── tfidf_vectorizer.pkl                  ← Vectorizador TF-IDF
│   ├── eda_scaler.pkl                        ← RobustScaler EDA (necesario para inferencia)
│   └── metrics.json                          ← Métricas finales
│
├── 🐳 Dockerfile                             ← Docker con imagen Alpine
├── 📄 requirements.txt                       ← Dependencias Python
├── 🔧 .streamlit/config.toml                 ← Configuración Streamlit
└── 📖 README.md
```

---

## 📊 Dataset

Los datos provienen del **UCI YouTube Spam Collection**, con comentarios reales de 5 vídeos musicales:

| Artista | Comentarios |
|---------|-------------|
| PSY — Gangnam Style | ~350 |
| LMFAO — Party Rock Anthem | ~350 |
| Eminem — Love The Way You Lie | ~350 |
| Katy Perry — Just Dance | ~350 |
| Shakira — Waka Waka | ~350 |

**Total:** 1 956 comentarios · Spam: 51.38% · No Spam: 48.62%  
*(Balance de solo 2.76% — sin necesidad de técnicas adicionales, Sprint 2 §2.1)*

---

## 🔬 Features de ingeniería (Sprint 2 — EDA)

| Feature | Descripción | Correlación CLASS | Decisión Sprint 2 |
|---------|-------------|-------------------|-------------------|
| `palabras_spam` | Palabras gancho: subscribe, free, click... | r = 0.69 | ✅ Incluida |
| `longitud_chars` | Número de caracteres | r = 0.34 | ✅ Incluida |
| `contiene_url` | 1 si contiene http/www/bit.ly | r = 0.33 | ✅ Incluida |
| `ratio_mayusculas` | Proporción de mayúsculas | r = 0.10 | ✅ Incluida |
| `log_exclamaciones` | log(1 + nº de "!") | r = 0.05 | ✅ Incluida (zero-inflated) |
| `longitud_palabras` | Número de palabras | r = 0.91 con chars | ❌ **ELIMINADA** (§4.2) |

**Escalado:** `RobustScaler` (mediana+IQR) — justificado por distribuciones asimétricas (skewness: 3.14, 2.95, 37.36, Sprint 2 §4.1).

---

## 🐛 Bugs corregidos respecto al documento original

| # | Sección | Problema | Solución |
|---|---------|----------|----------|
| 1 | Sprint 2 §5.5 | `scaler.fit_transform(df[...])` sobre **todo** el dataset → **data leakage** | `fit()` solo en train; `transform()` en val/test |
| 2 | Sprint 2 §4.2 | `longitud_palabras` incluida a pesar de r=0.91 con `longitud_chars` | Eliminada del vector de features |
| 3 | Sprint 2 §4.1 | `RobustScaler` mencionado pero no implementado en el pipeline | Implementado y guardado como `eda_scaler.pkl` |
| 4 | Sprint 1 | Columna `Artista` referenciada pero no existe en los CSV | Creada automáticamente al concatenar |

---

## ⚙️ Instalación y uso

### Opción A — Ejecución directa

```bash
git clone https://github.com/ssamu5/Red-Neuronal-Entrega.git
cd Red-Neuronal-Entrega
pip install -r requirements.txt
python -m streamlit run app.py
```

La app abre en `http://localhost:8501` y entrena automáticamente la primera vez (~2 min).

### Opción B — Notebook completo (Keras + GPU)

```bash
jupyter notebook notebooks/01_Entrenamiento_Red_Neuronal.ipynb
```

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssamu5/Red-Neuronal-Entrega/blob/main/notebooks/01_Entrenamiento_Red_Neuronal.ipynb)

### Opción C — Docker (Alpine)

```bash
docker build -t spam-detector .
docker run -p 8501:8501 spam-detector
```

> Imagen basada en `python:3.10-alpine` — ligera y segura.

---

## 🖥️ App Streamlit

| Pestaña | Descripción |
|---------|-------------|
| 🔍 **Analizar Comentario** | Texto libre → probabilidad de spam con medidor visual |
| 📊 **Análisis Batch** | Sube CSV → clasifica todos los comentarios |
| 📈 **Métricas del Modelo** | Accuracy, F1, AUC-ROC, matriz de confusión, curva ROC |

---

## 🛡️ Medidas anti-overfitting

1. **Arquitectura mínima** — 3 capas (256 → 128 → 64)
2. **Regularización L2** (λ = 1e-4) en capas densas
3. **Dropout** (0.30 y 0.20) entre capas ocultas
4. **EarlyStopping** sobre `val_loss` (paciencia = 7)
5. **ReduceLROnPlateau** — lr ÷ 2 si no mejora en 3 épocas
6. **ModelCheckpoint** — guarda solo el mejor según `val_auc`
7. **División estratificada** 70% / 15% / 15%

---

## 📈 Resultados esperados

| Métrica | Valor esperado |
|---------|---------------|
| Accuracy | ~95–97% |
| Precision | ~94–96% |
| Recall | ~95–97% |
| F1-Score | ~95–97% |
| AUC-ROC | ~0.98–0.99 |

---

## 🧱 Tecnologías

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Alpine-2496ED?logo=docker&logoColor=white)

---

## 👥 Equipo

| Nombre | Contribución |
|--------|-------------|
| **Ángel del Campo** | Arquitectura red neuronal, entrenamiento Keras, EDA multivariante |
| **Samuel Litago** | Ingeniería de features, pipeline de datos, Docker, deployment |
| **Artur Mas** | Análisis univariante (Sprint 2), detección de outliers, notebook EDA |
| **Pablo** | Planificación (PERT/Gantt), preprocesamiento, tratamiento de ausentes |

---

## 📚 Referencias

- Alberto, T. C., Lochter J. V., Almeida T. A. *TubeSpam: Comment Spam Filtering on YouTube*. ICMLA, 2015.
- [UCI YouTube Spam Collection](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection)
- Goodfellow, I. et al. *Deep Learning*. MIT Press, 2016.
- Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR, 2011.
