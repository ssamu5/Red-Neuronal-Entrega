# 🧠 Detector de Spam en YouTube con Red Neuronal Superficial

> **Proyecto I — Introducción a la IA**  
> Universitat Politècnica de València · Curso 2025–2026  
> **Equipo:** Ángel del Campo · Samuel Litago · Artur Mas · Pablo

---

## 📋 Descripción

Sistema de detección de spam en comentarios de YouTube mediante una **Red Neuronal Superficial** (Shallow Neural Network). El modelo combina vectorización TF-IDF con features de ingeniería derivadas del análisis exploratorio de datos (EDA).

### Arquitectura del modelo

```
Entrada: TF-IDF (5 000 features) + 5 features EDA = 5 005 dimensiones
         │
         ▼
    Dense(256, ReLU) → BatchNorm → Dropout(0.30)
         │
         ▼
    Dense(128, ReLU) → BatchNorm → Dropout(0.20)
         │
         ▼
       Dense(64, ReLU)
         │
         ▼
    Dense(1, Sigmoid) → Probabilidad de spam [0, 1]
```

**Optimizador:** Adam (lr=0.001)  
**Regularización:** L2 (λ=1e-4) + Dropout + EarlyStopping + ReduceLROnPlateau  
**Loss:** Binary Crossentropy

---

## 📁 Estructura del repositorio

```
Red-Neuronal-Entrega/
│
├── 📓 notebooks/
│   └── 01_Entrenamiento_Red_Neuronal.ipynb   ← Pipeline completo de entrenamiento
│
├── 📓 02_Analisis_Univariante_Artur.ipynb    ← Análisis univariante (Sprint 2)
│
├── 🤖 app.py                                 ← App Streamlit interactiva
│
├── 📊 datos/
│   ├── Youtube-Spam-Dataset.csv              ← Dataset original (1 956 comentarios)
│   ├── Youtube-Spam-Dataset equilibrado.csv  ← Versión balanceada
│   └── YouTube Comments Dataset (45K).csv   ← Dataset extendido (45 005 comentarios)
│
├── 🗂️ model/                                 ← Carpeta para el modelo entrenado
│   ├── spam_model.keras                      ← Modelo Keras (generado tras entrenar)
│   ├── tfidf_vectorizer.pkl                  ← Vectorizador TF-IDF
│   └── metrics.json                          ← Métricas finales del modelo
│
├── 🐳 Dockerfile                             ← Contenedor Docker
├── 📄 requirements.txt                       ← Dependencias Python
└── 📖 README.md
```

---

## 📊 Datasets

| Dataset | Filas | Spam | No Spam | Fuente |
|---------|-------|------|---------|--------|
| Youtube-Spam-Dataset.csv | 1 956 | ~50% | ~50% | UCI / Kaggle |
| YouTube 45K Rows.csv | 45 005 | Variable | Variable | Kaggle |
| **Total unificado** | **~47 000** | — | — | — |

Los datos del dataset original provienen de comentarios reales de vídeos de YouTube de artistas como **PSY, LMFAO, Eminem, Shakira y Katy Perry**, etiquetados manualmente como spam (1) o legítimos (0).

---

## ⚙️ Instalación y uso

### Opción A — Ejecución directa (recomendado para probar)

```bash
# 1. Clonar el repositorio
git clone https://github.com/Angel247-coder/Red-Neuronal-Entrega.git
cd Red-Neuronal-Entrega

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar la app Streamlit
python -m streamlit run app.py
```

La app se abrirá en `http://localhost:8501` y entrenará la red neuronal automáticamente la primera vez.

---

### Opción B — Entrenar el modelo completo (Keras + GPU)

```bash
# 1. Instalar dependencias adicionales
pip install tensorflow

# 2. Abrir el notebook de entrenamiento
jupyter notebook notebooks/01_Entrenamiento_Red_Neuronal.ipynb
```

O en **Google Colab** (gratis, con GPU):

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Angel247-coder/Red-Neuronal-Entrega/blob/main/notebooks/01_Entrenamiento_Red_Neuronal.ipynb)

---

### Opción C — Docker

```bash
docker build -t spam-detector .
docker run -p 8501:8501 spam-detector
```

---

## 🖥️ App Streamlit

La aplicación interactiva incluye **3 pestañas**:

| Pestaña | Descripción |
|---------|-------------|
| 🔍 **Analizar Comentario** | Introduce un comentario y obtén la probabilidad de spam con un medidor visual |
| 📊 **Análisis Batch** | Sube un CSV y clasifica todos los comentarios a la vez |
| 📈 **Métricas del Modelo** | Visualiza accuracy, F1, AUC-ROC, curvas de entrenamiento y matriz de confusión |

---

## 🔬 Features de ingeniería (Sprint 2 — EDA)

| Feature | Descripción |
|---------|-------------|
| `longitud_chars` | Número de caracteres del comentario |
| `ratio_mayusculas` | Proporción de letras en mayúsculas |
| `contiene_url` | 1 si contiene enlace (http, www, bit.ly...) |
| `log_exclamaciones` | log(1 + nº de signos !) |
| `palabras_spam` | 1 si contiene palabras típicas de spam (subscribe, free, click...) |

---

## 🛡️ Medidas anti-overfitting

Con un dataset relativamente pequeño, se aplicaron múltiples capas de regularización:

1. **Arquitectura mínima** — 3 capas ocultas (256 → 128 → 64)
2. **Regularización L2** (λ = 1e-4) en cada capa densa
3. **Dropout** (0.30 y 0.20) entre capas ocultas
4. **EarlyStopping** sobre `val_loss` (paciencia=7)
5. **ReduceLROnPlateau** para ajuste fino del learning rate
6. **División estratificada** 70% train / 15% val / 15% test

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

## 🧱 Tecnologías utilizadas

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)

---

## 👥 Autores

| Nombre | Rol |
|--------|-----|
| Ángel del Campo | Arquitectura red neuronal, entrenamiento, app Streamlit |
| Samuel Litago | Integración pipeline, Docker, deployment |
| Artur Mas | Análisis exploratorio (EDA), análisis univariante |
| Pablo | Preprocesamiento de datos, ingeniería de features |

---

## 📚 Referencias

- [UCI YouTube Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection)
- [Kaggle — YouTube Comments 45K](https://www.kaggle.com/)
- Alberto, T. C., Lochter J. V., Almeida T. A. *TubeSpam: Comment Spam Filtering on YouTube*. ICMLA, 2015.
- Goodfellow, I. et al. *Deep Learning*. MIT Press, 2016.
