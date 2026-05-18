# Detector de Spam en YouTube - Red Neuronal Superficial

> **Proyecto I - Introduccion a la IA**
> Universitat Politecnica de Valencia - Curso 2025-2026
> **Equipo:** Angel del Campo - Samuel Litago - Artur Mas

---

## Descripcion

Sistema de deteccion automatica de spam en comentarios de YouTube mediante una **Red Neuronal Superficial** (Shallow Neural Network). El modelo combina vectorizacion **TF-IDF** con features de ingenieria derivadas del analisis exploratorio de datos (Sprint 2).

---

## Arquitectura del modelo

```
Entrada: TF-IDF (5 000 features) + 5 features EDA (RobustScaler) = 5 005 dimensiones
         |
         v
    Dense(256, ReLU) -> BatchNorm -> Dropout(0.30)   [L2=1e-4]
         |
         v
    Dense(128, ReLU) -> BatchNorm -> Dropout(0.20)   [L2=1e-4]
         |
         v
       Dense(64, ReLU)
         |
         v
    Dense(1, Sigmoid) -> Probabilidad de spam [0, 1]
```

| Parametro | Valor |
|-----------|-------|
| Optimizador | Adam (lr=0.001) |
| Loss | Binary Crossentropy |
| Regularizacion | L2 (lambda=1e-4) + Dropout + EarlyStopping (paciencia=7) |
| Preprocesado texto | EDA features -> RobustScaler + TF-IDF word bigramas |
| Division | 70% train / 15% val / 15% test (estratificada) |

---

## Estructura del repositorio

```
Red-Neuronal-Entrega/
|
|-- notebooks/
|   +-- 01_Entrenamiento_Red_Neuronal.ipynb   <- Pipeline completo Keras (Sprint 3)
|
|-- 02_Analisis_Univariante_Artur.ipynb       <- EDA y analisis univariante (Sprint 2)
|
|-- app.py                                    <- App Streamlit interactiva (3 pestanas)
|
|-- [Datos - colocar en la misma carpeta]
|   |-- Youtube-Spam-Dataset.csv              <- 1 956 comentarios originales
|   +-- YouTube Comments Dataset (45K).csv   <- Dataset extendido (45 005 comentarios)
|
|-- model/                                    <- Artefactos del modelo (generados al entrenar)
|   |-- spam_model.keras                      <- Modelo Keras
|   |-- tfidf_vectorizer.pkl                  <- Vectorizador TF-IDF
|   |-- eda_scaler.pkl                        <- RobustScaler EDA (necesario para inferencia)
|   +-- metrics.json                          <- Metricas finales
|
|-- Dockerfile                                <- Docker con imagen Alpine
|-- requirements.txt                          <- Dependencias Python
|-- .streamlit/config.toml                    <- Configuracion Streamlit
+-- README.md
```

---

## Dataset

Los datos provienen del **UCI YouTube Spam Collection**, con comentarios reales de 5 videos musicales:

| Artista | Comentarios |
|---------|-------------|
| PSY - Gangnam Style | ~350 |
| LMFAO - Party Rock Anthem | ~350 |
| Eminem - Love The Way You Lie | ~350 |
| Katy Perry - Just Dance | ~350 |
| Shakira - Waka Waka | ~350 |

**Total:** 1 956 comentarios - Spam: 51.38% / No Spam: 48.62%
*(Balance de solo 2.76% - sin necesidad de tecnicas adicionales, Sprint 2 seccion 2.1)*

---

## Features de ingenieria (Sprint 2 - EDA)

| Feature | Descripcion | Correlacion CLASS | Decision Sprint 2 |
|---------|-------------|-------------------|-------------------|
| `palabras_spam` | Palabras gancho: subscribe, free, click... | r = 0.69 | Incluida |
| `longitud_chars` | Numero de caracteres | r = 0.34 | Incluida |
| `contiene_url` | 1 si contiene http/www/bit.ly | r = 0.33 | Incluida |
| `ratio_mayusculas` | Proporcion de mayusculas | r = 0.10 | Incluida |
| `log_exclamaciones` | log(1 + numero de "!") | r = 0.05 | Incluida (zero-inflated) |
| `longitud_palabras` | Numero de palabras | r = 0.91 con chars | **ELIMINADA** (seccion 4.2) |

**Escalado:** `RobustScaler` (mediana+IQR) - justificado por distribuciones asimetricas
(skewness: 3.14, 2.95, 37.36 - Sprint 2 seccion 4.1).

---

## Bugs corregidos respecto al documento original

| # | Seccion | Problema | Solucion |
|---|---------|----------|----------|
| 1 | Sprint 2 seccion 5.5 | `scaler.fit_transform(df[...])` sobre todo el dataset -> **data leakage** | `fit()` solo en train; `transform()` en val/test |
| 2 | Sprint 2 seccion 4.2 | `longitud_palabras` incluida a pesar de r=0.91 con `longitud_chars` | Eliminada del vector de features |
| 3 | Sprint 2 seccion 4.1 | `RobustScaler` mencionado pero no implementado en el pipeline | Implementado y guardado como `eda_scaler.pkl` |
| 4 | Sprint 1 | Columna `Artista` referenciada pero no existe en los CSV | Creada automaticamente al concatenar |

---

## Instalacion y uso

### Opcion A - Ejecucion directa

```bash
git clone https://github.com/ssamu5/Red-Neuronal-Entrega.git
cd Red-Neuronal-Entrega
pip install -r requirements.txt
python -m streamlit run app.py
```

La app abre en `http://localhost:8501` y entrena automaticamente la primera vez (~2 min).

### Opcion B - Notebook completo (Keras + GPU)

```bash
jupyter notebook notebooks/01_Entrenamiento_Red_Neuronal.ipynb
```

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssamu5/Red-Neuronal-Entrega/blob/main/notebooks/01_Entrenamiento_Red_Neuronal.ipynb)

### Opcion C - Docker (Alpine)

```bash
docker build -t spam-detector .
docker run -p 8501:8501 spam-detector
```

> Imagen basada en `python:3.10-alpine` - ligera y segura.

---

## App Streamlit

| Pestana | Descripcion |
|---------|-------------|
| Analizar Comentario | Texto libre -> probabilidad de spam con medidor visual |
| Analisis Batch | Sube CSV -> clasifica todos los comentarios |
| Metricas del Modelo | Accuracy, F1, AUC-ROC, matriz de confusion, curva ROC |

---

## Medidas anti-overfitting

1. **Arquitectura minima** - 3 capas (256 -> 128 -> 64)
2. **Regularizacion L2** (lambda = 1e-4) en capas densas
3. **Dropout** (0.30 y 0.20) entre capas ocultas
4. **EarlyStopping** sobre `val_loss` (paciencia = 7)
5. **ReduceLROnPlateau** - lr dividido por 2 si no mejora en 3 epocas
6. **ModelCheckpoint** - guarda solo el mejor segun `val_auc`
7. **Division estratificada** 70% / 15% / 15%

---

## Resultados esperados

| Metrica | Valor esperado |
|---------|---------------|
| Accuracy | ~95-97% |
| Precision | ~94-96% |
| Recall | ~95-97% |
| F1-Score | ~95-97% |
| AUC-ROC | ~0.98-0.99 |

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Alpine-2496ED?logo=docker&logoColor=white)

---

## Equipo

| Nombre | Contribucion |
|--------|-------------|
| **Angel del Campo** | Arquitectura red neuronal, entrenamiento Keras, EDA multivariante |
| **Samuel Litago** | Ingenieria de features, pipeline de datos, Docker, deployment |
| **Artur Mas** | Analisis univariante (Sprint 2), deteccion de outliers, notebook EDA |

---

## Referencias

- Alberto, T. C., Lochter J. V., Almeida T. A. *TubeSpam: Comment Spam Filtering on YouTube*. ICMLA, 2015.
- [UCI YouTube Spam Collection](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection)
- Goodfellow, I. et al. *Deep Learning*. MIT Press, 2016.
- Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR, 2011.
