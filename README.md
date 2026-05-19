# 🛡️ YouTube Spam Detector — Red Neuronal

**Proyecto I: Introducción a la IA**
**Universitat Politècnica de València — 2026**

Detector automático de spam en comentarios de YouTube mediante una **Red Neuronal Superficial (MLP)** entrenada con comentarios reales de 5 artistas (PSY, Katy Perry, LMFAO, Eminem, Shakira).

---

## Demo

La aplicación está desplegada en **Streamlit Cloud**:
> **[Ver demo online →](https://share.streamlit.io/)**

---

## Funcionalidades

| Función | Descripción |
|---|---|
| **Análisis individual** | Escribe o pega un comentario y obtén una predicción instantánea con probabilidad |
| **Análisis masivo (CSV)** | Sube un CSV con comentarios y analiza todos de golpe |
| **Indicadores de spam** | Visualiza qué señales detectó el modelo (URL, palabras clave, mayúsculas...) |
| **Métricas del modelo** | Consulta accuracy, F1, AUC-ROC, matriz de confusión y curva ROC |

---

## Stack tecnológico

- **Modelo:** `sklearn.MLPClassifier` — Red Neuronal Superficial (256→128→64→1)
- **Vectorización:** TF-IDF bigramas (5 000 features) + 5 features EDA
- **Frontend:** Streamlit
- **Contenerización:** Docker
- **Despliegue:** Streamlit Community Cloud

---

## Arquitectura del modelo

```
Input: 5 005 features
  ├── TF-IDF bigramas (5 000): representación textual
  └── Features EDA (5): longitud, mayúsculas, URL, exclamaciones, spam-keywords

Red Neuronal:
  Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1, Sigmoid)

Regularización: L2 (α=1e-4) + Early Stopping (patience=10)
Optimizador:    Adam (lr=0.001)
```

---

## Cómo ejecutar localmente

### Opción 1: Python directo

```bash
# Clonar repositorio
git clone https://github.com/ssamu5/Red-Neuronal-Entrega.git
cd Red-Neuronal-Entrega

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la app
streamlit run app.py
```

> El modelo se entrena automáticamente en el primer arranque (≈ 1 min) y se guarda en `model/` para ejecuciones posteriores.

### Opción 2: Docker

```bash
# Construir imagen
docker build -t spam-detector .

# Ejecutar contenedor
docker run -p 8501:8501 spam-detector
```

Acceder en: `http://localhost:8501`

---

## Entrenar el modelo en Google Colab

1. Abrir el notebook: `notebooks/01_Entrenamiento_Red_Neuronal.ipynb`
2. Subir los CSV a Colab Files
3. Ejecutar todas las celdas en orden
4. Descargar la carpeta `model/` generada
5. Copiar los artefactos al repositorio

Ver `DESCRIPCION_ENTRENAMIENTO.md` para detalles completos del proceso y las mejoras respecto al libro de referencia.

---

## Estructura del repositorio

```
📦 Red-Neuronal-Entrega/
├── app.py                          # Frontend Streamlit
├── Dockerfile                      # Contenedor Docker
├── requirements.txt                # Dependencias Python
├── DESCRIPCION_ENTRENAMIENTO.md    # Doc. técnica del entrenamiento
├── Youtube-Spam-Dataset.csv        # Dataset unificado (1 956 comentarios)
├── .streamlit/
│   └── config.toml                 # Configuración del tema
├── model/
│   ├── spam_model.pkl              # Modelo entrenado
│   ├── tfidf_vectorizer.pkl        # Vectorizador TF-IDF
│   └── metrics.json                # Métricas de evaluación
└── notebooks/
    └── 01_Entrenamiento_Red_Neuronal.ipynb   # Notebook Colab
```

---

## Dataset

- **Fuente:** YouTube Spam Collection (5 artistas), proporcionado por la asignatura
- **Tamaño:** 1 956 comentarios reales
- **Balance:** 51.4% spam / 48.6% no spam
- **Columnas:** `COMMENT_ID`, `AUTHOR`, `DATE`, `CONTENT`, `CLASS` (1=spam, 0=no spam)

---

## Equipo

| Miembro | Rol |
|---|---|
| Ángel del Campo Hernández | Análisis multivariante, stakeholders, unificación CSV |
| Samuel Litago Vallejo | Ingeniería de features, detección outliers, modelo y frontend |
| Artur Mas Llorca | Análisis univariante, desglose de tareas, seguimiento |
| Pablo | Planificación, PERT, Gantt |
