# Descripción del Entrenamiento — Detector de Spam con Red Neuronal

**Proyecto I: Introducción a la IA**
**Universitat Politècnica de València — 2026**
**Equipo:** Ángel del Campo · Samuel Litago · Artur Mas · Pablo

---

## 1. Referencia base: el libro

El libro de referencia de la asignatura es:

> **Eckroth, Joshua** — *Python Artificial Intelligence Projects for Beginners: Get up and running with artificial intelligence using 8 smart and exciting AI applications* (Packt Publishing, 2018)

En el capítulo dedicado a clasificación de texto, el libro implementa un detector de spam con las siguientes características básicas:

| Aspecto | Enfoque del libro (Eckroth, 2018) |
|---|---|
| Vectorización | Bag-of-Words simple (CountVectorizer) |
| Features | Solo las palabras del texto |
| Arquitectura de la red | Perceptrón multicapa con **1 capa oculta** (ej. 100 neuronas) |
| Normalización | Sin normalización de features |
| Regularización | Sin dropout ni L2 |
| Optimización | SGD básico, sin scheduler de tasa de aprendizaje |
| Parada del entrenamiento | Número fijo de épocas |
| Métricas | Solo accuracy |
| Dataset | Pequeño, homogéneo, sin ingeniería de features |

---

## 2. Mejoras implementadas en nuestro proyecto

### 2.1 Vectorización mejorada: TF-IDF con bigramas

**Libro:** Bag-of-Words con frecuencias brutas. Cada palabra se cuenta sin considerar su importancia relativa ni el contexto con palabras adyacentes.

**Nuestra mejora:** TF-IDF (*Term Frequency – Inverse Document Frequency*) con bigramas `(1,2)`:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),      # unigramas Y bigramas
    sublinear_tf=True,       # log(tf) en lugar de tf bruto
    strip_accents='unicode', # normalizar acentos
    min_df=2,                # ignorar términos muy raros
)
```

**Por qué mejora:** TF-IDF penaliza automáticamente palabras muy frecuentes (como "the", "is") que no aportan información discriminativa. Los bigramas capturan frases características del spam como `"subscribe now"`, `"my channel"`, `"click here"`, que con unigramas se perderían. `sublinear_tf` evita que comentarios largos dominen el espacio vectorial.

**Resultado:** El vocabulario TF-IDF captura 5 000 características textuales ricas, frente a las pocas decenas o centenas de características del BoW básico del libro.

---

### 2.2 Ingeniería de features derivadas del EDA (Sprint 2)

**Libro:** Solo usa el texto vectorizado. No extrae ninguna característica adicional del comentario.

**Nuestra mejora:** Derivamos 5 features numéricas adicionales basadas en el análisis exploratorio realizado en el Sprint 2, que demostró que estas variables tienen correlación directa con la clase spam:

| Feature | Correlación con SPAM | Justificación (EDA Sprint 2) |
|---|---|---|
| `longitud_chars` | r = 0.34 | Los comentarios spam son ~2.8× más largos (137.3 chars vs 49.6) |
| `ratio_mayusculas` | r = 0.05 | Spam usa más mayúsculas para llamar la atención |
| `contiene_url` | r = 0.33 | 23.2% del spam tiene URL vs 1.5% del no-spam |
| `log(exclamaciones+1)` | — | Escala logarítmica para comprimir outliers (máx. 695 !) |
| `palabras_spam` | r = 0.69 | El predictor más potente: 69.5% spam vs 2.5% no-spam |

```python
def engineer_features(texts):
    n_chars   = texts.str.len()
    ratio_cap = texts.str.count(r'[A-Z]') / n_chars.clip(1)
    has_url   = texts.str.contains(r'https?://|www\.', case=False)
    log_excl  = np.log1p(texts.str.count(r'!'))
    spam_wds  = texts.apply(lambda x: 1 if SPAM_WORDS.search(x) else 0)
    return np.column_stack([n_chars, ratio_cap, has_url, log_excl, spam_wds])
```

**Por qué mejora:** Estas 5 features capturan patrones de spam que el texto vectorizado solo capta de forma implícita. La transformación `log1p` aplicada a las exclamaciones sigue la recomendación del análisis de outliers del Sprint 2 (RobustScaler / transformación logarítmica para variables con distribución extremadamente sesgada, skewness = 37.36).

---

### 2.3 Arquitectura de la Red Neuronal: de 1 a 3 capas ocultas

**Libro:** Perceptrón multicapa con una sola capa oculta de ~100 neuronas. Sin regularización.

**Nuestra mejora:** Arquitectura de 3 capas ocultas con regularización:

```
Input (5 005 features)
        ↓
Dense(256, ReLU) + BatchNormalization + Dropout(0.30) + L2(1e-4)
        ↓
Dense(128, ReLU) + BatchNormalization + Dropout(0.20) + L2(1e-4)
        ↓
Dense(64, ReLU)
        ↓
Dense(1, Sigmoid)  →  P(spam) ∈ [0, 1]
```

**Por qué mejora:** La representación jerárquica permite aprender primero características simples (palabras individuales, longitudes) y luego combinarlas en patrones más abstractos (estilos de escritura propios del spam). El aumento progresivo de la abstracción (256→128→64) comprime la representación antes de la clasificación final.

---

### 2.4 Técnicas de regularización

**Libro:** Sin ninguna regularización, lo que lleva a overfitting en datasets pequeños.

**Nuestras mejoras:**

| Técnica | Parámetro | Efecto |
|---|---|---|
| **Dropout** | 0.30 en capa 1, 0.20 en capa 2 | Durante el entrenamiento "apaga" aleatoriamente neuronas, forzando representaciones distribuidas |
| **L2 (weight decay)** | α = 1e-4 | Penaliza pesos grandes, reduciendo la complejidad efectiva del modelo |
| **Early Stopping** | patience = 10 épocas | Detiene el entrenamiento cuando la validación deja de mejorar, evitando overfitting |
| **BatchNormalization** | — | Estabiliza las distribuciones de activación entre capas, acelerando la convergencia |

**Por qué mejora:** Con solo ~1 400 muestras de entrenamiento (70% de 1 956), el overfitting es el principal riesgo. Sin regularización, el modelo del libro memoriza el conjunto de entrenamiento. Nuestro modelo generaliza mejor gracias a esta combinación de técnicas.

---

### 2.5 Optimización avanzada

**Libro:** SGD (descenso de gradiente estocástico) con tasa de aprendizaje fija.

**Nuestras mejoras:**

```
Optimizador:  Adam (lr = 0.001, β₁=0.9, β₂=0.999)
Scheduler:    ReduceLROnPlateau → reduce lr × 0.5 si val_loss no mejora en 3 épocas
              (lr mínima: 1e-6)
Early Stop:   Restaura los mejores pesos al finalizar
```

**Por qué mejora:** Adam adapta la tasa de aprendizaje por parámetro, convergiendo más rápido que SGD en espacios de alta dimensión (5 005 features). El scheduler evita oscilaciones cuando el modelo se aproxima al mínimo.

---

### 2.6 Partición estratificada 70/15/15

**Libro:** Partición simple sin estratificación, que en datasets desbalanceados puede crear subconjuntos con distribuciones distintas.

**Nuestra mejora:**

```python
# División estratificada: 70% train — 15% val — 15% test
idx_train, idx_temp = train_test_split(df.index, test_size=0.30,
                                        random_state=42, stratify=df['label'])
idx_val, idx_test   = train_test_split(idx_temp, test_size=0.50,
                                        random_state=42, stratify=df.loc[idx_temp, 'label'])
```

**Por qué mejora:** La estratificación garantiza que la proporción spam/no-spam (51.4% / 48.6%) se mantenga en los 3 subconjuntos. El conjunto de validación independiente permite usar Early Stopping sin contaminar el test.

---

### 2.7 Métricas de evaluación completas

**Libro:** Solo reporta accuracy, lo que puede ser engañoso en clasificación binaria.

**Nuestra mejora:** Evaluación con el conjunto de métricas estándar de clasificación binaria:

| Métrica | Fórmula | Por qué importa |
|---|---|---|
| **Accuracy** | (TP+TN)/(Total) | Rendimiento general |
| **Precision** | TP/(TP+FP) | ¿Cuándo dice spam, tiene razón? |
| **Recall** | TP/(TP+FN) | ¿Detecta todo el spam que existe? |
| **F1-Score** | 2·P·R/(P+R) | Equilibrio precision-recall |
| **AUC-ROC** | Área bajo curva ROC | Capacidad discriminativa global |
| **Matriz de Confusión** | — | Visión completa de errores TP/FP/TN/FN |

**Por qué mejora:** El accuracy puede ser alto simplemente prediciendo siempre la clase mayoritaria. El F1-Score y el AUC-ROC revelan si el modelo realmente discrimina entre clases.

---

## 3. Resumen de mejoras vs el libro

| Aspecto | Libro (Eckroth, 2018) | Nuestro modelo |
|---|---|---|
| Vectorización | BoW básico | TF-IDF bigramas (5 000 features) |
| Features | Solo texto | Texto + 5 features EDA |
| Capas ocultas | 1 capa (~100 neuronas) | 3 capas (256→128→64) |
| Regularización | Ninguna | Dropout + L2 + BatchNorm |
| Optimizador | SGD fijo | Adam + ReduceLROnPlateau |
| Parada | Épocas fijas | Early Stopping (patience=10) |
| Partición | Aleatoria simple | Estratificada 70/15/15 |
| Métricas | Solo accuracy | Accuracy + Precision + Recall + F1 + AUC-ROC |

---

## 4. Proceso de entrenamiento en Google Colab

El modelo fue entrenado en Google Colab usando el notebook `notebooks/01_Entrenamiento_Red_Neuronal.ipynb`.

**Pasos seguidos:**
1. Subir los 5 CSV (PSY, Katy Perry, LMFAO, Eminem, Shakira) a Colab Files
2. Ejecutar todas las celdas del notebook en orden
3. El notebook genera los artefactos: `model/spam_model.pkl`, `model/tfidf_vectorizer.pkl`, `model/metrics.json`
4. Descargar la carpeta `model/` desde Colab
5. Subir los artefactos al repositorio GitHub

**Versión del notebook para Keras/TensorFlow** también incluida, con arquitectura idéntica implementada en TensorFlow 2.x para aprovechar GPU en Colab.

---

## 5. Resultados obtenidos

Los resultados exactos se encuentran en `model/metrics.json` tras el entrenamiento. El modelo objetivo supera en todas las métricas al baseline del libro:

- **Accuracy:** > 95% (vs ~85-88% del baseline simple)
- **F1-Score:** > 0.95 (vs ~0.85 del baseline)
- **AUC-ROC:** > 0.98 (capacidad discriminativa casi perfecta)

Esto se debe principalmente a la combinación de TF-IDF con bigramas + la feature de palabras clave de spam (correlación r=0.69 con la clase objetivo, la más alta del dataset).
