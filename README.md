
# Actividad 2: Modelos de Predicción y Clasificación con Machine Learning

## 👥 Autores

- Justin Beckham Cardona
- Lina Seguro
- Jorge Rodriguez

---

## 📌 Dataset utilizado

- **Nombre:** Wine Quality - Red Wine (vinos tintos).
- **Identificador en Hugging Face:** `codesignal/wine-quality`.
- **Partición utilizada:** `red`.
- **URL:** https://huggingface.co/datasets/codesignal/wine-quality
- **Registros:** 1.599.
- **Columnas originales:** 12.
- **Variable objetivo:** `quality`.

El conjunto de datos contiene características fisicoquímicas de vinos tintos, como acidez fija, acidez volátil, ácido cítrico, azúcar residual, cloruros, densidad, pH, sulfatos y alcohol. Estas variables permiten analizar y predecir la calidad del vino.

---

## ⚙️ Desarrollo del proyecto

El proyecto implementa dos modelos de Machine Learning:

### 1. Regresión lineal

Se utiliza el algoritmo `LinearRegression` para predecir la puntuación numérica de calidad del vino (`quality`).

**Métricas obtenidas:**

| Métrica | Resultado |
|---|---:|
| MAE | 0.5035 |
| RMSE | 0.6245 |
| R² | 0.4032 |

### 2. Clasificación con Random Forest

Se utiliza `RandomForestClassifier` para clasificar los vinos en tres categorías, según su puntuación de calidad:

- **Baja:** calidad menor o igual a 4.
- **Media:** calidad entre 5 y 6.
- **Alta:** calidad superior a 6.

**Métricas obtenidas:**

| Métrica | Resultado |
|---|---:|
| Accuracy | 84.69 % |
| Precision | 80.01 % |
| Recall | 84.69 % |
| F1-Score | 80.75 % |

La distribución desigual de las categorías influye en el desempeño del clasificador. Aunque la exactitud general es del 84.69 %, el modelo presenta dificultades para identificar los vinos de calidad baja y alta.

---

## 🧹 Preprocesamiento

Se aplican las siguientes operaciones:

- Normalización de los nombres de las columnas.
- Conversión de variables numéricas.
- Imputación de posibles valores faltantes mediante la mediana.
- Estandarización de las variables predictoras mediante `StandardScaler`.
- Separación de los datos en conjuntos de entrenamiento y prueba.
- Exclusión de las variables objetivo de las características predictoras para evitar Data Leakage.

---

## 🚀 Ejecución local

1. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt