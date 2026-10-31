
import datetime
import pandas as pd
import numpy as np

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def run_pipeline():

    print("=" * 70)
    print("INICIANDO PIPELINE DE MACHINE LEARNING")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. CARGA DEL DATASET
    # ------------------------------------------------------------

    dataset_id = "codesignal/wine-quality"

    print(f"\nCargando dataset: {dataset_id}")

    hf_dataset = load_dataset(dataset_id, split="red")
    df = pd.DataFrame(hf_dataset)

    # Normalización de nombres de columnas
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]

    # Conversión de variables numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dimensiones originales
    filas_originales, columnas_originales = df.shape

    print(f"Registros cargados: {filas_originales}")
    print(f"Columnas originales: {columnas_originales}")

    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    # ------------------------------------------------------------
    # 2. PREPROCESAMIENTO
    # ------------------------------------------------------------

    # Variable objetivo excluida de las predictoras
    num_features = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != "quality"
    ]

    cat_features = df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features)
    ])

    print("\nPreprocesamiento configurado correctamente.")
    print("Variables predictoras:", num_features)

    # ------------------------------------------------------------
    # 3. MODELO DE REGRESIÓN LINEAL
    # ------------------------------------------------------------

    print("\nEntrenando modelo de Regresión Lineal...")

    X_reg = df.drop(columns=["quality"])
    y_reg = df["quality"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=42
    )

    reg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    reg_pipeline.fit(X_train_r, y_train_r)

    y_pred_r = reg_pipeline.predict(X_test_r)

    mae_r = mean_absolute_error(y_test_r, y_pred_r)
    rmse_r = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2_r = r2_score(y_test_r, y_pred_r)

    print("\nRESULTADOS DE REGRESIÓN LINEAL")
    print(f"MAE:  {mae_r:.4f}")
    print(f"RMSE: {rmse_r:.4f}")
    print(f"R²:   {r2_r:.4f}")

    # ------------------------------------------------------------
    # 4. CREACIÓN DE CATEGORÍAS DE CALIDAD
    # ------------------------------------------------------------

    def categorizar_calidad(quality):

        if quality <= 4:
            return "Baja"
        elif quality <= 6:
            return "Media"
        else:
            return "Alta"

    df["categoria_calidad"] = df["quality"].apply(
        categorizar_calidad
    )

    # ------------------------------------------------------------
    # 5. MODELO DE CLASIFICACIÓN RANDOM FOREST
    # ------------------------------------------------------------

    print("\nEntrenando modelo Random Forest...")

    # Se excluyen las variables objetivo para evitar Data Leakage
    X_cls = df.drop(
        columns=["quality", "categoria_calidad"]
    )

    y_cls = df["categoria_calidad"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls,
        y_cls,
        test_size=0.2,
        random_state=42,
        stratify=y_cls
    )

    cls_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ))
    ])

    cls_pipeline.fit(X_train_c, y_train_c)

    y_pred_c = cls_pipeline.predict(X_test_c)

    # Métricas de clasificación
    acc_c = accuracy_score(y_test_c, y_pred_c)

    prec_c = precision_score(
        y_test_c,
        y_pred_c,
        average="weighted",
        zero_division=0
    )

    rec_c = recall_score(
        y_test_c,
        y_pred_c,
        average="weighted",
        zero_division=0
    )

    f1_c = f1_score(
        y_test_c,
        y_pred_c,
        average="weighted",
        zero_division=0
    )

    labels = ["Baja", "Media", "Alta"]

    cm_c = confusion_matrix(
        y_test_c,
        y_pred_c,
        labels=labels
    )

    print("\nRESULTADOS DE CLASIFICACIÓN")
    print(f"Accuracy:  {acc_c:.4f}")
    print(f"Precision: {prec_c:.4f}")
    print(f"Recall:    {rec_c:.4f}")
    print(f"F1-Score:  {f1_c:.4f}")

    print("\nMatriz de confusión:")
    print(pd.DataFrame(
        cm_c,
        index=[f"Real {l}" for l in labels],
        columns=[f"Pred {l}" for l in labels]
    ))

    print("\nReporte de clasificación:")
    print(classification_report(
        y_test_c,
        y_pred_c,
        labels=labels,
        zero_division=0
    ))

    # ------------------------------------------------------------
    # 6. GENERACIÓN DEL INFORME
    # ------------------------------------------------------------

    print("\nGenerando informe...")

    now_str = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    distribucion = df["categoria_calidad"].value_counts()

    matriz_df = pd.DataFrame(
        cm_c,
        index=[f"Real {l}" for l in labels],
        columns=[f"Pred {l}" for l in labels]
    )

    reporte = f"""
======================================================================
RESUMEN AUTOMÁTICO DE EJECUCIÓN DE MODELOS DE MACHINE LEARNING
Fecha de ejecución: {now_str}
======================================================================

1. IDENTIFICACIÓN DEL DATASET Y PREPROCESAMIENTO
----------------------------------------------------------------------
- Dataset ID: codesignal/wine-quality (Hugging Face)
- Partición utilizada: red (vinos tintos)
- Enlace: https://huggingface.co/datasets/codesignal/wine-quality
- Registros totales: {filas_originales}
- Columnas originales: {columnas_originales}
- Variables predictoras: {len(num_features)}
- Variable objetivo para regresión: quality
- Preprocesamiento: imputación por mediana y escalado estándar.
- Prevención de Data Leakage: las variables objetivo se excluyen
  de las características predictoras y las transformaciones
  se ajustan únicamente con los datos de entrenamiento.

2. RESULTADOS DE REGRESIÓN LINEAL
----------------------------------------------------------------------
- Algoritmo: LinearRegression
- Variable objetivo: quality
- MAE: {mae_r:.4f}
- RMSE: {rmse_r:.4f}
- R²: {r2_r:.4f}

3. RESULTADOS DE CLASIFICACIÓN (RANDOM FOREST)
----------------------------------------------------------------------
- Algoritmo: RandomForestClassifier
- Variable objetivo: categoria_calidad
- Accuracy: {acc_c:.4f}
- Precision: {prec_c:.4f}
- Recall: {rec_c:.4f}
- F1-Score: {f1_c:.4f}

4. DISTRIBUCIÓN DE LAS CATEGORÍAS
----------------------------------------------------------------------
{distribucion.to_string()}

5. MATRIZ DE CONFUSIÓN
----------------------------------------------------------------------
{matriz_df.to_string()}

6. CONCLUSIONES
----------------------------------------------------------------------
- La regresión lineal explicó el {r2_r * 100:.2f}% de la variabilidad
  en la calidad del vino.

- Random Forest obtuvo una exactitud del {acc_c * 100:.2f}%,
  pero su desempeño varía entre las categorías de calidad.

- El desbalance en las categorías influye en los resultados
  de clasificación.

- Ambos modelos permiten analizar la calidad del vino mediante
  predicciones numéricas y categóricas.

======================================================================
FIN DEL INFORME
======================================================================
"""

    with open(
        "resumen_modelos.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(reporte)

    print("\n¡Informe generado exitosamente!")
    print("Archivo: resumen_modelos.txt")


if __name__ == "__main__":
    run_pipeline()
