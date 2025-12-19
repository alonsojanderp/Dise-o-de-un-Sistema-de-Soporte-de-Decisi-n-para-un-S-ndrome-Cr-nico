# Fenotipado y Predicción de Severidad (Simulación en Python)

## Contexto Epidemiológico
Usted es parte del equipo de Bioestadística de un hospital que trata un **Síndrome Crónico complejo**. Se sospecha que existen **subgrupos de pacientes (fenotipos)** que responden de manera diferente a la terapia, por lo que se requiere:

1) **Descubrir fenotipos** (No Supervisado)  
2) **Predecir el Outcome de severidad** para una intervención temprana (Supervisado)

Este proyecto integra conceptos de **Validación**, **Modelos Complejos** y **Aprendizaje No Supervisado**.

---

## Requerimientos Técnicos
- Código principal: `fenotipado_y_prediccion.py`
- Dataset simulado: 15 variables (biomarcadores/síntomas) + outcome binario `Severidad`
- Escalado obligatorio con `StandardScaler`
- PCA a 2 componentes + gráfico PC1 vs PC2 coloreado por severidad
- KMeans con `n_clusters=3` sobre datos escalados (NO PCA)
- Modelo supervisado integrado: `RandomForestClassifier` usando el **Fenotipo** como predictor
- Métrica final: **AUC** en conjunto de prueba (Train/Test 70/30)

---

## Estructura del Repositorio (como está subido en GitHub)
- `README.md`
- `fenotipado_y_prediccion.py`
- `requirements.txt`
- `synthetic_fenotypes_dataset.csv`
- `pca_pc1_pc2_severidad.png`
- `roc_curve_random_forest.png`

> Nota: Las imágenes están en la **raíz** del repo (por eso las rutas son directas).

---

## Cómo ejecutar el proyecto
Instalar dependencias:

```bash
pip install -r requirements.txt
