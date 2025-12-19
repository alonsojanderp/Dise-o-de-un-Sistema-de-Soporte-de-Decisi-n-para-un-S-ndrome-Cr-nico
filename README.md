



[README_GITHUB.md](https://github.com/user-attachments/files/24259654/README_GITHUB.md)

# Fenotipado y Predicción de Severidad en Datos de Alta Dimensionalidad

## Contexto Epidemiológico
Usted es parte del equipo de Bioestadística de un hospital que trata un **Síndrome Crónico complejo**. 
Se sospecha que existen **subgrupos de pacientes (fenotipos)** que responden de manera diferente a la terapia.

El objetivo del proyecto es:
1) Descubrir fenotipos clínicos mediante aprendizaje no supervisado.
2) Predecir el outcome de severidad para apoyar una intervención temprana.

---

## Requerimientos Técnicos
- Script principal: `fenotipado_y_prediccion.py`
- 15 variables clínicas simuladas
- Outcome binario: Severidad (0 = Baja, 1 = Alta)
- Escalado obligatorio con `StandardScaler`
- PCA a 2 componentes
- Clustering KMeans (k = 3)
- Random Forest para predicción
- Métrica final: AUC

---

## Preparación del Dataset
- 600 pacientes simulados
- 15 biomarcadores/síntomas
- Dataset exportado como `synthetic_fenotypes_dataset.csv`

---

## Escalado y PCA

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_.sum())
```

PC1 + PC2 explican aproximadamente el **58.5%** de la varianza.

---

## PCA según Severidad

```python
import matplotlib.pyplot as plt

plt.figure()
for s in [0, 1]:
    mask = y.values == s
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.7, label=f"Severidad={s}")
plt.legend()
plt.show()
```

![PCA](pca_pc1_pc2_severidad.png)

Interpretación: Existe una separación parcial entre severidad alta y baja.

---

## Clustering con KMeans

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df["Fenotipo"] = kmeans.fit_predict(X_scaled)
```

Se identifican 3 fenotipos clínicos, evidenciando heterogeneidad del síndrome.

---

## Modelado Supervisado

```python
from sklearn.model_selection import train_test_split

X_model = df[features + ["Fenotipo"]]
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=0.3, stratify=y, random_state=42
)
```

---

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
```

---

## Evaluación

```python
from sklearn.metrics import roc_auc_score, roc_curve

y_proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(auc)
```

![ROC](roc_curve_random_forest.png)

AUC en test ≈ 0.60

---

## Reflexión Final
El uso de fenotipado no supervisado permite identificar subgrupos clínicos relevantes. 
Integrar esta información en modelos supervisados mejora la predicción temprana de severidad 
y apoya la toma de decisiones clínicas, siempre considerando validación y aspectos éticos.
