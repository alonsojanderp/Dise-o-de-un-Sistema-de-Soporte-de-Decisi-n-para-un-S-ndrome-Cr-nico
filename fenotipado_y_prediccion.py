
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

RANDOM_STATE = 42

def simulate_high_dimensional_clinical_data(n_samples=600, n_features=15, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    latent_pheno = rng.choice([0,1,2], size=n_samples, p=[0.35,0.40,0.25])
    centers = np.array([
        rng.normal(-0.8, 0.4, n_features),
        rng.normal(0.0, 0.4, n_features),
        rng.normal(0.9, 0.4, n_features)
    ])
    rho = 0.55
    cov = np.fromfunction(lambda i,j: rho ** abs(i-j), (n_features,n_features))
    L = np.linalg.cholesky(cov)
    X = np.zeros((n_samples,n_features))
    for i in range(n_samples):
        X[i] = centers[latent_pheno[i]] + 0.8*(rng.normal(0,1,n_features) @ L)
    beta = np.zeros(n_features)
    beta[[0,3,7,10,13]] = [0.9,-0.7,0.8,-0.6,0.7]
    pheno_effect = np.array([-0.6,0.0,0.7])[latent_pheno]
    logits = (X @ beta)/3 + pheno_effect + rng.normal(0,0.6,n_samples)
    p = 1/(1+np.exp(-logits))
    y = rng.binomial(1,p)
    cols = [f"X{i+1:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["Severidad"] = y.astype(int)
    return df

def main():
    df = simulate_high_dimensional_clinical_data()
    features = [c for c in df.columns if c.startswith("X")]
    X = df[features]
    y = df["Severidad"]

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print("Varianza PC1+PC2:", pca.explained_variance_ratio_.sum())

    os.makedirs("figures", exist_ok=True)
    plt.figure()
    for s in [0,1]:
        m = y.values==s
        plt.scatter(X_pca[m,0], X_pca[m,1], alpha=0.7, label=f"Severidad={s}")
    plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("figures/pca_pc1_pc2_severidad.png", dpi=200)
    plt.close()

    df["Fenotipo"] = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init="auto").fit_predict(X_scaled)

    X_model = df[features + ["Fenotipo"]]
    Xtr,Xte,ytr,yte = train_test_split(X_model,y,test_size=0.3,stratify=y,random_state=RANDOM_STATE)

    pre = ColumnTransformer([
        ("num", StandardScaler(), features),
        ("cat", OneHotEncoder(), ["Fenotipo"])
    ])
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", pre), ("rf", rf)])
    pipe.fit(Xtr,ytr)

    proba = pipe.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte,proba)
    print("AUC:", auc)

    fpr,tpr,_ = roc_curve(yte,proba)
    plt.figure()
    plt.plot(fpr,tpr,label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig("figures/roc_curve_random_forest.png", dpi=200)
    plt.close()

    df.to_csv("synthetic_fenotypes_dataset.csv", index=False)

if __name__ == "__main__":
    main()
