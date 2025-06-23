import pandas as pd
import streamlit as st
import numpy as np
from utils import cargar_datos
from utils import limpiar_datos
from utils import features
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")
st.markdown("<h1 style='color:#00BFFF;'>Generación de Insights.</h1>", unsafe_allow_html=True)
st.divider()

st.header("Análisis por modelo.")
st.write("En base a la evaluación vista en la solapa anterior vemos que no hay mucha diferencia entre los resultados de los modelos. " \
"Sin embargo, vemos que el modelo con mejor coeficiente de silueta es: `K-Prototype con PCA`. Por esta razón, es el modelo elegido para la creación de los clusters. Repasemos la comparación entre los coeficientes de silueta:")

silueta_por_modelo = pd.DataFrame({
    "Modelo": [
        "K-Prototype",
        "K-Prototype (PCA)",
        "K-Means",
        "K-Means (PCA)",
        "Agglomerative",
        "Agglomerative (PCA)"
    ],
    "Coef_Silueta": [
        0.245368,
        0.340683,
        0.212936,
        0.329097,
        0.192293,
        0.306680
    ]
})
st.dataframe(silueta_por_modelo)


df_silueta = pd.DataFrame(silueta_por_modelo)

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

ax = sns.barplot(
    x="Modelo",
    y="Coef_Silueta",
    data=df_silueta.sort_values("Coef_Silueta", ascending=False),
    palette=colors,
    linewidth=1
)
plt.title("Comparación de Coeficientes de Silueta por Modelo", fontsize=14, pad=20, fontweight="bold")
plt.xlabel("Modelo", fontsize=12, labelpad=10)
plt.ylabel("Coeficiente de Silueta", fontsize=12, labelpad=10)
plt.ylim(0, 0.6)

# Añadir etiquetas de valor en las barras
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha="center", va="center",
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=10
    )

# Mostrar gráfico
plt.tight_layout()
st.pyplot(plt.gcf())
