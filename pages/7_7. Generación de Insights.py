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

# Estilo de fondo negro
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6), facecolor='black')

# Colores inspirados en el otro gráfico
colors = ["tomato", "lime", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"]

# Barplot
ax = sns.barplot(
    x="Modelo",
    y="Coef_Silueta",
    data=df_silueta.sort_values("Coef_Silueta", ascending=False),
    palette=colors,
    linewidth=1
)

# Títulos y etiquetas blancas
ax.set_title("Comparación de Coeficientes de Silueta por Modelo", fontsize=14, pad=20, fontweight="bold", color='white')
ax.set_xlabel("Modelo", fontsize=12, labelpad=10, color='white')
ax.set_ylabel("Coeficiente de Silueta", fontsize=12, labelpad=10, color='white')
ax.set_ylim(0, 0.6)
ax.set_facecolor('black')
plt.gcf().patch.set_facecolor('black')
ax.tick_params(colors='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', color='white')

# Etiquetas en barras
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha="center", va="center",
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=10,
        color='white'
    )

# Mostrar en Streamlit
plt.tight_layout()
st.pyplot(plt.gcf())
