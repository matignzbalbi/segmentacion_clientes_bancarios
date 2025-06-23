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
import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


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

#####################################################################################################
st.header("Análisis de los clusters.")
st.write("Ahora haremos una análisis de los clústeres dados por K-Prototypes (con PCA):")


# Cargar los datos
data = pd.read_csv("Grupo 2 - Clientes Bancarios.csv", sep='\t', encoding='utf-8')
data = data.dropna()
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
data = data.drop(columns=['Z_Revenue','Z_CostContact','ID'])
data["Age"] = 2025 - data["Year_Birth"]
data["Spent"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]
data["Children"] = data["Kidhome"] + data["Teenhome"]
estado_numerico = data["Marital_Status"].replace({
    "Single": 1, "Married": 2, "Together": 2, "Divorced": 1, "Widow": 1
}).astype(int)

data["Family_Size"] = estado_numerico + data["Children"]
data['TotalAcceptedCmp'] = data[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']].sum(axis=1)
data['NumTotalPurchases'] = data[['NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumDealsPurchases']].sum(axis=1)
data['Customer_Tenure'] = 2025 - data['Dt_Customer'].dt.year
data = data.drop(columns=['Dt_Customer','Year_Birth'])

# Simplificar categorías
data['Marital_Status'] = data['Marital_Status'].replace({
    'Absurd': 'Single', 'YOLO': 'Single', 'Alone': 'Single', 'Divorced': 'Single', 'Widow': 'Single'
})
data['Education'] = data['Education'].replace({
    'PhD': "Postgraduate", 'Graduation': "Graduate", '2n Cycle': "Postgraduate",
    "Master": "Postgraduate", "Basic": "Undergraduate"
})

# Separar columnas categóricas y numéricas
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
binary_cols = [col for col in data.columns if data[col].nunique() == 2 and data[col].dtype in [int, float]]
numerical_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns if col not in binary_cols]

# Escalar datos numéricos
scaler = RobustScaler()
numerical_scaled = scaler.fit_transform(data[numerical_cols])
numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_cols, index=data.index)

# Aplicar PCA
pca = PCA(n_components=6)
numerical_pca = pca.fit_transform(numerical_scaled_df)
pca_df = pd.DataFrame(numerical_pca, columns=[f'PC{i+1}' for i in range(6)], index=data.index)

# Combinar con variables categóricas y binarias
combined_data = pd.concat([pca_df, data[categorical_cols + binary_cols]], axis=1)
dfMatrix = combined_data.to_numpy()
catColumnsPos = [combined_data.columns.get_loc(col) for col in categorical_cols + binary_cols]

# K-Prototypes con PCA
kprototype = KPrototypes(n_jobs=-1, n_clusters=4, init='Huang', random_state=0)
clusters = kprototype.fit_predict(dfMatrix, categorical=catColumnsPos)

# Añadir clusters al DataFrame original
data['clusters'] = clusters

# Graficar distribución de clusters
data['clusters'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribución de Clusters')
plt.xlabel('Cluster')
plt.ylabel('Cantidad de Clientes')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
