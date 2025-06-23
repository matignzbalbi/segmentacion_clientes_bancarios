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
st.markdown("<h1 style='color:#00BFFF;'>Resultados.</h1>", unsafe_allow_html=True)
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
st.divider()

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

# Paso 2: Reemplazo seguro con fillna
estado_numerico = data['Marital_Status'].replace({
    "Single": 1,
    "Married": 2,
    "Together": 3,
    "Divorced": 1,
    "Widow": 1,
    "Alone": 1,
    "Absurd": 1,
    "YOLO": 1
}).fillna(1)

# Paso 3: Conversión segura
estado_numerico = pd.to_numeric(estado_numerico, errors='coerce').fillna(1).astype(int)

# Finalmente calcular Family_Size
data["Family_Size"] = estado_numerico + data["Children"]


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

#Nos quedamos con los clientes que tengan un salario < 120000
data = data[data['Income']<120000]

#Nos quedamos con los clientes que tengan < 90
data = data[data['Age']<90]

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats

num_vars = data.select_dtypes(include=np.number).columns.tolist()

iqr_flags = pd.DataFrame(index=data.index)

for col in num_vars:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_flags[col] = (data[col] < lower) | (data[col] > upper)

data['is_outlier_IQR'] = iqr_flags.any(axis=1)

z_scores = np.abs(stats.zscore(data[num_vars]))
data['is_outlier_Z'] = (z_scores > 3).any(axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[num_vars])
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
data['is_outlier_LOF'] = lof.fit_predict(X_scaled) == -1


data['outlier_todos'] = data['is_outlier_IQR'] & data['is_outlier_Z'] & data['is_outlier_LOF']


def replace_confirmed_outliers_with_median(df, column):
    median = df[column].median()
    df.loc[df['outlier_todos'], column] = df.loc[df['outlier_todos'], column].apply(lambda x: median)
    return df

for col in num_vars:
    data = replace_confirmed_outliers_with_median(data, col)

data = data.drop(columns=["is_outlier_LOF", "outlier_todos", "is_outlier_IQR", "is_outlier_Z"])

data = data[["Education", "Marital_Status", "Income", "Recency", "Age",
 "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds",
 "TotalAcceptedCmp", 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases','NumWebVisitsMonth']]

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

# Estilo de fondo negro y configuración general
sns.set_style("darkgrid")
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'

# Función para aplicar estilo consistente

def aplicar_estilo(ax, title=None, xlabel=None, ylabel=None, rotate_xticks=False):
    if title:
        ax.set_title(title, fontsize=14, color='white')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, color='white')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, color='white')
    if rotate_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('black')

# 1. Distribución de Clusters
st.subheader("Distribución de Clusters")
fig, ax = plt.subplots()
data['clusters'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Distribución de Clusters')
ax.set_xlabel('Cluster')
ax.set_ylabel('Cantidad de Clientes')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
st.pyplot(plt.gcf())

# 2. Scatterplot: Income vs Spent
st.subheader("Relación entre Income y Spent por Cluster")
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Income', y='MntMeatProducts', data=data, hue='clusters', palette='Set2')
plt.title('Income vs Spent')
plt.ylabel('Spent')
st.pyplot(plt.gcf())

# 3. Scatterplots: Income vs los gastos en los distintos tipos de categorías.
st.subheader("Relación entre Income y los gastos en los distintos tipos de categorías.")
mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts','MntSweetProducts', 'MntGoldProds']
fig, ax = plt.subplots(2, 3, figsize=(24, 12))
for i, col in enumerate(mnt_cols):
    row = i // 3
    col_idx = i % 3
    sns.scatterplot(x='Income', y=(data[col]), data=data, hue='clusters', palette='Set2', ax=ax[row, col_idx])
    ax[row, col_idx].set_title(f'Amount of {col.replace("Mnt", "")} (Original Scale)')
plt.tight_layout()
st.pyplot(plt.gcf())

# 4. Histogramas: Age
st.subheader("Distribución de Edad por Cluster")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i in range(data['clusters'].nunique()):
    sns.histplot(data[data['clusters'] == i]['Age'], bins=30, kde=True, ax=axes[i], color=plt.cm.Set2(i))
    mean_Age = data[data['clusters'] == i]['Age'].mean()
    axes[i].axvline(mean_Age, color='white', linestyle='--', label=f'Media: {mean_Age:.2f}')
    axes[i].text(mean_Age + 0.1, axes[i].get_ylim()[1]*0.8, f'{mean_Age:.2f}', color='white', fontsize=12)
    axes[i].set_title(f'Distribución de Age para Cluster {i}')
for j in range(data['clusters'].nunique(), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
st.pyplot(plt.gcf())

# 5. Histogramas: Income
st.subheader("Distribución de Income por Cluster")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i in range(data['clusters'].nunique()):
    sns.histplot(data[data['clusters'] == i]['Income'], bins=30, kde=True, ax=axes[i], color=plt.cm.Set2(i))
    mean_Income = data[data['clusters'] == i]['Income'].mean()
    axes[i].axvline(mean_Income, color='white', linestyle='--', label=f'Media: {mean_Income:.2f}')
    axes[i].text(mean_Income + 0.1, axes[i].get_ylim()[1]*0.8, f'{mean_Income:.2f}', color='white', fontsize=12)
    axes[i].set_title(f'Distribución de Income para Cluster {i}')
for j in range(data['clusters'].nunique(), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
st.pyplot(plt.gcf())

# 6. Histogramas: NumWebVisitsMonth
st.subheader("Distribución de NumWebVisitsMonth por Cluster")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i in range(data['clusters'].nunique()):
    sns.histplot(data[data['clusters'] == i]['NumWebVisitsMonth'], bins=30, kde=True, ax=axes[i], color=plt.cm.Set2(i))
    mean_NumWebVisitsMonth = data[data['clusters'] == i]['NumWebVisitsMonth'].mean()
    axes[i].axvline(mean_NumWebVisitsMonth, color='white', linestyle='--', label=f'Media: {mean_NumWebVisitsMonth:.2f}')
    axes[i].text(mean_NumWebVisitsMonth + 0.1, axes[i].get_ylim()[1]*0.8, f'{mean_NumWebVisitsMonth:.2f}', color='white', fontsize=12)
    axes[i].set_title(f'Distribución de NumWebVisitsMonth para Cluster {i}')
for j in range(data['clusters'].nunique(), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
st.pyplot(plt.gcf())

# 7. Marital Status
st.subheader("Estado Civil por Cluster")
marital_status_by_cluster = data.groupby('clusters')['Marital_Status'].value_counts(normalize=True).unstack().fillna(0)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i in range(data['clusters'].nunique()):
    cluster_data = marital_status_by_cluster.loc[i]
    cluster_data.plot(kind='bar', ax=axes[i], color=sns.color_palette('Set2', len(cluster_data.index)))
    axes[i].set_title(f'Distribución de Marital Status para Cluster {i}')
    axes[i].tick_params(axis='x', rotation=45)
for j in range(data['clusters'].nunique(), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
st.pyplot(plt.gcf())

# 8. Education
st.subheader("Nivel Educativo por Cluster")
education_by_cluster = data.groupby('clusters')['Education'].value_counts(normalize=True).unstack().fillna(0)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i in range(data['clusters'].nunique()):
    cluster_data = education_by_cluster.loc[i]
    cluster_data.plot(kind='bar', ax=axes[i], color=sns.color_palette('Set2', len(cluster_data.index)))
    axes[i].set_title(f'Distribución de Education para Cluster {i}')
    axes[i].tick_params(axis='x', rotation=45)
for j in range(data['clusters'].nunique(), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
st.pyplot(plt.gcf())

# 9. Income Promedio
st.subheader("Promedio de Income por Cluster")
Income_by_cluster = data.groupby('clusters')['Income'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='clusters', y='Income', data=Income_by_cluster, palette='Set2')
plt.title('Income por Cluster')
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
st.pyplot(plt.gcf())

# 10. NumWebVisitsMonth Promedio
st.subheader("Promedio de NumWebVisitsMonth por Cluster")
NumWebVisitsMonth_by_cluster = data.groupby('clusters')['NumWebVisitsMonth'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='clusters', y='NumWebVisitsMonth', data=NumWebVisitsMonth_by_cluster, palette='Set2')
plt.title('NumWebVisitsMonth por Cluster')
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
st.pyplot(plt.gcf())

# 11. Recency Promedio
st.subheader("Recency Promedio por Cluster")
Recency_by_cluster = data.groupby('clusters')['Recency'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='clusters', y='Recency', data=Recency_by_cluster, palette='Set2')
plt.title('Recency por Cluster')
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
st.pyplot(plt.gcf())

# 12. TotalAcceptedCmp Promedio
st.subheader("Promedio de TotalAcceptedCmp por Cluster")
TotalAcceptedCmp_by_cluster = data.groupby('clusters')['TotalAcceptedCmp'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='clusters', y='TotalAcceptedCmp', data=TotalAcceptedCmp_by_cluster, palette='Set2')
plt.title('TotalAcceptedCmp por Cluster')
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
st.pyplot(plt.gcf())

# 13. Age Promedio
st.subheader("Promedio de Edad por Cluster")
Age_by_cluster = data.groupby('clusters')['Age'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='clusters', y='Age', data=Age_by_cluster, palette='Set2')
plt.title('Age por Cluster')
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
st.pyplot(plt.gcf())

# 14. Promedios Numéricos por Cluster
st.subheader("Promedios de las distintas variables numéricas por Cluster")
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
cluster_avg = data.groupby('clusters')[numeric_cols].mean()
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.barplot(x=cluster_avg.index, y=cluster_avg[col], palette='viridis')
    plt.title(f'Promedio de {col} por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(f'Promedio de {col}')
    st.pyplot(plt.gcf())
