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

df = cargar_datos()

st.title("Evaluación del Modelo.")
st.divider()
st.write("""Como se detallo en 4. Modelamiento, se realizaron dos modelos con K-Prototypes donde uno incluyé **PCA** y el otro no para 
         las variables numéricas.

En este apartado vamos a detallar los **resultados individuales** de cada modelo y compararlos.
    """)


st.divider()

st.header("K-Prototypes")
st.write("En principio, preparamos los datos para los k-prototype. Obtuvimos los indices de las columnas categoricas en el conjuntod de datos combinados." \
"Luego realizamos un analisis de clustering utilizando el modelo y evaluamos la calidad de los clusteres." \
"Por ultimo, mostramos los resultado en un grafico utilizando el metodo del codo y otro utilizando el indice de silueta.")

codigo = '''from kmodes.kprototypes import KPrototypes
dfMatrix = combined_data.to_numpy()'''
st.code(codigo)
from kmodes.kprototypes import KPrototypes

# Preparar los datos para los K-Prototypes
st.write("Preparamos los datos para K-Prototype")

codigo = '''catColumnsPos = [combined_data.columns.get_loc(col) for col in categorical_cols.tolist() + binary_cols]
print('\nColumnas categóricas           : {}'.format(categorical_cols.tolist() + binary_cols))
print('Posición de las columnas categóricas  : {}'.format(catColumnsPos))'''
st.code(codigo)
st.write("Obtenemos los indices de las columnas categoricas")

codigo = '''from sklearn.metrics import silhouette_score
cost = []
sil_scores = []
# Rango de k
for x in range(2, 6):
    kprototype = KPrototypes(n_jobs=-1, n_clusters=x, init='Huang', random_state=0)

    # Ajustar el modelo y predecir los clusters
    clusters = kprototype.fit_predict(dfMatrix, categorical=catColumnsPos)

    # Almacenar el coste (costo total)
    cost.append(kprototype.cost_)

    # Calcular el Silhouette Score usando la distancia euclidiana para los clusters
    sil_score = silhouette_score(numerical_data_pca, clusters, metric='euclidean')
    sil_scores.append(sil_score)

    print(f'Inicio de la agrupación para k={x}: {clusters[:10]}')  # Imprimir los primeros 10 clusters para abreviar'''
st.code(codigo)
st.write("En este codigo estamos evaluando la calidad de clusters para datos mixtos usando kprototypes, determinando el numero optimo de clusters mediante costo y el silhouette score")
import streamlit as st
import matplotlib.pyplot as plt

# Simulación de datos (reemplazá por tus listas reales)
cost = [16500, 15000, 13600, 12800]
sil_scores = [0.33, 0.23, 0.23, 0.17]

# Crear figura
fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
plt.subplots_adjust(wspace=0.3)

# Subplot 1 - Método del Codo
axs[0].plot(range(2, 6), cost, marker='o', color='tomato')
axs[0].set_title('Método del Codo (Coste vs k)', color='white')
axs[0].set_xlabel('Número de Clusters (k)', color='white')
axs[0].set_ylabel('Coste', color='white')
axs[0].set_xticks(range(2, 6))
axs[0].tick_params(colors='white')
axs[0].set_facecolor('black')

# Subplot 2 - Índice de Silueta
axs[1].plot(range(2, 6), sil_scores, marker='o', color='lime')
axs[1].set_title('Índice de Silueta (Silhouette Score vs k)', color='white')
axs[1].set_xlabel('Número de Clusters (k)', color='white')
axs[1].set_ylabel('Índice de Silueta', color='white')
axs[1].set_xticks(range(2, 6))
axs[1].tick_params(colors='white')
axs[1].set_facecolor('black')

# Mostrar en Streamlit
st.pyplot(fig)

st.write("En el primer grafico del Metodo del Codo podemos identificar el valor optimo de k; buscando el punto donde la disminucion del coste se vuelve menos significativa. En este caso, aproximandamente en k=3 o k=4, lo que sugiere que estas opciones pueden representar una segmentacion adecuada." \
"El índice de silueta mide la calidad de los clusters. Aunque k=2 presenta el valor más alto, lo cual indica mejor cohesión y separación, también puede ser una segmentación demasiado general. Se observa que a medida que aumenta k,el índice de silueta disminuye, lo cual sugiere una mayor dispersión o superposición entre grupos. La elección entre k = 2, k = 3 o k = 4 dependera del equilibrio entre simplicidad y riqueza interpretativa.")




st.divider()
st.header("K-Means")
codigo = '''from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

education_cols = ["Education"]
marital_cols = ["Marital_Status"]

numerical_cols_all = data.select_dtypes(include=["int", "float"]).columns.tolist()

numerical_cols = [col for col in numerical_cols_all if col not in education_cols + marital_cols]

ct = ColumnTransformer(
    [
        ("education", OrdinalEncoder(), education_cols),
        ("marital", LabelEncoder(), marital_cols), # LabelEncoder devuelve 1D, pero ColumnTransformer lo maneja internamente
        ("num", RobustScaler(), numerical_cols),
    ],
    remainder='passthrough' # Mantener las columnas no especificadas
)

data_transformed = data.copy()

# Aplicar OrdinalEncoder a Education
ordinal_encoder = OrdinalEncoder()
data_transformed['Education'] = ordinal_encoder.fit_transform(data_transformed[education_cols])

# Aplicar LabelEncoder a Marital_Status
label_encoder = LabelEncoder()
data_transformed['Marital_Status'] = label_encoder.fit_transform(data_transformed['Marital_Status'])

# Aplicar RobustScaler a las columnas numéricas
scaler = RobustScaler()
data_transformed[numerical_cols] = scaler.fit_transform(data_transformed[numerical_cols])
data_transformed'''
st.code(codigo)
st.write("Aplicamos transformaciones para poder trabajar los datos, utilizamos ordinalencoder para asignar valores numericos segun un orden establecido. Y LabelEncoder para asignar un entero a cada categoria distinta." \
"Utilizamos RobustScaler para escalar las columnas numericas ya que habian valores atipicos.")
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler , StandardScaler

education_cols = ["Education"]
marital_cols = ["Marital_Status"]

numerical_cols_all = df.select_dtypes(include=["int", "float"]).columns.tolist()

numerical_cols = [col for col in numerical_cols_all if col not in education_cols + marital_cols]

ct = ColumnTransformer(
    [
        ("education", OrdinalEncoder(), education_cols),
        ("marital", LabelEncoder(), marital_cols), # LabelEncoder devuelve 1D, pero ColumnTransformer lo maneja internamente
        ("num", RobustScaler(), numerical_cols),
    ],
    remainder='passthrough' # Mantener las columnas no especificadas
)

data_transformed = df.copy()

# Aplicar OrdinalEncoder a Education
ordinal_encoder = OrdinalEncoder()
data_transformed['Education'] = ordinal_encoder.fit_transform(data_transformed[education_cols])

# Aplicar LabelEncoder a Marital_Status
label_encoder = LabelEncoder()
data_transformed['Marital_Status'] = label_encoder.fit_transform(data_transformed['Marital_Status'])

# Aplicar RobustScaler a las columnas numéricas
scaler = RobustScaler()
data_transformed[numerical_cols] = scaler.fit_transform(data_transformed[numerical_cols])
data_transformed




st.divider()
st.header("K-Means sin PCA")
codigo = '''kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
predict_kmeans = kmeans.fit_predict(data_transformed)
clusters_df = pd.DataFrame(predict_kmeans)

silueta_kmeans = silhouette_score(data_transformed, clusters_df)
print(round(silueta_kmeans, 4))'''
st.code(codigo)
st.write("0.2189")

codigo = '''# Aplicar PCA a características numéricas
pca = PCA(n_components=None)
numerical_data_pca = pca.fit_transform(data_transformed)

# Calcular el ratio de varianza explicada y la varianza explicada acumulada
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)'''
st.code(codigo)
st.write("Aplicamos PCA a las numericas y luego calculamos el ratio de varianza explicada y la varianza explicada acumulada")


# Datos de ejemplo (reemplaza con tus datos reales)
componentes = np.arange(1, 11)
varianza_explicada = np.array([0.4, 0.3, 0.15, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01])
varianza_acumulada = np.cumsum(varianza_explicada)

# Crear dos columnas para los gráficos
col1, col2 = st.columns(2)

# Gráfico 1: Scree Plot
with col1:
    st.header("Scree Plot")
    
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig1.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    
    # Configuración del gráfico
    ax1.plot(componentes, varianza_explicada, 'cyan', marker='o', linestyle='-', linewidth=2, markersize=8, label='Varianza explicada')
    ax1.set_title('Scree Plot', color='white', fontsize=14)
    ax1.set_xlabel('Componente principal', color='white')
    ax1.set_ylabel('Razón de varianza explicada', color='white')
    
    # Configurar colores de ejes
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    for spine in ax1.spines.values():
        spine.set_color('white')
    
    ax1.grid(color='gray', linestyle='--', linewidth=0.5)
    ax1.legend(facecolor='black', labelcolor='white')
    
    st.pyplot(fig1)

# Gráfico 2: Varianza Acumulada
with col2:
    st.header("Varianza Acumulada")
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.patch.set_facecolor('black')
    ax2.set_facecolor('black')
    
    # Configuración del gráfico
    ax2.plot(componentes, varianza_acumulada, 'magenta', marker='o', linestyle='-', linewidth=2, markersize=8, label='Varianza acumulada')
    ax2.set_title('Varianza Acumulada Explicada', color='white', fontsize=14)
    ax2.set_xlabel('Número de componentes', color='white')
    ax2.set_ylabel('Varianza acumulada explicada', color='white')
    
    # Configurar colores de ejes
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    for spine in ax2.spines.values():
        spine.set_color('white')
    
    ax2.grid(color='gray', linestyle='--', linewidth=0.5)
    ax2.legend(facecolor='black', labelcolor='white')
    
    st.pyplot(fig2)

# Información adicional
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .stMarkdown, .stHeader {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
st.info("En el segundo grafico podemos observcar como aumenta la varianza explicada a medida que añadimos mas componentes.")
st.divider()



st.header("PCA con 9 componentes")
codigo = '''pca = PCA(n_components=9)
numerical_data_pca = pca.fit_transform(data_transformed)

# Crear un dataFrame para las características numéricas transformadas PCA
pca_columns = [f'PC{i+1}' for i in range(numerical_data_pca.shape[1])]
numerical_pca_df = pd.DataFrame(numerical_data_pca, columns=pca_columns, index=data_transformed.index)'''
st.code(codigo)

st.header("K-Means con PCA")
codigo = '''from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
predict_kmeans_pca = kmeans.fit_predict(numerical_data_pca)
clusters_df = pd.DataFrame(predict_kmeans)

silueta_kmeans_pca = silhouette_score(numerical_data_pca, clusters_df)
print(round(silueta_kmeans_pca, 4))'''
st.code(codigo)
st.info("0.2858")

