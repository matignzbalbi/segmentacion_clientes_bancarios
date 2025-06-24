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
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler , StandardScaler
from kmodes.kprototypes import KPrototypes



st.set_page_config(layout="wide")

df = cargar_datos()

st.markdown("<h1 style='color:#00BFFF;'>Evaluación del Modelo.</h1>", unsafe_allow_html=True)
st.divider()
st.write("""Como se detallo en 4. Modelamiento, se realizaron 3 modelos donde cada uno incluyé **PCA** y el otro no para 
         las variables numéricas.

En este apartado vamos a detallar los **resultados individuales** de cada modelo y compararlos.
    """)


st.divider()

##################### K-Prototypes #####################



st.header("K-Prototypes.")
st.write("""Cómo se menciono en 5. Modelamiento, K-Prototypes es una solución ideal para nuestro problema ya que se adapta fácilmente a nuestras variables 
         (categóricas y numéricas), en este apartado se detalla el código con los pasos que seguimos para ejecutar el modelo y preparar los datos para ejecutarlo.
         
Debajo también se encuentra el gráfico del método del codo y el índice silueta, los cuales sirvieron para identificar la cantidad ideal de clusters para nuestro problema.""")


# Preparar los datos para los K-Prototypes


codigo = '''from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes

dfMatrix = combined_data.to_numpy()

catColumnsPos = [combined_data.columns.get_loc(col) for col in categorical_cols.tolist() + binary_cols]
print('\nColumnas categóricas           : {}'.format(categorical_cols.tolist() + binary_cols))
print('Posición de las columnas categóricas  : {}'.format(catColumnsPos))

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
    
with st.expander("Ver código preparación de datos y gráficos para K-Prototypes"):
    st.code(codigo, language="python")


st.image("https://i.imgur.com/DnYpsIk.png")



st.write(
    "El gráfico del Método del Codo sugiere que *k = 3* o *k = 4* podrían ser valores adecuados, ya que la reducción del coste se vuelve menos significativa a partir de esos puntos. "
    "Aunque el índice de silueta es más alto para *k = 2*, lo que indica mejor separación, también puede resultar una segmentación demasiado general. "
    "La elección final dependerá del equilibrio entre simplicidad y nivel de detalle deseado."
)

st.subheader("Resultados para K-Prototypes sin PCA.")
st.markdown("Silhouette Score: **0.2453**")

##################### K-Prototypes con PCA #####################
st.divider()

st.header("K-Prototypes con PCA.")
st.write("Nuevamente para determinar la cantidad ideal de clusters y análizar el índice silueta se realizaron los siguientes gráficos:")
st.image("https://i.imgur.com/iRSQ7k5.png")


codigo_pcak = ''''
# Aplicar PCA con el número seleccionado de componentes
pca = PCA(n_components=6)
numerical_data_pca = pca.fit_transform(numerical_data_scaled)

# Crear un dataFrame para las características numéricas transformadas PCA
pca_columns = [f'PC{i+1}' for i in range(numerical_data_pca.shape[1])]
numerical_pca_df = pd.DataFrame(numerical_data_pca, columns=pca_columns, index=data.index)

# Combinar características numéricas transformadas mediante PCA con características categóricas
combined_data = pd.concat([numerical_pca_df, data[categorical_cols], data[binary_cols]], axis=1)

combined_data.head(5)
'''


codigo_kpca = '''
dfMatrix = combined_data.to_numpy()

kprototype = KPrototypes(n_jobs=-1, n_clusters=4, init='Huang', random_state=0)
clusters_kproto_pca = kprototype.fit_predict(dfMatrix, categorical=catColumnsPos)

silueta_kprototype_pca = silhouette_score(numerical_data_pca, clusters)

print(silueta_kprototype_pca)
'''

with st.expander("Código PCA"):
    st.code(codigo_pcak, language="python")

with st.expander("Código Ejecución K-Prototypes con PCA."):
    st.code(codigo_kpca, language="python")
    
st.subheader("Resultados para K-Prototypes con PCA.")
st.markdown("Silhouette Score: **0.3406**")    

##################### K-Means #####################


st.divider()
st.header("K-Means")
codigo = '''
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

# Columnas categóricas
education_cols = ["Education"]
marital_cols = ["Marital_Status"]

# Columnas numéricas (exceptuando las categóricas)
numerical_cols_all = data.select_dtypes(include=["int", "float"]).columns.tolist()
numerical_cols = [col for col in numerical_cols_all if col not in education_cols + marital_cols]

# Definimos el ColumnTransformer
ct = ColumnTransformer(
    [
        ("education", OrdinalEncoder(), education_cols),
        ("num", RobustScaler(), numerical_cols),
    ],
    remainder='passthrough'
)

# Aplicación de las transformaciones
data_transformed = data.copy()

# Aplicamos LabelEncoder por fuera del ColumnTransformer
label_encoder = LabelEncoder()
data_transformed["Marital_Status"] = label_encoder.fit_transform(data_transformed["Marital_Status"])

# Aplicamos el ColumnTransformer
data_transformed[education_cols + numerical_cols] = ct.fit_transform(data_transformed)

data_transformed
'''

with st.expander("Ver código de transformación de datos."):
    st.code(codigo, language="python")



st.write(
    "Se aplicaron transformaciones para preparar los datos: *OrdinalEncoder* para asignar un orden numérico cuando fue pertinente, *LabelEncoder* para codificar categorías distintas, y *RobustScaler* para escalar variables numéricas con valores atípicos."
)



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
st.dataframe(data_transformed.head(5))



codigo = '''kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
predict_kmeans = kmeans.fit_predict(data_transformed)
clusters_df = pd.DataFrame(predict_kmeans)

silueta_kmeans = silhouette_score(data_transformed, clusters_df)
print(round(silueta_kmeans, 4))'''

with st.expander("Ejecución K-Means"):
    st.code(codigo, language="python")

st.subheader("Resultados para K-Means sin PCA.")
st.markdown("Silhouette Score: **0.2129**")


st.divider()





st.header("K-Means con PCA")
st.image("https://i.imgur.com/Gvkjp8O.png")

st.write("Convertimos nuestro dataset ahora solo con variables numéricas a solo 6 componentes principales.")
codigo = '''pca = PCA(n_components=6)
numerical_data_pca = pca.fit_transform(data_transformed)

# Crear un dataFrame para las características numéricas transformadas PCA
pca_columns = [f'PC{i+1}' for i in range(numerical_data_pca.shape[1])]
numerical_pca_df = pd.DataFrame(numerical_data_pca, columns=pca_columns, index=data_transformed.index)'''


with st.expander("Ejecución PCA."):
    st.code(codigo, language="python")


codigo = '''from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
predict_kmeans_pca = kmeans.fit_predict(numerical_data_pca)
clusters_df = pd.DataFrame(predict_kmeans)

silueta_kmeans_pca = silhouette_score(numerical_data_pca, clusters_df)
print(round(silueta_kmeans_pca, 4))'''

st.write("Posteriormente ejecutamos el modelo los resultados obtenidos con PCA.")


with st.expander("Ejecución del Modelo con PCA.", expanded=True):
    st.code(codigo, language="python")
    
st.subheader("Resultados para K-Means con PCA")
st.markdown("Silhouette Score: **0.3290**")



st.divider()


########################### Agglomerative ########################


st.header("Agglomerative")
codigo = '''AC = AgglomerativeClustering(n_clusters=4, linkage="ward")
predict_AC = AC.fit_predict(data_transformed)

silueta_agglomerative_nopca = silhouette_score(data_transformed, predict_AC)
print(round(silueta_agglomerative_nopca, 5))'''

with st.expander("Ejecución Agglomerative", expanded=True):
    st.code(codigo, language="python")


st.subheader("Resultados para Agglomerative sin PCA.")
st.markdown("Silhouette Score: **0.1922**")



st.divider()


st.subheader("Agglomerative con PCA")

col1, col2 = st.columns(2)

with col1:
    st.image("https://i.imgur.com/oL1sxb2.png")

with col2:
    st.image("https://i.imgur.com/ZT2IRHD.png")

codigo = '''AC = AgglomerativeClustering(n_clusters=4, linkage="ward")
predict_AC = AC.fit_predict(PCA_ds)

silueta_agglomerative = silhouette_score(PCA_ds, predict_AC)
print(silueta_agglomerative)'''



with st.expander("Ejecución Agglomerative con PCA", expanded=True):
    st.code(codigo, language="python")



st.subheader("Resultados para Agglomerative con PCA.")
st.markdown("Silhouette Score: **0.3066**")