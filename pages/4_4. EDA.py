import pandas as pd
import streamlit as st
from utils import cargar_datos
from utils import limpiar_datos
from utils import features
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide")


df = cargar_datos()
df = limpiar_datos(df)

st.title("Análisis Exploratorio de Datos.")
st.divider()

# elimino outliers
#Nos quedamos con los clientes que tengan un salario < 120000
df = df[df['Income']<120000]

#cambiamos year birth por age
df["Age"] = 2025-df["Year_Birth"]
df = df[df['Age']<90]

#Compras totales
df['NumTotalPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats

num_vars = df.select_dtypes(include=np.number).columns.tolist()

iqr_flags = pd.DataFrame(index=df.index)

for col in num_vars:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_flags[col] = (df[col] < lower) | (df[col] > upper)

df['is_outlier_IQR'] = iqr_flags.any(axis=1)

z_scores = np.abs(stats.zscore(df[num_vars]))
df['is_outlier_Z'] = (z_scores > 3).any(axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_vars])
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df['is_outlier_LOF'] = lof.fit_predict(X_scaled) == -1


print("Outliers por IQR:", df['is_outlier_IQR'].sum())
print("Outliers por Z-score:", df['is_outlier_Z'].sum())
print("Outliers por LOF:", df['is_outlier_LOF'].sum())
df['outlier_todos'] = df['is_outlier_IQR'] & df['is_outlier_Z'] & df['is_outlier_LOF']
print("Outliers detectados por los 3 métodos:", df['outlier_todos'].sum())




df['outlier_score'] = df[['is_outlier_IQR', 'is_outlier_Z', 'is_outlier_LOF']].sum(axis=1)
df['outlier_conf'] = df['outlier_score'] >= 2
print("Outliers detectados por al menos 2 métodos:", df['outlier_conf'].sum())


num_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
            'NumWebVisitsMonth', 'NumTotalPurchases']

def count_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers)

outliers_count = {}

for col in num_cols:
    outliers_count[col] = count_outliers_iqr(df, col)

for col, count in outliers_count.items():
    print(f"{col}: {count} outliers")





def replace_outliers_with_median(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = df[column].median()

    df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
    return df

for col in num_cols:
    df = replace_outliers_with_median(df, col)








st.subheader("Primeras filas del dataset")
st.dataframe(df.head())
st.divider()

st.subheader("Cantidad de valores nulos por columna")
st.dataframe(df.isnull().sum())
st.divider()


st.subheader("Distribución de algunas Variables Numéricas")


fig = px.histogram(df, x="Age", nbins=400, marginal="rug", opacity=0.75)
fig.update_layout(
    title="Distribución de Age",
    xaxis_title="Edad",
    yaxis_title="Frecuencia",
    width=900,
    height=500
)
st.plotly_chart(fig)
st.divider()

fig = px.histogram(df, x="Children", nbins=10, marginal="rug", opacity=0.75)
fig.update_layout(
    title="Distribución de Children",
    xaxis_title="Children",
    yaxis_title="Frecuencia",
    width=830,
    height=800
)
st.plotly_chart(fig)
st.divider()


fig = px.histogram(df, x="Marital_Status", nbins=200, marginal="rug", opacity=0.75)
fig.update_layout(
    title="Distribución de Marital_Status",
    xaxis_title="Marital Status",
    yaxis_title="Frecuencia",
    width=700,
    height=600
)
st.plotly_chart(fig)
st.divider()

st.subheader("Frecuencias")
st.write("Analizamos las frecuencias y valores de las variables categóricas:  ")

codigo = '''df = pd.dataFrame(df)
frecuencia = df['Marital_Status'].value_counts()
frecuencia'''
st.code(codigo)
df = pd.DataFrame(df)
frecuencia = df['Marital_Status'].value_counts()
frecuencia
st.divider()


codigo = '''df = pd.dataFrame(df)
frecuencia = df['Education'].value_counts()
frecuencia a'''
st.code(codigo)
df = pd.DataFrame(df)
frecuencia = df['Education'].value_counts()
frecuencia
st.divider()


df["Spent"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numeric_cols].corr()

np.fill_diagonal(corr_matrix.values, np.nan)
mask = (corr_matrix.abs() < 0.65) | (corr_matrix.isna())  # Invertimos la condición
filtered_corr = corr_matrix.mask(mask)  # Esto mantendrá solo |r| ≤ 0.75

# Eliminar filas/columnas completamente vacías
filtered_corr = filtered_corr.dropna(how='all', axis=0).dropna(how='all', axis=1)

if filtered_corr.empty:
    st.warning("Todas las correlaciones son fuertes (|r| > 0.75). No hay valores en el rango deseado.")
else:
    fig = px.imshow(
        filtered_corr,
        text_auto=".2f",
        color_continuous_scale='RdBu',
        zmin=-0.75, 
        zmax=0.75,
        labels=dict(color="Correlación"),
        x=filtered_corr.columns,
        y=filtered_corr.index
    )
    fig.update_layout(
        title="Matriz de Correlación Filtrada (|r| ≤ 0.75)",
        width=800,
        height=800,
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Gráfico entre Income y Spent")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Income",
    y="Spent",
    color="red", 
    alpha=0.6     
)

plt.title("Income vs Spent", fontsize=16)
plt.xlabel("Income", fontsize=12)
plt.ylabel("Spent", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)

st.pyplot(plt)
st.write("Se observa una relación positiva (a mayor ingreso, mayor gasto) pero con una dispersión significativa. La mayoría de los datos se concentran en ingresos menores a $60000")
st.divider()

st.header("Income en relación a Education")
# Configurar figura
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=df,
    x="Education",
    y="Income",
    palette="viridis",
    estimator="mean"  
)


st.subheader("Gráfico comparando el tipo de Educación con los Gastos y el Ingreso")
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6))

sns.barplot(x="Education", y="Income", data=df, ax=ax0, palette="viridis")
ax0.set_title("Income According to Education")
ax0.set_ylabel("Income (USD)")
ax0.set_xlabel("")

sns.barplot(x="Education", y="Spent", data=df, ax=ax1, palette="viridis")
ax1.set_title("Total Spent by Education")
ax1.set_ylabel("Spent (USD)")
ax1.set_xlabel("")

education_counts = df["Education"].value_counts()
ax0.text(x=-0.3, y=10000, s=f"n: {education_counts[0]}", fontsize=10)
ax0.text(x=0.7, y=10000, s=f"n: {education_counts[1]}", fontsize=10)
ax0.text(x=1.7, y=10000, s=f"n: {education_counts[2]}", fontsize=10)

plt.tight_layout()

st.pyplot(fig)
st.write("Podemos observar que los de menor educación son los que tienen menos ingresos y los que menos gastan, los graduados y postgraduados tienen un nivel similar de ingresos y gastos. Los que no estan graduados representan un grupo pequeño en comparación a los otros.")
st.divider()


st.title("Análisis por Estado Civil")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6))

sns.barplot(
    x='Marital_Status',
    y='Income',
    data=df,
    ax=ax0,
    palette="viridis"
)
ax0.set_title('Income According to Marital Status')
ax0.set_ylabel('Income (USD)')
ax0.set_xlabel('')

sns.barplot(
    x='Marital_Status',
    y='Spent',
    data=df,
    ax=ax1,
    palette="viridis"
)
ax1.set_title('Spent By Customers by Their Marital Status')
ax1.set_ylabel('Spent (USD)')
ax1.set_xlabel('')

# 5. Añadir anotaciones de conteo (n=)
marital_counts = df['Marital_Status'].value_counts().sort_index()

for i, count in enumerate(marital_counts):
    ax0.text(
        x=i-0.35,
        y=10000,   
        s=f"n = {count}",
        fontsize=10
    )
    ax1.text(
        x=i-0.35,
        y=df['Spent'].max()*0.1,
        s=f"n = {count}",
        fontsize=10
    )

plt.tight_layout()

st.pyplot(fig)

st.write("Podemos observar que los grupos tiene un ingreso similar, sin embargo las parejas y los matrimonios tienen un gasto menor de los que estan solteros")
st.divider()


st.title("Análisis de Ingresos y Ventas por Número de Hijos")

children = [0, 1, 2, 3]  # Número de hijos
income = [70000, 60000, 50000, 40000]  # Ingresos promedio
mean_sales = [1105.25, 474.70, 246.74, 255.50]  # Ventas promedio
counts = [633, 1117, 416, 50]  # Conteo de registros (n)

# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Gráfico de Ingresos ---
bars1 = ax1.bar(children, income, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_title('Ingresos por Número de Hijos', fontsize=14)
ax1.set_xlabel('Número de Hijos', fontsize=12)
ax1.set_ylabel('Ingreso ($)', fontsize=12)
ax1.set_ylim(0, 75000)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir etiquetas de conteo (n)
for i, (bar, count) in enumerate(zip(bars1, counts)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 2000, 
             f"n: {count}", ha='center', va='bottom', fontsize=10)

# --- Gráfico de Ventas Promedio ---
bars2 = ax2.bar(children, mean_sales, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_title('Ventas Promedio por Número de Hijos', fontsize=14)
ax2.set_xlabel('Número de Hijos', fontsize=12)
ax2.set_ylabel('Ventas Promedio ($)', fontsize=12)
ax2.set_ylim(0, 1200)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir etiquetas de Mean Sales
for i, (bar, sales) in enumerate(zip(bars2, mean_sales)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 20, 
             f"Mean Sales: {sales:.2f}", ha='center', va='bottom', fontsize=10)

# Ajustar layout
plt.tight_layout()

# Mostrar en Streamlit
st.pyplot(fig)

# Mostrar datos en tabla
st.subheader("Datos Resumidos")
data_table = {
    "Número de Hijos": children,
    "Ingreso Promedio": income,
    "Ventas Promedio": mean_sales,
    "Conteo (n)": counts
}
st.table(data_table)

st.write("Podemos observar que los que no tienen hijos tienen un mayor ingreso a cualquier cantidad de hijos, y también son los que mas gastos presentan, mostrando que los que tienen hijos presentan menos gastos.")

st.divider()

st.title("Distribución de Ventas por Categoría")

mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

# Crear figura
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Ajustar espacio entre subplots

# Generar histogramas
for i, col in enumerate(mnt_cols):
    ax = axes[i//3, i%3]
    sns.histplot(df[col], bins=100, ax=ax, color='skyblue', kde=True)
    
    # Personalización
    ax.set_title(f'Distribución de {col}', fontsize=12)
    ax.set_xlabel('Monto ($)', fontsize=10)
    ax.set_ylabel('Frecuencia', fontsize=10)
    
    # Añadir texto con el total vendido
    ax.text(x=df[col].max()/3, 
            y=ax.get_ylim()[1]*0.8,  # Posición vertical relativa
            s=f"Total vendido:\n${df[col].sum():,.0f}",
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))

# Mostrar en Streamlit
st.pyplot(fig)

# Mostrar estadísticas adicionales
st.subheader("Resumen Estadístico")
st.dataframe(df[mnt_cols].describe().style.format("{:,.2f}"))
st.write("Podemos observar que el vino y la carne son los productos más comprados, todos los productos presentan una distribucion sesgada a la derecha, lo cual implica que los clientes suelen comprar cantidades bajas de productos")
st.divider()




# Agrupación y cálculo de métricas
education_stats = df.groupby('Education')[['Income', 'MntWines', 'MntFruits', 
                                          'MntMeatProducts', 'MntFishProducts',
                                          'MntSweetProducts', 'MntGoldProds']].agg(['mean', 'sum'])

# Formateo para mejor visualización
education_stats.columns = ['_'.join(col).strip() for col in education_stats.columns.values]
education_stats.reset_index(inplace=True)

# Mostrar en Streamlit
st.title("Compras según el Nivel Educativo")

# DataFrame con estilo
styled_df = education_stats.style\
    .format({
        'Income_mean': '${:,.2f}',
        'Income_sum': '${:,.2f}',
        'MntWines_mean': '${:,.2f}',
        'MntWines_sum': '${:,.2f}',
        'MntFruits_mean': '${:,.2f}',
        'MntFruits_sum': '${:,.2f}',
        'MntMeatProducts_mean': '${:,.2f}',
        'MntMeatProducts_sum': '${:,.2f}',
        'MntFishProducts_mean': '${:,.2f}',
        'MntFishProducts_sum': '${:,.2f}',
        'MntSweetProducts_mean': '${:,.2f}',
        'MntSweetProducts_sum': '${:,.2f}',
        'MntGoldProds_mean': '${:,.2f}',
        'MntGoldProds_sum': '${:,.2f}'
    })\
    .set_properties(**{'text-align': 'center'})\
    .set_table_styles([{
        'selector': 'th',
        'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]
    }])

st.dataframe(styled_df)
st.divider()


# Agrupación y cálculo de métricas
education_stats = df.groupby('Marital_Status')[['Income', 'MntWines', 'MntFruits', 
                                          'MntMeatProducts', 'MntFishProducts',
                                          'MntSweetProducts', 'MntGoldProds']].agg(['mean', 'sum'])

# Formateo para mejor visualización
education_stats.columns = ['_'.join(col).strip() for col in education_stats.columns.values]
education_stats.reset_index(inplace=True)

# Mostrar en Streamlit
st.title("Compras según el Nivel Educativo")

# DataFrame con estilo
styled_df = education_stats.style\
    .format({
        'Income_mean': '${:,.2f}',
        'Income_sum': '${:,.2f}',
        'MntWines_mean': '${:,.2f}',
        'MntWines_sum': '${:,.2f}',
        'MntFruits_mean': '${:,.2f}',
        'MntFruits_sum': '${:,.2f}',
        'MntMeatProducts_mean': '${:,.2f}',
        'MntMeatProducts_sum': '${:,.2f}',
        'MntFishProducts_mean': '${:,.2f}',
        'MntFishProducts_sum': '${:,.2f}',
        'MntSweetProducts_mean': '${:,.2f}',
        'MntSweetProducts_sum': '${:,.2f}',
        'MntGoldProds_mean': '${:,.2f}',
        'MntGoldProds_sum': '${:,.2f}'
    })\
    .set_properties(**{'text-align': 'center'})\
    .set_table_styles([{
        'selector': 'th',
        'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]
    }])

st.dataframe(styled_df)
st.divider()