import streamlit as st

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

