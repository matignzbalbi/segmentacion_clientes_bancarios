import streamlit as st

st.set_page_config(layout="wide")
st.markdown("<h1 style='color:#00BFFF;'>Generación de Insights.</h1>", unsafe_allow_html=True)
st.title("Generación de Insights.")
st.divider()

st.header("Resultados.")
st.write("En base a la evaluación vista en la solapa anterior vemos que no hay mucha diferencia entre los resultados de los modelos. Por esta razón, decidimos utilizar `K-Prototypes` con `PCA`, ya que en este modelo, estamos teniendo en cuenta todas las variables.")
st.write("El resultado que obtuvimos fueron cuatro clusters, y podemos visualizarlos de esta manera:" )
