import streamlit as st

st.set_page_config(layout="wide")
st.markdown("<h1 style='color:#00BFFF;'>Toma de decisiones.</h1>", unsafe_allow_html=True)
st.divider()

st.header("Segmentación.")
st.divider()
st.markdown('''
        Debido a las carácteristicas de los clústers generados por KPrototypes proponemos los\
        siguientes perfiles para cada uno de ellos:
        * `Cluster 1`: **Segmento Bronce (Ingresos Bajos)**.
        * `Cluster 2`: **Segmento Plata (Ingresos Medios)**.
        * `Cluster 3`: **Segmento Oro (Ingresos Medios Altos)**.
        * `Cluster 4`: **Segmento Platino (Ingresos Altos)**.
            ''')

st.divider()
st.header("Estrategias propuestas por tipo de cliente.")


st.divider()
####################################### SEG Bronce #######################################

st.markdown("<h3><u>Segmento Bronce (Ingresos Bajos):</u></h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Insight principal:")
    st.markdown("""
    - Perfil estándar en ingresos y educación, mayormente entre 45 y 70 años, con comportamiento digital activo.
    - Consultan la web con frecuencia y responden bien a promociones relevantes.
    """)

with col2:
    st.subheader("Patrón de compral:")
    st.markdown("""
    - Buscan directamente ofertas, ignorando el catálogo.
    - Compran más que los de bajo ingreso, especialmente vinos y productos de oro.
    """)

with col3:
    st.subheader("Estrategia clave:")
    st.markdown("""
    - Personalizar ofertas web visibles y atractivas.
    - Ampliar la variedad de productos promocionados.
    - Usar newsletters con descuentos dirigidos.
    """)



st.divider()
####################################### SEG Plata #######################################

st.markdown("<h3><u>Segmento Plata (Ingresos Medios):</u></h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Insight principal:")
    st.markdown("""
    - Grupo con menor poder adquisitivo, evaluador de precios y bajo volumen de compra.
    - Alta navegación online pero baja conversión.
    """)

with col2:
    st.subheader("Patrón de compra:")
    st.markdown("""
    - Prefiere productos económicos.
    - Sensible a descuentos grandes (no a 2x1).
    - Ignora catálogo, prioriza el precio.
    """)

with col3:
    st.subheader("Estrategia clave:")
    st.markdown("""
    - Ofrecer productos económicos y promociones visibles.
    - Enfocar campañas con precios bajos, no cantidad.
    - Email marketing con productos destacados y descuento fuerte.
    """)


st.divider()
####################################### SEG Oro #######################################
st.markdown("<h3><u>Segmento Oro (Ingresos Medios Altos):</u></h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Insight principal:")
    st.markdown("""
    - Grupo pequeño pero con alto volumen por persona.
    - Prefiere conveniencia y calidad sobre descuentos.
    """)

with col2:
    st.subheader("Patrón de compra:")
    st.markdown("""
    - Compra grandes cantidades de pescado, fruta y dulces.
    - Usa más el catálogo que las ofertas.
    - Activo en tienda y web.
    """)

with col3:
    st.subheader("Notas y Estrategias")
    st.markdown("""
    - Optimizar el catálogo digital.
    - Envíos gratis y promociones por volumen.
    - Fidelizar con programas tipo membresía o recompensas.
    """)


st.divider()
####################################### SEG Platino #######################################

st.markdown("<h3><u>Segmento Platino (Ingresos Altos):</u></h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Insight principal:")
    st.markdown("""
    - Comprador mayor (60-70 años), con comportamiento planificado y definido.
    - Educación alta, sin hijos en su mayoría, más receptivo a campañas recientes.
    """)

with col2:
    st.subheader("Patrón de compra:")
    st.markdown("""
    - Prefiere vinos y carne.
    - Compra cantidades moderadas, sin depender de descuentos.
    - Usa catálogo, no busca ofertas.
    """)

with col3:
    st.subheader("Estrategia clave:")
    st.markdown("""
    - Crear promociones personalizadas para productos clave.
    - Incentivar fidelidad con beneficios exclusivos (membresía, recompensas).
    - Ofrecer delivery conveniente y gratuito.
    """)