import streamlit as st

st.set_page_config(layout="wide")
st.title("Toma de decisiones.")
st.divider()

st.header("Segmentación.")

st.markdown('''
        Debido a las carácteristicas de los clústers generados por KPrototypes proponemos los\
        siguientes perfiles para cada uno de ellos:
        * `Cluster 1`: **Segmento de Bajos Ingresos**.
        * `Cluster 2`: **Segmento de Ingresos Medios**.
        * `Cluster 3`: **Segmento de Ingresos Medios Plus**.
        * `Cluster 4`: **Segmento de Altos Ingresos**.
            ''')

st.divider()
st.header("Estrategias propuestas por tipo de cliente.")


st.subheader("Segmento de Bajos Ingresos:")
st.markdown('''
- Introducir líneas de productos económicos y ampliar la variedad de artículos con descuentos significativos.

- Personalizar promociones basadas en el comportamiento de compra de este grupo.

- Asegurar que estén informados sobre ofertas a través de campañas de email marketing y newsletters.
''')
st.divider()

st.subheader("Segmento de Ingresos Medios:")
st.markdown('''
- Mejorar la visibilidad de las ofertas en la web y personalizarlas según los intereses del cliente.

- Incrementar la variedad de productos y marcas disponibles.

- Experimentar con descuentos en nuevas categorías para estimular el consumo.

- Fomentar la conversión de clientes regulares en compradores frecuentes mediante una experiencia de usuario optimizada, promociones atractivas y campañas de email marketing o newsletters.
''')
st.divider()
st.subheader("Segmento de Altos Ingresos:")
st.markdown('''
- Enfocarse en optimizar el catálogo, asegurando que sea claro y accesible.

- Ofrecer beneficios como envíos gratuitos o descuentos en el delivery para incentivar compras en grandes cantidades.

- Incentivar la fidelidad mediante programas de membresía o beneficios post-compra.

- Que los productos sean frescos puede ser un factor clave para que sigan consumiendo
''')
st.divider()
st.subheader("Cliente D:")
st.markdown('''
- Crear promociones exclusivas y diferenciadas, enfocadas en productos de alta demanda para este segmento.

- Incentivar la fidelidad mediante programas de membresía o beneficios post-compra.

- Ofrecer envíos gratuitos o descuentos en el delivery para aumentar la conveniencia.

''')

