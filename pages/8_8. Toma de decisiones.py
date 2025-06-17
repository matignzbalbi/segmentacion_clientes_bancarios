import streamlit as st

st.set_page_config(layout="wide")
st.title("Toma de decisiones.")
st.divider()

st.header("Segmentación.")
st.divider()
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
st.divider()
st.markdown("<h3><u>Segmento de Bajos Ingresos:</u></h3>", unsafe_allow_html=True)
st.markdown('''
Productos:
- Compra pequeñas cantidades de productos, con patrones de compra similares en todas las categorías.
- Ligera preferencia por productos de oro y pescado.


Compras:
- Adquiere productos con poca frecuencia, mostrando una baja respuesta a ofertas.
- Visita frecuentemente la página web, pero no suele concretar compras.
- Tiende a ignorar el catálogo y se enfoca en productos económicos.


Familia:
- Perfil variado: casados, solteros o en pareja.
- Generalmente tienen uno o más hijos.


Perfil:
- Incluye graduados, posgraduados y, de manera única, personas sin estudios superiores.
- Mayoría entre 40 y 50 años.
- Es el grupo con menores ingresos y menor gasto promedio, lo que se refleja en un volumen de compra reducido.
- Frecuentemente presenta quejas.
- Baja aceptación de campañas promocionales, especialmente las iniciales.


Notas:
- Debido a sus limitaciones económicas, este segmento prioriza las ofertas con altos porcentajes de descuento, mostrando poco interés en promociones tipo 2x1 debido al bajo volumen de productos comprados.
- Evalúa cuidadosamente el precio de sus compras, lo que podría explicar la baja conversión en la web si no encuentran productos económicos o descuentos atractivos.


Estrategias:
- Introducir líneas de productos económicos y ampliar la variedad de artículos con descuentos significativos.
- Personalizar promociones basadas en el comportamiento de compra de este grupo.
- Asegurar que estén informados sobre ofertas a través de campañas de email marketing y newsletters.''')

st.divider()
st.markdown("<h3><u>Segmento de Ingresos Medios:</u></h3>", unsafe_allow_html=True)
st.markdown('''
Productos:
- Patrones de compra similares a los del segmento de bajos ingresos, pero con un volumen ligeramente superior.
- Preferencia notable por vinos y productos de oro.


Compras:
- Comprador activo en la web, con alta frecuencia de visitas.
- Tiende a ignorar el catálogo y busca directamente ofertas.
- Responde positivamente a promociones, especialmente si son relevantes.


Familia:
- Perfil variado: casados, solteros o en pareja.
- Generalmente tienen uno o más hijos.


Perfil:
- Mayoría de graduados y posgraduados.
- Mayoría entre 45 y 70 años.
- Representa el estándar promedio en términos de ingresos, gasto y comportamiento de compra.
- Baja aceptación de campañas promocionales en general.


Notas:
- Este segmento está atento a las ofertas publicadas en la web y suele aprovechar cualquier tipo de promoción, independientemente del producto o marca.
- La falta de variedad en productos o la orientación de las ofertas hacia categorías específicas (como vinos o marcas particulares) podría limitar sus compras en otras categorías.


Estrategias:
- Mejorar la visibilidad de las ofertas en la web y personalizarlas según los intereses del cliente.
- Incrementar la variedad de productos y marcas disponibles.
- Experimentar con descuentos en nuevas categorías para estimular el consumo.
- Fomentar la conversión de clientes regulares en compradores frecuentes mediante una experiencia de usuario optimizada, promociones atractivas y campañas de email marketing o newsletters.
''')
st.divider()
st.markdown("<h3><u>Segmento de Altos Ingresos Plus:</u></h3>", unsafe_allow_html=True)
st.markdown('''
Productos:
- Fuerte preferencia por pescado, frutas y dulces.
- Compra en grandes cantidades, a pesar de ser un grupo reducido.
- Consumo en otras categorías similar al segmento de ingresos medios, pero ligeramente superior.


Compras:
- Comprador activo en la web, aunque con menos visitas frecuentes.
- Prefiere adquirir productos a través del catálogo en lugar de buscar ofertas.
- Compra en grandes cantidades, priorizando la conveniencia sobre los descuentos.


Familia:
- Perfil variado: casados, solteros o en pareja.
- Generalmente tienen cero o un hijo.


Perfil:
- Mayoría de graduados, con pocos posgraduados.
- Mayoría entre 45 y 55 años.
- Destaca por su alto volumen de compras por persona.
- Baja aceptación de campañas promocionales.


Notas:
- Este cliente parece tener claro lo que desea comprar y aprovecha ofertas relacionadas con la cantidad de productos más que con el precio.
- Es probable que consuma productos de gama media-alta, con una rotación rápida debido a su corta vida útil (como frutas y pescado).
- La conveniencia y la disponibilidad de productos específicos son clave para este segmento.


Estrategias:
- Enfocarse en optimizar el catálogo, asegurando que sea claro y accesible.
- Ofrecer beneficios como envíos gratuitos o descuentos en el delivery para incentivar compras en grandes cantidades.
- Incentivar la fidelidad mediante programas de membresía o beneficios post-compra.
- Que los productos sean frescos puede ser un factor clave para que sigan consumiendo''')
st.divider()
st.markdown("<h3><u>Segmento de Altos Ingresos:</u></h3>", unsafe_allow_html=True)
st.markdown('''
Productos:
- Preferencia por vinos y carne.
- Compra cantidades moderadamente superiores a la media.


Compras:
- Comprador activo en la web, aunque con visitas menos frecuentes.
- Prefiere el catálogo sobre las ofertas y realiza compras planificadas.
- Adquiere cantidades moderadas de productos, sin depender de promociones.


Familia:
- Perfil variado: solteros, casados o en pareja.
- Mayoría sin hijos, aunque algunos están casados.


Perfil:
- Mayoría de graduados y posgraduados.
- Mayoría entre 60 y 70 años.
- Grupo longevo que compra cantidades ligeramente superiores a lo estandar.
- Mayor aceptación de campañas promocionales en comparación con otros segmentos, especialmente las más recientes.


Notas:
- Este cliente parece tener preferencias definidas y realiza compras planificadas, mostrando poca atención a las ofertas.
- El volumen moderado de compras puede estar relacionado con la conveniencia o la disponibilidad de productos específicos que consume con regularidad.
- La aceptación de campañas recientes sugiere potencial para fidelizar a este grupo.


Estrategias:
- Crear promociones exclusivas y diferenciadas, enfocadas en productos de alta demanda para este segmento.
- Incentivar la fidelidad mediante programas de membresía o beneficios post-compra.
- Ofrecer envíos gratuitos o descuentos en el delivery para aumentar la conveniencia.
''')
st.divider()
