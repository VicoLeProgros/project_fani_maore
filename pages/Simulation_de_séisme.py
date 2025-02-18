import streamlit as st

# Lire le fichier HTML
with open("pages/carte_villages_seismes.html", "r", encoding="utf-8") as file:
    html_content = file.read()
st.set_page_config(layout="wide")
st.components.v1.html(html_content, height=900, scrolling=True)

st.sidebar.header("Calculs utilisés")
st.sidebar.text("Le cercle le plus large a été créé en utilisant une formule d'approximation empirique pour estimer la distance maximale touchée qui est la suivante:")
st.sidebar.latex(r"R = 10^{0.5M - 1.8}")
st.sidebar.markdown(r"$R$ = Distance jusqu'à laquelle l'intensité atteint un niveau significatif")
st.sidebar.markdown(r"$M$= Magnitude sur l'échelle de Richter")
st.sidebar.text("Ensuite les autres cercles ont été créés en utilisant une formule générale d'atténuation de l'intensité:")
st.sidebar.latex(r"I = I_0 - \beta \times \log_{10}(D)")
st.sidebar.markdown(r"$I$ = Intensité ressentie à une distance D")
st.sidebar.markdown(r"$I_0$ = Intensité maximale à l'épicentre")
st.sidebar.markdown(r"$D$ = Distance à l'épicentre (km)")
st.sidebar.markdown(r"$\beta$ = coefficient qui varie en fonction du type de sol, nous avons beta=4 qui correspond au type de sol de mayotte (volcanique)")
