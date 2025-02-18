from streamlit_folium import st_folium
import streamlit as st
import folium
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip()
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    return df

import locale


st.set_page_config(
    layout="wide",
)
# Définir la locale en français
locale.setlocale(locale.LC_TIME, 'fr_FR')

# --- Chargement des données ---
data = load_data('basemayotte.csv')

st.title("Analyse des Répliques")

# Sélectionner l'année et le mois
year_filter = st.sidebar.selectbox("Choisir l'année", data['Time'].dt.year.unique())

# Utiliser le nom du mois en français dans le selectbox
# Pour cela, on formate la date en nom du mois (en fonction de la locale configurée)
month_filter = st.sidebar.selectbox("Choisir le mois", data['Time'].dt.strftime('%B').unique())

# Filtrer les données par année et par mois (en comparant la chaîne de caractères du mois)
filtered_data = data[
    (data['Time'].dt.year == year_filter) & 
    (data['Time'].dt.strftime('%B') == month_filter)
]

# Afficher le nombre d'événements pour la période choisie
st.sidebar.markdown(f"**Événements pour {month_filter} {year_filter}**")
st.sidebar.markdown(f"Nombre d'événements: {len(filtered_data)}")


# --- 1. Sélection de l'événement principal ---
st.sidebar.header("Événement Principal")

# Seuil de magnitude pour la sélection d'un mainshock (par exemple, 5.0)
seuil_principal = st.sidebar.slider(
    "Seuil de magnitude pour l'événement principal", 
    min_value=3.0, max_value=9.0, value=3.0, step=0.1
)

# Filtrer les événements candidats pour être un mainshock parmi la période sélectionnée
candidats_principal = (
    filtered_data[filtered_data['Magnitude'] >= seuil_principal]
    .sort_values('Time')
    .reset_index(drop=True)
)

if candidats_principal.empty:
    st.error("Aucun événement ne satisfait le seuil de magnitude choisi.")
    st.stop()

# Création d'une liste descriptive pour la sélection
options_principal = candidats_principal.apply(
    lambda row: f"{row['Time'].strftime('%Y-%m-%d %H:%M:%S')} | Mag: {row['Magnitude']:.1f}", axis=1
).tolist()

ev_principal_str = st.sidebar.selectbox("Choisissez l'événement principal", options_principal)
# Récupérer l'index de l'événement sélectionné dans la liste
index_selection = options_principal.index(ev_principal_str)
principal = candidats_principal.iloc[index_selection]
principal['Temps'] = principal['Time'].strftime('%Y-%m-%d %H:%M')

#Fonction pour calcul de distance
def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Rayon de la Terre en km
        dLat = np.radians(lat2 - lat1)
        dLon = np.radians(lon2 - lon1)
        a = np.sin(dLat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c


col1, col2 = st.columns([4,6])

with col1:
    st.markdown(f"### Événement principal sélectionné")
    st.write(principal[['Temps', 'Magnitude', 'Latitude', 'Longitude', 'Profondeur']])

    # --- 2. Paramétrage des seuils pour les répliques ---
    st.sidebar.header("Paramètres des Répliques")
    rayon_spatial = st.sidebar.slider("Rayon spatial pour les répliques (km)", 
                                          min_value=5, max_value=200, value=50, step=5)
    fenetre_temporelle_jours = st.sidebar.slider("Fenêtre temporelle après l'événement principal (jours)", 
                                         min_value=1, max_value=30, value=7, step=1)
    fenetre_temporelle = pd.Timedelta(days=fenetre_temporelle_jours)

    # --- 3. Identification des répliques ---
    # On considère comme répliques les événements survenant après le mainshock,
    # dans le rayon spatial défini et dans la fenêtre temporelle donnée.

    # On sélectionne les événements survenant après l'événement principal
    candidats = data[data['Time'] > principal['Time']].copy()
    # Calcul de la distance entre le mainshock et chaque candidat
    candidats['Distance'] = candidats.apply(
        lambda row: haversine(principal['Latitude'], principal['Longitude'], row['Latitude'], row['Longitude']),
        axis=1
    )
    # Calcul de la différence de temps
    candidats['Différence de temps'] = candidats['Time'] - principal['Time']

    # Filtrer en fonction des seuils spatial et temporel
    repliques = candidats[
        (candidats['Distance'] <= rayon_spatial) &
        (candidats['Différence de temps'] <= fenetre_temporelle)
    ].copy()
    
    repliques["Temps"] = repliques["Time"].apply(lambda x : x.strftime('%Y-%m-%d %H:%M'))
    
    st.markdown("### Répliques Identifiées")
    st.write(f"Nombre de répliques trouvées : {len(repliques)}")
    st.dataframe(repliques[['Temps', 'Magnitude', 'Latitude', 'Longitude', 'Distance', 'Différence de temps']].round(2))

# --- 4. Visualisation sur la carte ---
# On affiche le mainshock et les répliques sur la carte.
# Le mainshock sera affiché en rouge, les premières répliques en orange et les dernières en vert.
# Définir le centre de la carte sur le mainshock
m = folium.Map(location=[principal['Latitude'], principal['Longitude']], zoom_start=10)

# Ajouter le mainshock
popup_principal = (f"<b>Secousse principale</b><br>"
                   f"<b>Temps:</b> {principal['Temps']}<br>"
                   f"<b>Magnitude:</b> {principal['Magnitude']}<br>"
                   f"<b>Profondeur:</b> {principal['Profondeur']} km")
folium.CircleMarker(
    location=[principal['Latitude'], principal['Longitude']],
    radius=10 + principal['Magnitude'] * 2,
    color='red',
    fill=True,
    fill_color='red',
    fill_opacity=0.9,
    popup=popup_principal
).add_to(m)

# Trier les répliques par date
repliques_sorted = repliques.sort_values(by='Time')

# Diviser les répliques en deux groupes: premières (orange) et dernières (vertes)
premieres_repliques = repliques_sorted.iloc[:len(repliques_sorted)//2]
dernières_repliques = repliques_sorted.iloc[len(repliques_sorted)//2:]

# Ajouter une légende
legend_html = """
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 200px; height: 120px; 
                 background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                 padding: 10px;">
         <b>Legend:</b><br>
         <i style="background: orange; width: 15px; height: 15px; float: left; margin-right: 5px;"></i> 
         First aftershocks<br>
         <i style="background: green; width: 15px; height: 15px; float: left; margin-right: 5px;"></i> 
         Last aftershocks<br>
         <i style="background: #FF4500; width: 15px; height: 15px; float: left; margin-right: 5px;"></i> 
         Mainshock<br>
     </div>
"""
# Ajouter la légende à la carte
m.get_root().html.add_child(folium.Element(legend_html))


# Ajouter les premières répliques en orange
for idx, row in premieres_repliques.iterrows():
    popup_text = (f"<b>Réplique</b><br>"
                  f"<b>Temps:</b> {row['Time']}<br>"
                  f"<b>Magnitude:</b> {row['Magnitude']}<br>"
                  f"<b>Distance:</b> {row['Distance']:.1f} km<br>"
                  f"<b>ΔTemps:</b> {row['Différence de temps']}")
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5 + row['Magnitude'] * 2,
        color='orange',
        fill=True,
        fill_color='orange',
        fill_opacity=0.7,
        popup=popup_text
    ).add_to(m)
    # Optionnel : tracer une ligne reliant le mainshock à la réplique
    folium.PolyLine(
        locations=[(principal['Latitude'], principal['Longitude']),
                   (row['Latitude'], row['Longitude'])],
        color='orange',
        weight=2,
        opacity=0.6
    ).add_to(m)

# Ajouter les dernières répliques en vert
for idx, row in dernières_repliques.iterrows():
    popup_text = (f"<b>Réplique</b><br>"
                  f"<b>Temps:</b> {row['Time']}<br>"
                  f"<b>Magnitude:</b> {row['Magnitude']}<br>"
                  f"<b>Distance:</b> {row['Distance']:.1f} km<br>"
                  f"<b>ΔTemps:</b> {row['Différence de temps']}")
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5 + row['Magnitude'] * 2,
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=0.7,
        popup=popup_text
    ).add_to(m)
    # Optionnel : tracer une ligne reliant le mainshock à la réplique
    folium.PolyLine(
        locations=[(principal['Latitude'], principal['Longitude']),
                   (row['Latitude'], row['Longitude'])],
        color='green',
        weight=2,
        opacity=0.6
    ).add_to(m)

with col2:
    # Afficher la carte
    st.markdown("### Carte Interactive des Répliques")
    st_folium(m, width=800, height=800)






