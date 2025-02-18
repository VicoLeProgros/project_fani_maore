import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import folium
import streamlit as st
from streamlit_folium import folium_static


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip()
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    v_bat = pd.read_excel("types_batiments.xlsx",sheet_name="PRINC2")
    return df,v_bat
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 30% !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

data,villes_bat = load_data('basemayotte.csv')
fani_maore = (-12.8479964, 45.4654728)

data['Mois'] = data['Time'].dt.to_period('M').astype(str)  # Format AAAA-MM
data['Année'] = data['Time'].dt.year

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Rayon de la Terre en km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # Distance en kilomètres
    return distance



# Filtrer pour ne garder que les séismes de magnitude > 2
data_filtered = data[data['Magnitude'] > 4].copy()

# ----- 2. Conversion en GeoDataFrame et reprojection -----
# Création d'un GeoDataFrame en EPSG:4326 à partir des colonnes Longitude/Latitude
gdf = gpd.GeoDataFrame(
    data_filtered,
    geometry=gpd.points_from_xy(data_filtered.Longitude, data_filtered.Latitude),
    crs="EPSG:4326"
)

# Reprojeter dans un CRS en mètres adapté à Mayotte.
# Pour Mayotte, on peut utiliser l'UTM zone 38S (EPSG:32738).
gdf = gdf.to_crs(epsg=32738)

# ----- 3. Création d'une grille de 1000 m x 1000 m -----
xmin, ymin, xmax, ymax = gdf.total_bounds
grid_width = 4000  # largeur en mètres
grid_height = 4000  # hauteur en mètres

# Création des intervalles pour construire la grille
cols = np.arange(xmin, xmax + grid_width, grid_width)
rows = np.arange(ymin, ymax + grid_height, grid_height)

polygons = []
for x in cols[:-1]:
    for y in rows[:-1]:
        polygons.append(Polygon([
            (x, y),
            (x + grid_width, y),
            (x + grid_width, y + grid_height),
            (x, y + grid_height)
        ]))

# Création du GeoDataFrame de la grille
grid = gpd.GeoDataFrame({'geometry': polygons}, crs=gdf.crs)

# ----- 4. Affecter chaque événement à une cellule de la grille -----
# Réaliser une jointure spatiale pour associer chaque séisme à la cellule correspondante
joined = gpd.sjoin(gdf, grid, how="left", predicate="within")

# ----- 5. Calcul du risque par cellule -----
# Calculer le nombre total d'années dans les données (pour normaliser par an)
total_years = data_filtered['Année'].nunique()

# Compter le nombre d'événements par cellule (chaque cellule est identifiée par 'index_right')
risk_df = joined.groupby('index_right').size().reset_index(name='event_count')

# Calculer la fréquence annuelle (λ) et le risque selon la loi de Poisson
risk_df['lambda'] = risk_df['event_count'] / total_years
risk_df['risk'] = 1 - np.exp(-risk_df['lambda'])  # probabilité d'au moins un événement l'année suivante

# Fusionner ces informations avec la grille
grid['risk'] = risk_df.set_index('index_right')['risk']
grid['risk'] = grid['risk'].fillna(0)  # aucune observation → risque 0

# ----- 6. Reprojection pour l'affichage sur Leaflet (EPSG:4326) -----
grid = grid.to_crs(epsg=4326)


# Conversion du catalogue en GeoDataFrame
gdf = gpd.GeoDataFrame(
    data_filtered,
    geometry=gpd.points_from_xy(data_filtered.Longitude, data_filtered.Latitude),
    crs="EPSG:4326"
)

# ----- Définition d'une GMPE simplifiée -----
def gmpe(magnitude, distance):
    """
    Modèle très simplifié de prédiction de l'accélération de pic au sol (PGA) en g.
    Pour éviter log(0), on ajoute une constante 'c' à la distance.
    
    PGA = exp( a + b * magnitude - ln(distance + c) )
    """
    a = -1.5
    b = 0.5
    c = 10.0  # facteur d'échelle (distance en km)
    return np.exp(a + b * magnitude - np.log(distance + c))

# Seuil d'excès (par exemple, PGA >= 0.1 g) -> à partir de 0.1, des dégats peuvent survenir
PGA_threshold = st.sidebar.slider('Seuil PGA (en g)', min_value=0.05, max_value=1.0, value=0.15, step=0.01)  # Curseur pour ajuster le seuil PGA

# ----- Estimation du taux annuel des événements -----
# On calcule la durée d'observation en années
total_months = data_filtered['Mois'].nunique()
observation_years = total_months / 12

# On attribue à chaque événement un taux annuel (approximé par 1 occurrence sur la période)
data_filtered['annual_rate'] = 1 / observation_years

# ----- Calcul du PSHA pour chaque village -----
villes_pha = []
for idx, ville in villes_bat.iterrows():
    total_annual_rate = 0  # somme des contributions de chaque événement
    for _, event in data_filtered.iterrows():
        # Calcul de la distance (en km) entre le village et l'événement
        d = haversine(ville["Latitude"], ville["Longitude"],
                      event["Latitude"], event["Longitude"])
        # Prédiction de PGA via la GMPE
        pga = gmpe(event["Magnitude"], d)
        
        # Si le PGA prédit dépasse le seuil, on ajoute le taux annuel de cet événement
        if pga >= PGA_threshold:
            total_annual_rate += event['annual_rate']
    
    # Calcul de la probabilité annuelle d'excéder le seuil (loi de Poisson)
    p_exceed = 1 - np.exp(-total_annual_rate)
    
    villes_pha.append({
        'Nom Village': ville["Nom Village"],
        'Latitude': ville["Latitude"],
        'Longitude': ville["Longitude"],
        'total_annual_rate': total_annual_rate,
        'P_exceed': p_exceed
    })

villes_pha_df = pd.DataFrame(villes_pha)



# Créer une carte centrée sur Mayotte
m = folium.Map(location=fani_maore, zoom_start=10)

# Ajouter le marqueur du volcan Fani Maoré
folium.Marker(
    location=fani_maore,
    popup="Volcan Fani Maoré",
    icon=folium.Icon(color="red", icon="volcano", prefix='fa')
).add_to(m)


# Créer les groupes de couches
square_group = folium.FeatureGroup(name="Grille de Risque")
circle_group = folium.FeatureGroup(name="Villes avec Probabilité")

# Fonction de style pour colorer les cellules en fonction du risque (pour la grille de risque)
def style_function(feature):
    risk = feature['properties']['risk']
    if risk is None:
        risk = 0
    if risk < 0.2:
        color = '#00ff00'  # vert
    elif risk < 0.25:
        color = '#ffff00'  # jaune
    elif risk < 0.5:
        color = '#ffa500'  # orange
    elif risk < 0.8:
        color = '#ff0000'  # rouge
    else:
        color = '#8b0000'  # rouge foncé
    return {
        'fillOpacity': 0.6,
        'weight': 0.5,
        'color': color
    }

# Filtrer les cellules avec un risque inférieur à 0.0001
grid_filtered = grid[grid['risk'] >= 0.0001].round(4)

# Ajouter la grille filtrée dans le groupe "Grille de Risque"
folium.GeoJson(
    grid_filtered.to_json(),
    name="Grille de Risque",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(fields=['risk'], aliases=['Risque'])
).add_to(square_group)

# Ajouter des cercles pour chaque ville dans le groupe "Villes avec Probabilité"
min_radius = 700  # Rayon minimal (en mètres) pour que les cercles restent visibles
max_radius = 1200  # Rayon maximal pour les plus grandes probabilités
max_prob = villes_pha_df['P_exceed'].max()  # Trouver la probabilité maximale

for idx, row in villes_pha_df.iterrows():
    color = 'red' if row['P_exceed'] > 0.05 else 'green'
    radius = min_radius + (row['P_exceed'] / max_prob) * (max_radius - min_radius)
    
    folium.Circle(
        location=[row['Latitude'], row['Longitude']],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.4,
        popup=f"<u>{row['Nom Village']}</u><br>Probabilité annuelle: <b>{row['P_exceed']*100:.2f} %</b>"
    ).add_to(circle_group)

# Ajouter la légende
legend_html = '''
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 200px; height: 160px;
                 background-color: white; border:2px solid grey; z-index: 9999;
                 font-size: 12px; padding: 10px;">
     <b>Légende du risque</b><br>
     <i style="background: #00ff00; width: 18px; height: 18px; float: left;"></i> Faible (0 - 0.2)<br>
     <i style="background: #ffff00; width: 18px; height: 18px; float: left;"></i> Très faible (0.2 - 0.25)<br>
     <i style="background: #ffa500; width: 18px; height: 18px; float: left;"></i> Modéré (0.25 - 0.5)<br>
     <i style="background: #ff0000; width: 18px; height: 18px; float: left;"></i> Élevé (0.5 - 0.8)<br>
     <i style="background: #8b0000; width: 18px; height: 18px; float: left;"></i> Extrême (> 0.8)
     </div>
'''

m.get_root().html.add_child(folium.Element(legend_html))

# Ajouter les groupes de couches à la carte
square_group.add_to(m)
circle_group.add_to(m)

# Ajouter un contrôle de couches pour permettre de basculer entre les couches
folium.LayerControl().add_to(m)



# Affichage de la carte dans Streamlit
st.title(f"Carte des risques annuel de séisme dépassant un PGA de {PGA_threshold}")
st.markdown("""
    <style>
        iframe {
            height: 85vh;
            width: 100%;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)
folium_static(m,width=1500)
st.sidebar.markdown("""
###
| PGA (𝑔)         | Effets attendus                                              |
|------------------|-------------------------------------------------------------|
| < 0.02 g         | Aucun ressenti, pas de dégâts                               |
| 0.02 - 0.05 g    | Ressenti léger, pas de dégâts                               |
| 0.05 - 0.10 g    | Légères vibrations, possible chute d'objets                 |
| 0.10 - 0.20 g    | Petites fissures dans les murs, légers dommages aux bâtiments fragiles |
| 0.20 - 0.40 g    | Dommages modérés aux bâtiments non renforcés                |
| 0.40 - 0.60 g    | Dommages importants aux bâtiments en maçonnerie non renforcée |

""")
st.sidebar.markdown("""
## Détails des calculs et méthodes
""")

st.sidebar.markdown("""
### 1. Calcul de la distance (Formule de Haversine)
Pour déterminer la distance entre deux points à partir de leurs coordonnées (latitude et longitude), nous utilisons la formule de Haversine :
""")
st.sidebar.latex(r"""
a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)
""")
st.sidebar.latex(r"""
c = 2 \, \arctan2\left(\sqrt{a}, \sqrt{1 - a}\right)
""")
st.sidebar.latex(r"""
d = R \times c
""")
st.sidebar.markdown("""
où $ \phi_1 $ et $ \phi_2 $ sont les latitudes (en radians), $ \Delta \phi $ et $ \Delta \lambda $ représentent les différences de latitude et de longitude, et $ R $ est le rayon de la Terre (6371 km).
""")

st.sidebar.markdown("""
### 2. Grille de risque et calcul du risque par cellule
Les séismes filtrés sont attribués à des cellules d'une grille (ici, de dimension 4000m*4000m). Pour chaque cellule :
""")
st.sidebar.markdown("""
1. **Fréquence annuelle $ \lambda $** :
""")
st.sidebar.latex(r"""
\lambda = \frac{N}{T}
""")
st.sidebar.markdown("""
avec $ N $ le nombre d'événements et $ T $ le nombre d'années d'observation.
""")
st.sidebar.markdown("""
2. **Risque annuel** (loi de Poisson) :
""")
st.sidebar.latex(r"""
\text{Risque} = 1 - \exp(-\lambda)
""")

st.sidebar.markdown("""
### 3. Modèle GMPE simplifié pour le calcul du PGA
Le PGA (Peak Ground Acceleration) est estimé par :
""")
st.sidebar.latex(r"""
\text{PGA} = \exp\Big(a + b \cdot M - \ln(d + c)\Big)
""")
st.sidebar.markdown("""
avec $ M $ la magnitude, $ d $ la distance (en km), $ a = -1.5 $, $ b = 0.5 $ et $ c = 10.0 $.
""")

st.sidebar.markdown("""
### 4. Estimation du taux annuel et de la probabilité d’excéder un seuil de PGA
- **Taux annuel par événement** :
""")
st.sidebar.latex(r"""
\text{annual\_rate} = \frac{1}{\text{Nombre d'années d'observation}}
""")
st.sidebar.markdown("""
- **Probabilité d'excéder le seuil** :
""")
st.sidebar.latex(r"""
P_{\text{exceed}} = 1 - \exp\left(-\sum \text{annual\_rate}\right)
""")
