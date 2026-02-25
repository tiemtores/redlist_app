import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, Fullscreen
from streamlit_folium import st_folium
import numpy as np
import io

# Configuration de la page
st.set_page_config(page_title="UICN Stratégie - Aide à la Décision", layout="wide", page_icon="📈")

# Couleurs officielles de l'UICN
COLORS = {
    'EX': '#000000', 'EW': '#542344', 'CR': '#d40000', 
    'EN': '#f57e20', 'VU': '#f9d342', 'NT': '#00cc66', 
    'LC': '#006666', 'DD': '#d1d1d1'
}

COUNTRY_DATA = {
    "Bénin": [9.3, 2.3], "Burkina Faso": [12.2, -1.6], "Cap-Vert": [16.0, -24.0],
    "Côte d'Ivoire": [7.5, -5.5], "Gambie": [13.4, -15.3], "Ghana": [7.9, -1.0],
    "Guinée": [9.9, -9.7], "Guinée-Bissau": [11.8, -15.2], "Libéria": [6.4, -9.4],
    "Mali": [17.6, -4.0], "Mauritanie": [21.0, -10.9], "Niger": [17.6, 8.1],
    "Nigéria": [9.1, 8.7], "Sénégal": [14.5, -14.4], "Sierra Leone": [8.5, -11.8],
    "Togo": [8.6, 0.8], "Toute l'Afrique de l'Ouest": [14.0, -2.0]
}

@st.cache_data
def load_data():
    categories = ['EX', 'EW', 'CR', 'EN', 'VU', 'NT', 'LC', 'DD']
    fauna = ["Lion d'Afrique", "Éléphant de savane", "Chimpanzé", "Hippopotame nain", "Lamantin", "Lycaon", "Tortue luth", "Girafe", "Calao", "Requin-marteau"]
    flora = ["Baobab africain", "Karité", "Fromager", "Acacia", "Palmier à huile", "Khaya (Acajou)", "Ebène de l'Ouest", "Cola"]
    countries = list(COUNTRY_DATA.keys())[:-1]
    
    data = []
    for _ in range(6000):
        is_fauna = np.random.choice([True, False], p=[0.6, 0.4])
        sp_name = np.random.choice(fauna if is_fauna else flora)
        c_name = np.random.choice(countries)
        cat = np.random.choice(categories, p=[0.01, 0.01, 0.15, 0.20, 0.25, 0.13, 0.20, 0.05])
        center = COUNTRY_DATA[c_name]
        data.append({
            'Espèce': sp_name, 'Type': 'Faune' if is_fauna else 'Flore',
            'Statut': cat, 'Année': np.random.randint(2015, 2026),
            'Pays': c_name, 'Lat': center[0] + np.random.uniform(-2, 2), 
            'Lon': center[1] + np.random.uniform(-2, 2),
            'Intensité': 1.0 if cat == 'CR' else (0.7 if cat == 'EN' else 0.4) if cat in ['CR', 'EN', 'VU'] else 0
        })
    return pd.DataFrame(data)

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #d40000;'>🛡️ STRATÉGIE DE CONSERVATION : AFRIQUE DE L'OUEST</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; border-bottom: 2px solid #d40000; padding-bottom: 10px;'>Outil d'Aide à la Décision Stratégique</h4>", unsafe_allow_html=True)

# --- STYLE ---
st.markdown("""
<style>
    .kpi-card { padding: 15px; border-radius: 12px; color: white; text-align: center; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .species-box { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #d40000; margin-bottom: 10px; }
    .metric-small { font-size: 0.8rem; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

# --- DATA & FILTERS ---
df = load_data()
st.sidebar.header("🎛️ Paramètres")
selected_country = st.sidebar.selectbox("Pays Cible", list(COUNTRY_DATA.keys()), index=len(COUNTRY_DATA)-1)
year_range = st.sidebar.slider("Période d'Analyse", 2015, 2025, (2018, 2025))

st.sidebar.divider()
st.sidebar.markdown("""
**📚 Sources & Référentiels :**
- **UICN Red List** (2025) : Statut de conservation mondial.
- **WDPA / UNEP-WCMC** : Réseau des aires protégées.
- **CEDEAO** : Cadre biodiversité régional.
- **KBA Partnership** : Zones clés de biodiversité.
""")

filtered_df = df[df['Année'].between(year_range[0], year_range[1])]
regional_df = filtered_df.copy() # Garder pour le benchmark
if selected_country != "Toute l'Afrique de l'Ouest":
    filtered_df = filtered_df[filtered_df['Pays'] == selected_country]

# --- KPI ---
st.write("")
c1, c2, c3, c4 = st.columns(4)
total = len(filtered_df)
urgent = len(filtered_df[filtered_df['Statut'] == 'CR'])
fauna_t = len(filtered_df[(filtered_df['Type'] == 'Faune') & (filtered_df['Statut'].isin(['CR', 'EN', 'VU']))])
flora_t = len(filtered_df[(filtered_df['Type'] == 'Flore') & (filtered_df['Statut'].isin(['CR', 'EN', 'VU']))])

def kpi_box(color, title, value, sub):
    st.markdown(f'<div class="kpi-card" style="background-color: {color};"><small>{title}</small><br><b style="font-size: 25px;">{value}</b><br><span class="metric-small">{sub}</span></div>', unsafe_allow_html=True)

with c1: kpi_box("#d40000", "URGENCE (CR)", urgent, f"{(urgent/total*100):.1f}% du total" if total > 0 else "0%")
with c2: kpi_box("#27ae60", "FAUNE MENACÉE", fauna_t, f"{fauna_t} espèces")
with c3: kpi_box("#f39c12", "FLORE MENACÉE", flora_t, f"{flora_t} espèces")
with c4: kpi_box("#2c3e50", "TOTAL ÉVALUÉ", total, f"Période {year_range[0]}-{year_range[1]}")

# --- TOP MENACÉS & BENCHMARK ---
st.divider()
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("🚨 Focus : Espèces les plus Critiques")
    cf, cfl = st.columns(2)
    def top_sp(df, t):
        s = df[(df['Type'] == t) & (df['Statut'] == 'CR')]
        return s['Espèce'].value_counts().idxmax() if not s.empty else "N/A"
    cf.markdown(f'<div class="species-box"><b>🐾 Top Faune :</b><br><span style="color:#d40000; font-size:18px;">{top_sp(filtered_df, "Faune")}</span></div>', unsafe_allow_html=True)
    cfl.markdown(f'<div class="species-box"><b>🌿 Top Flore :</b><br><span style="color:#d40000; font-size:18px;">{top_sp(filtered_df, "Flore")}</span></div>', unsafe_allow_html=True)

with col_right:
    st.subheader("📊 Benchmark Régional")
    if selected_country != "Toute l'Afrique de l'Ouest":
        # Comparaison % de menaces Pays vs Région avec sécurité division par zéro
        reg_threat_avg = (len(regional_df[regional_df['Statut'].isin(['CR', 'EN', 'VU'])]) / len(regional_df)) * 100 if len(regional_df) > 0 else 0
        country_threat = (len(filtered_df[filtered_df['Statut'].isin(['CR', 'EN', 'VU'])]) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        
        fig_bench = go.Figure(go.Indicator(
            mode = "gauge+number", value = country_threat,
            title = {'text': "% Menace vs Moyenne Régionale", 'font': {'size': 14}},
            gauge = {
                'axis': {'range': [0, 100]}, 
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': reg_threat_avg}
            }
        ))
        fig_bench.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_bench, use_container_width=True)

# --- CARTE ---
st.subheader(f"🗺️ Spatialisation & Hotspots : {selected_country}")
# Coordonnées sécurisées
map_center = COUNTRY_DATA.get(selected_country, [14.0, -2.0])
zoom = 5 if selected_country == "Toute l'Afrique de l'Ouest" else 7

m = folium.Map(location=map_center, zoom_start=zoom, tiles="cartodbpositron")
m.add_child(Fullscreen())

# WDPA
folium.WmsTileLayer(url="https://gis.unep-wcmc.org/arcgis/services/wdpa/wdpa/MapServer/WmsServer", layers="1", name="Réseau WDPA", fmt="image/png", transparent=True, overlay=True).add_to(m)

# Hotspots
fg = folium.FeatureGroup(name="Hotspots de Menaces", show=False)
heat_data = filtered_df[filtered_df['Statut'].isin(['CR', 'EN', 'VU'])][['Lat', 'Lon', 'Intensité']].values.tolist()
if heat_data: HeatMap(heat_data, radius=20, blur=15, min_opacity=0.3).add_to(fg)
fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width="100%", height=500, key=f"map_display_{selected_country}")

# --- TENDANCES ET INVENTAIRE ---
st.divider()
st.subheader("📈 Tendances & Inventaires Détaillés")
tab1, tab2, tab3 = st.tabs(["📊 Global", "🐾 Faune", "🌿 Flore"])

def process_tab(t_filter=None):
    d = filtered_df if t_filter is None else filtered_df[filtered_df['Type'] == t_filter]
    c_a, c_b = st.columns([1, 1])
    with c_a:
        # Tendance des autres statuts (Bar chart)
        st.write("**Répartition globale des statuts (Tendance des autres) :**")
        cat_order = ['EX', 'EW', 'CR', 'EN', 'VU', 'NT', 'LC', 'DD']
        stats_dist = d['Statut'].value_counts().reindex(cat_order).fillna(0)
        fig_bar = px.bar(stats_dist, color=stats_dist.index, color_discrete_map=COLORS, template="plotly_white")
        fig_bar.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)
    with c_b:
        # Évolution temporelle
        st.write("**Évolution temporelle des évaluations :**")
        evol = d.groupby(['Année', 'Statut']).size().reset_index(name='N')
        fig_line = px.line(evol, x='Année', y='N', color='Statut', color_discrete_map=COLORS, template="plotly_white")
        fig_line.update_layout(height=300)
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.write("**Inventaire des espèces cibles :**")
    st.dataframe(d[['Espèce', 'Statut', 'Année', 'Pays']].sort_values(by='Statut'), use_container_width=True, height=300)
    csv = d.to_csv(index=False).encode('utf-8')
    st.download_button(f"📥 Télécharger la liste ({t_filter if t_filter else 'Global'})", csv, f"uicn_export_{t_filter if t_filter else 'global'}.csv", "text/csv")

with tab1: process_tab()
with tab2: process_tab("Faune")
with tab3: process_tab("Flore")

# --- NOTE D'ORIENTATION ---
st.divider()
st.subheader(f"📝 Note d'Orientation Stratégique : {selected_country}")

if not filtered_df.empty:
    # Analyse dynamique
    top_threat_type = filtered_df[filtered_df['Statut'].isin(['CR', 'EN', 'VU'])]['Type'].value_counts().idxmax()
    urgent_ratio = (len(filtered_df[filtered_df['Statut'] == 'CR']) / len(filtered_df)) * 100
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"""
        **Priorités d'Action :**
        1. **Focus Biome :** La conservation doit porter en priorité sur la **{top_threat_type}** qui concentre le plus grand nombre d'espèces menacées.
        2. **Urgence Critique :** {urgent:.0f} espèces (soit {urgent_ratio:.1f}%) nécessitent des plans de sauvegarde immédiats (Statut CR).
        3. **Aires Protégées :** Il est recommandé d'auditer la couverture WDPA sur les zones de hotspots identifiées.
        """)
    with col_b:
        st.warning(f"""
        **Recommandations Politiques :**
        - **Renforcement législatif :** Accroître la protection des habitats critiques pour les espèces de type **{top_threat_type}**.
        - **Suivi des Données :** {len(filtered_df[filtered_df['Statut'] == 'DD'])} espèces sont classées en 'Données Déficientes' ; un inventaire de terrain est nécessaire.
        - **Coopération Régionale :** Harmoniser les efforts avec les pays voisins pour les espèces transfrontalières.
        """)
else:
    st.write("Sélectionnez une période ou un pays avec des données pour générer l'analyse.")
