import os
import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from Organize_All_Data_v2 import first_match, SCENE_REGEX

load_dotenv()

FIG_HEIGHT = 700
FIG_WIDTH = 900
DATA_DIR = os.getenv("ORGANIZED_DATA_DIR")

def invalidate_export():
    """Désactive le bouton de téléchargement (à faire à chaque modif de la figure)"""
    if "export" in st.session_state:
        del st.session_state.export

####################################################################################################################
# 1. SELECTION DES DONNÉES
####################################################################################################################

st.set_page_config(layout="wide")
st.title("Trajectoires cellulaires")
st.subheader('Sélection des données')

# -------- utils --------

def list_dirs(path):
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ])

def list_csv(path):
    return sorted([
        f for f in os.listdir(path)
        if f.endswith("Position_Relative.csv")
    ])

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# -------- sélection hiérarchique --------

tools = list_dirs(DATA_DIR)

if not tools:
    st.error("Les dossiers Fiji / Imaris n'existent pas.")
    st.stop()

tool = st.selectbox("Outil", tools, on_change=invalidate_export)

tool_path = os.path.join(DATA_DIR, tool)
dates = list_dirs(tool_path)

if not dates:
    st.error(f"Le dossier de {tool} est vide.")
    st.stop()

date = st.selectbox("Date", dates, on_change=invalidate_export)

date_path = os.path.join(tool_path, date)
files = list_csv(date_path)

if not files:
    st.error("Aucun CSV trouvé pour cette date.")
    st.stop()

filename = st.selectbox("Fichier CSV", files, on_change=invalidate_export)

# -------- chargement --------

csv_path = os.path.join(date_path, filename)
mega_csv_path = os.path.join(date_path, "MEGA_" + date + ".csv")
df_pos = load_data(csv_path)
df_mega = load_data(mega_csv_path)

scene = int(first_match(SCENE_REGEX, csv_path))
df_meta = df_mega[(df_mega["DATE"]==date) & (df_mega["SCENE"]==scene)]

st.subheader("Metadonnées")

df_meta.rename(columns = {'AGE': 'AGE (SEMAINES)'})
st.table(df_meta[['CLASSE', 'SEXE', 'AGE', 'IMC', 'TRAITEMENT']].iloc[0].to_frame().T)

####################################################################################################################
# 2. SIDEBAR
####################################################################################################################

st.sidebar.header("Paramètres de la figure")
st.sidebar.subheader('Échelles')

# Graduations
tick_interval = st.sidebar.slider(
    "Intervalle des graduations",
    min_value = 1,
    max_value = 50,
    value =10,  # valeur par défaut 5
    step =1,
    on_change=invalidate_export
)

# Afficher la grille ou non
show_grid = st.sidebar.checkbox("Afficher la grille", value=True)

# Échelles des axes (zoom)
axis_range = st.sidebar.slider(
    "Échelle des axes X/Y",
    min_value=0,
    max_value=300,
    value=100,  
    step=10,
    on_change=invalidate_export
)

# Titres
st.sidebar.subheader('Titres')
x_title = st.sidebar.text_input("Titre de l'axe X", value="Position X (µm)", on_change=invalidate_export)
y_title = st.sidebar.text_input("Titre de l'axe Y", value="Position Y (µm)", on_change=invalidate_export)
fig_title = st.sidebar.text_input("Titre de la figure", value=f"Trajectoires – {date} - Scène {scene}", on_change=invalidate_export)

# Tailles de police
st.sidebar.subheader('Tailles de police')
title_font = st.sidebar.number_input("Titre de la figure", value=24, on_change=invalidate_export)
axis_title_font = st.sidebar.number_input("Titres des axes", value=24, on_change=invalidate_export)
axis_font = st.sidebar.number_input("Axes", value=18, on_change=invalidate_export)
legend_title_font = st.sidebar.number_input("Titre de la légende", value=24, on_change=invalidate_export)
legend_font = st.sidebar.number_input("Légende", value=18, on_change=invalidate_export)

####################################################################################################################
# 3. TRACÉ DE LA FIGURE
####################################################################################################################

st.subheader('Figure')

# Filtres
track_ids_all = sorted(df_pos["TrackID"].unique())
track_ids = st.multiselect(
    "TrackID",
    track_ids_all,
    default=track_ids_all
)

show_legend = st.checkbox("Afficher la légende", value=True)

# Filtrage
df_f = df_pos[df_pos["TrackID"].isin(track_ids)]

# Plot
fig = px.line(
    df_f,
    x="PositionX_Relative",
    y="PositionY_Relative",
    color="TrackID",
    width = FIG_WIDTH,
    height = FIG_HEIGHT,
    color_discrete_sequence=px.colors.qualitative.Dark24
)

# Appliquer l'échelle de la sidebar
# Appliquer ticks et grille
gridcolor ="rgba(128, 128, 128, 0.3)"
fig.update_xaxes(dtick=tick_interval, showgrid=show_grid, gridcolor=gridcolor, range=[-axis_range, axis_range])
fig.update_yaxes(dtick=tick_interval, showgrid=show_grid, gridcolor=gridcolor, scaleanchor="x", scaleratio=1, range=[-axis_range, axis_range])


fig.update_layout(
    xaxis_title = dict(text=x_title, font=dict(size=axis_title_font)), 
    yaxis_title = dict(text=y_title, font=dict(size=axis_title_font)),
    xaxis = dict(tickfont=dict(size=axis_font)),
    yaxis = dict(tickfont=dict(size=axis_font)),
    title = dict(text=fig_title, font=dict(size=title_font)),
    showlegend=show_legend, 
    legend=dict(
        title_text="TrackID",
        title_font=dict(size=legend_title_font),
        font=dict(size=legend_font),
        traceorder="normal"
    ),
    plot_bgcolor="white", 
    width=FIG_WIDTH, 
    height=FIG_HEIGHT
    )

st.plotly_chart(fig, use_container_width=False)

####################################################################################################################
# 4. EXPORT DE LA FIGURE
####################################################################################################################

if "export_data" not in st.session_state:
    st.session_state.export_data = None
    st.session_state.export_name = None
    st.session_state.export_mime = None


st.subheader('Exporter la figure')
figure_fname = f"trajectoires_{date}_scene{scene}"
export_format = st.selectbox("Format", ["PNG", "HTML"], on_change=invalidate_export)
c1, c2 = st.columns(2)

with c1:
    if st.button("Préparer l’export"):
        if export_format == "PNG":
            data = fig.to_image(format="png", scale=2)
            st.session_state.export = {
                "data": data,
                "name": figure_fname+".png",
                "mime": "image/png"
            }
        else:
            data = fig.to_html(full_html=True, include_plotlyjs="cdn")
            st.session_state.export = {
                "data": data,
                "name": figure_fname+".html",
                "mime": "text/html"
            }

with c2:
    if "export" in st.session_state:
        st.download_button(
            "Télécharger",
            data=st.session_state.export["data"],
            file_name=st.session_state.export["name"],
            mime=st.session_state.export["mime"]
        )


