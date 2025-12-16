import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from Organize_All_Data_v2 import first_match, SCENE_REGEX

load_dotenv()

FIG_HEIGHT = 700
FIG_WIDTH = 900
DATA_DIR = os.getenv("ORGANIZED_DATA_DIR")
TOOL_NAMES = {'Imaris': 'Data_Imaris_Organised', 'Fiji': 'Data_Fiji_Organised' }
COLUMNS_OF_INTEREST = ["DATE", "SCENE", "ESPECE", "TISSU", "AGE (SEMAINES)", "SEXE", "IMC", 'TRAITEMENT']

def invalidate_export():
    """Désactive le bouton de téléchargement (à faire à chaque modif de la figure)"""
    if "export" in st.session_state:
        del st.session_state.export

def list_dirs(path):
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ])

def list_csv(path):
    csvs = {}
    for f in os.listdir(path):
        if f.endswith("Position_Relative.csv"):
            scene = int(first_match(SCENE_REGEX, f))
            #scene = 'inconnu' if np.isnan(scene) else int(scene)
            csvs[scene] = f
    return csvs

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if 'metadata' not in st.session_state:
    st.session_state['metadata'] = []
if "dataframes" not in st.session_state:
    st.session_state["dataframes"] = []
if 'scenes' not in st.session_state:
    st.session_state['scenes'] = []
if "export_data" not in st.session_state:
    st.session_state.export_data = None
    st.session_state.export_name = None
    st.session_state.export_mime = None


####################################################################################################################
# 1. SELECTION DES DONNÉES
####################################################################################################################

st.set_page_config(layout="wide")
st.title("Trajectoires cellulaires")
st.subheader('Sélection des données')

# -------- sélection hiérarchique --------
c1, c2, c3, c4 = st.columns(4)

with c1:
    tools = list_dirs(DATA_DIR)
    if not tools:
        st.error("Les dossiers Fiji / Imaris n'existent pas.")
        st.stop()
    tool = st.selectbox("Outil", TOOL_NAMES.keys())
    tool_path = os.path.join(DATA_DIR, TOOL_NAMES[tool])


with c2:
    dates = list_dirs(tool_path)
    if not dates:
        st.error(f"Le dossier de {tool} est vide.")
        st.stop()
    date = st.selectbox("Date", dates)
    date_path = os.path.join(tool_path, date)

with c3:

    files = list_csv(date_path)

    if not files:
        st.error("Aucun CSV trouvé pour cette date.")
        st.stop()

    scene = int(st.selectbox("Scène / Image", files.keys()))
    file_path = os.path.join(date_path, files[scene])

with c4:
    if st.button("Ajouter cette scène", on_click=invalidate_export) and file_path not in st.session_state['scenes']:

        # Charge les nouvelles données
        df_pos = pd.read_csv(file_path)
        mega_csv_path = os.path.join(date_path, "MEGA_" + date + ".csv")
        df_mega = pd.read_csv(mega_csv_path)

        # Metadonnées
        df_meta = df_mega[(df_mega["DATE"]==date) & (df_mega["SCENE"]==scene)].iloc[0]
        df_meta["TOOL"] = tool
        df_meta = df_meta.rename({'AGE': 'AGE (SEMAINES)', "CLASSE": "TISSU"})
        df_meta = df_meta[COLUMNS_OF_INTEREST].astype(str)
        df_meta = df_meta.fillna("Unknown")


        st.session_state["dataframes"].append(df_pos)
        st.session_state["metadata"].append(df_meta)
        st.session_state['scenes'].append(file_path)

if st.session_state["metadata"]:
    # Concaténation des données
    combined_dfs = []
    combined_meta = []

    for idx, (df, meta) in enumerate(zip(st.session_state["dataframes"], st.session_state["metadata"])):
        # Ajouter la colonne ID à chaque dataframe avec l'index correspondant
        df_copy = df.copy()
        df_copy["SceneID"] = idx
        df_copy['TrackID'] = df_copy["TrackID"].apply(lambda x: str(idx) + '_' +str(x))
        combined_dfs.append(df_copy)
        
        # Convertir la Series metadata en DataFrame si nécessaire, ajouter l'ID
        meta_df = meta.to_frame().T if isinstance(meta, pd.Series) else meta.copy()
        meta_df["SceneID"] = idx
        combined_meta.append(meta_df)

    # Concaténer tout en un seul DataFrame
    df_pos = pd.concat(combined_dfs, ignore_index=True)
    df_meta = pd.concat(combined_meta, ignore_index=True)

####################################################################################################################
# 2. AFFICHAGE DES DONNÉES
####################################################################################################################

st.subheader("Liste des scènes sélectionnées")

if st.session_state["metadata"]:
    st.write("Le Scène ID ci-dessous a été ajouté pour identifier" + 
                "à quelle scène appartient chaque cellule, il a été également" +
                "ajouté en préfixe des TrackID de chaque cellule (voire section suivante)")

    # Ajouter une colonne temporaire pour les boutons
    df_meta_display = df_meta.copy()
    df_meta_display["Effacer"] = [""] * len(df_meta_display)

    # Affichage de l'en-tête
    header_cols = st.columns(len(df_meta_display.columns))
    for j, col_name in enumerate(df_meta_display.columns):
        header_cols[j].write(f"**{col_name}**")

    # Affichage des lignes
    for i, row in df_meta_display.iterrows():
        cols = st.columns(len(df_meta_display.columns))
        for j, col_name in enumerate(df_meta_display.columns):
            if col_name == "Effacer":
                if cols[j].button("❌", key=f"delete_{i}", on_click=invalidate_export):
                    st.session_state["dataframes"].pop(i)
                    st.session_state["metadata"].pop(i)
                    st.session_state["scenes"].pop(i)
                    st.rerun()
            else:
                cols[j].write(row[col_name])

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
fig_title = st.sidebar.text_input("Titre de la figure", value=f"Trajectoires des cellules", on_change=invalidate_export)

# Tailles de police
st.sidebar.subheader('Tailles de police')
title_font = st.sidebar.number_input("Titre de la figure", value=24, on_change=invalidate_export)
axis_title_font = st.sidebar.number_input("Titres des axes", value=24, on_change=invalidate_export)
axis_font = st.sidebar.number_input("Axes", value=18, on_change=invalidate_export)
legend_title_font = st.sidebar.number_input("Titre de la légende", value=24, on_change=invalidate_export)
legend_font = st.sidebar.number_input("Légende", value=18, on_change=invalidate_export)



####################################################################################################################
# 4. TRACÉ DE LA FIGURE
####################################################################################################################

if st.session_state['metadata']:

    st.subheader('Figure')

    # # Filtres
    track_ids_all = sorted(df_pos["TrackID"].unique())
    track_ids = st.multiselect(
        "TrackID",
        track_ids_all,
        default=track_ids_all
    )

    show_legend = st.checkbox("Afficher la légende", value=True)

    # Choix de coloration
    color_columns = ["TrackID"] + COLUMNS_OF_INTEREST
    color_choice = st.selectbox("Choisir la colonne pour colorer", color_columns, on_change=invalidate_export)
    df_plot = df_pos.copy()

    # Ajouter la colonne choisie pour colorer si ce n'est pas TrackID
    if color_choice in color_columns[1:]:
        df_plot[color_choice] = df_plot["SceneID"].map(
            df_meta.set_index("SceneID")[color_choice]
        )

    # Filtrage
    df_plot = df_plot[df_plot["TrackID"].isin(track_ids)].astype(str)


    # Plot
    fig = px.line(
        df_plot,
        x="PositionX_Relative",
        y="PositionY_Relative",
        color=color_choice,
        line_group='TrackID',
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
            title_text=color_choice,
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

if st.session_state['metadata']:

    st.subheader('Exporter la figure')
    st.write("Attention, pour les PNG, toutes les modifications faites à l'interieur du graphique ne " +
              "sont pas prises en compte dans l'export, seules celles faites avec les menus sur le "+
              "coté et au dessus de la figure le seront.")

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


