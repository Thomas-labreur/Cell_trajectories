#####################################################################################################################
# 0. IMPORT DES LIBRAIRIES
#####################################################################################################################

import re                           # Manipulation d'expressions régulières
import numpy as np                  # Mathématiques (sqrt, mean, std, ..)
import pandas as pd                 # Manipulation de données tabulaires
import os                           # Manipulation des répertoires
from os.path import isfile, join
from fnmatch import fnmatch
from typing import List, Tuple
from dotenv import load_dotenv

#####################################################################################################################
# 1. FONCTIONS UTILITAIRES
#####################################################################################################################

def first_match(patterns, text):
    """Renvoie le premier match dans le texte de l'une des regex fournies."""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(0)
    return np.nan

def clean(df: pd.DataFrame, unnamed: bool = True) -> pd.DataFrame: 
    """ Retire les espaces dans les noms de colonne et efface les colonnes unnamed"""
    df = df.rename(columns=lambda c: c.replace(" ", ""))
    if unnamed:
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df

def open_metadata():
    """Lis le dataframe récap rempli par Sylvie qui contient les information complémentaires"""
    df_meta = pd.read_excel(PATHINFOTABLE).rename(columns={'Scene/Image': 'Scene'})
    df_meta = df_meta.dropna(subset=['Scene'])
    df_meta['Scene'] = df_meta['Scene'].astype(str).str.extract(r'(\d+)$').astype(int)
    return df_meta

def open_imaris_csv(parameter: str, scene_dir: str) -> Tuple[pd.DataFrame, str]:
    """
    Ouvre le csv du répertoire scene_dir qui contient le nom du paramètre souhaité. 
    Renvoie aussi le nom du fichier.
    """
    # Cherche le fichier de la scène qui contient le paramètre dans son nom
    file_path = next(
        (os.path.join(root, f) for root, _, files in os.walk(scene_dir) for f in files if parameter in f),
        None,
    )

    # Si paramètre manquant, afficher avertissement
    if file_path is None:
        print("_________________________________________________________________")
        print("            /!\\ ATTENTION PARAMETRE MANQUANT /!\\")
        print(f"Le fichier du parametre {parameter} n'a pas été trouvé dans le répertoire:")
        print(scene_dir)
        print("_________________________________________________________________")
        return None, None
    
    # Ouvre le csv (si en-tête bizarre, il faut sauter les 3 premieres lignes)
    df_csv = pd.read_csv(file_path, on_bad_lines = 'skip', encoding = "latin", sep = ";")
    if len(df_csv.columns) < 2:
        df_csv = pd.read_csv(file_path, on_bad_lines = 'skip', skiprows = 3, 
                             encoding = "latin", engine = 'python')

    # Renvoi le dataframe et le chemin du fichier
    return df_csv, file_path

def aggregate_by_id(df: pd.DataFrame, column: str, name: str, agg:str = 'mean', factor: int = 1, dropAll = True) -> pd.DataFrame:
    """
    Aggrège les données d'une colonne d'un dataframe par TrackID avec mise à l'échelle si besoin.
    
    Arguments:
        df (pd.DataFrame): un dataframe avec une colonne 'TrackID'
        column (str): nom d'une colonne de df
        name (str): nom de la future colonne agrégée
        agg (str, 'mean' ou 'sum', default 'mean'): type d'agrégation
        faction (int, default 1): facteur multiplicatif à appliquer sur la colonne agrégée
        dropAll (bool, default True): Si True, ne renvoie que les colonnes 'TrackID' et agrégée

    Renvoi :
        pd.DataFrame: dataframe contenant 'TrackID' et l'agrégation, et éventuellement la première 
            ligne de chaque TrackID sur les autres colonnes de df
    """

    # Initalise le dataframe
    dfc = df[['TrackID']].copy() if dropAll else df.copy()
    dfc = dfc.drop_duplicates(subset='TrackID')

    # Change les valeurs en float
    if "," in str(df[column][0]):
        df[column] = [float(l.replace(",",".")) for l in df[column]]

    # Calcule l'aggrégation par TrackID
    if agg == 'mean':
        df_agg = factor * df.groupby('TrackID')[column].mean().rename(name)
    elif agg == 'sum':
        df_agg = factor * df.groupby('TrackID')[column].sum().rename(name)
    else:
        raise ValueError("Le paramètre agg ne supporte que 'sum' et 'mean'.")

    # Joint les dataframes
    return dfc.merge(df_agg, on='TrackID', how="left")

def relative_positions(df_p:pd.DataFrame, unnamed: bool=True, changeUnit: bool=False):
    """
    Ajoute les colonnes des positions relatives dans le dataframe.

    Arguments:
        df_p (pd.DataFrame): un dataframe avec des colonnes PositionX, PositionY, TrackID, Time
        unnamed (bool, default True): efface les colonnes unnamed si True
        changeUnit(bool, default False): converti les positions de pixel à micromètre 
            Mettre True pour Fiji, et dans ce cas seules les colonnes PositionX, PositionY, 
            TrackID, Time sont renvoyées.     
    Renvois
        pd.Dataframe: le dataframe avec les colonnes positions relatives
    """

    # Nettoyage
    df_p = clean(df_p, unnamed).dropna()

    # Position initiale par TrackID
    df_init = (
        df_p.sort_values(['TrackID', 'Time'])
            .drop_duplicates('TrackID', keep='first')
            .set_index('TrackID')
    )

    # Récupération des positions absolues
    x0 = df_p['TrackID'].map(df_init['PositionX'])
    y0 = df_p['TrackID'].map(df_init['PositionY'])

    # Conversion éventuelle (avec fiji l'unité est le pixel, et 1 px = 0.454 micromètre)
    factor = 0.454 if changeUnit else 1.0

    # Colonnes de position relative
    df_p['PositionX_Relative'] = (df_p['PositionX'] - x0) * factor
    df_p['PositionY_Relative'] = (df_p['PositionY'] - y0) * factor

    # Colonnes finales 
    return df_p if changeUnit else df_p[[ 'Time', 'TrackID', 'PositionX_Relative', 'PositionY_Relative']] 

def check_issues(list_df: List[pd.DataFrame], list_paths: List[str], xp: str, scene: int, fiji:bool = False): 
    """ 
    Affiche des messages dans la console en cas de problème (doublons, saut dans le temps).

    Arguments
        list_df (List[pd.DataFrame]): liste des dataframe issus des csv d'une même scene
        list_fpaths (List[str]): liste des noms de fichiers de ces csv
        scene (int): le numéro de la scène
        xp (str): date de l'expérience
        fiji (bool, default False): Si true, vérifie les doublons inter-fichiers (utile pour Fiji)
    """    
    id_already_seen = {}
    for i, df in enumerate(list_df):
        for id in df['TrackID'].drop_duplicates():

            # Récupère la liste des temps et la durée totale de l'expérience
            times = df[df['TrackID']==id]['Time']
            unique_times = times.drop_duplicates()
            times = list(times)
            duration = times[-1] - (times[0]-1)
            filename = os.path.basename(list_paths[i])

            # S'il y a des doublons de temps dans un fichier, affiche un avertissement
            if len(times) != len(unique_times):
                print("_________________________________________________________________")
                print("            /!\\ ATTENTION PROBLEME DE DOUBLONS /!\\")
                print(f"Expérience {xp} | Scène {scene} | Cellule {id}")
                print(f"Fichier {filename}")
                print("_________________________________________________________________")

            # S'il y a un saut de temps dans un fichier, affiche un avertissement
            if len(times) < duration - 10:
                hole_size= duration-len(times)
                print("_________________________________________________________________")
                print("            /!\\ ATTENTION PROBLEME TIME MANQUANT /!\\")
                print(f"Expérience {xp} | Scène {scene} | Cellule {id} | {hole_size} Temps manquants")
                print(f"Fichier {filename}")
                print("_________________________________________________________________")

            # Pour fiji, on vérifie aussi les doublons entre les fichiers
            if fiji and id in id_already_seen.keys():
                doubled_file = os.path.basename(list_paths[id_already_seen[id]])
                print("_________________________________________________________________")
                print("            /!\\ ATTENTION PROBLEME DE DOUBLONS /!\\")
                print(f"Expérience {xp} | Scène {scene} | Cellule {id}")
                print(f"Cette Cellule est présente dans deux fichiers: {filename} et {doubled_file}]")
                print("_________________________________________________________________")
            else:
                id_already_seen[id] = i

#####################################################################################################################
# 2. CREATION DE DATAFRAMES
#####################################################################################################################

def create_df_paths(xp_dir: str, unwanted: List[str]) -> pd.DataFrame:
    """
    Génères un dataframe contenant les informations de chaque fichiers csv d'une expérience.

    Arguments:
        xp_dir (str): le chemin du dossier de l'expérience (généralement identifié par sa date)
        unwanted (List[str]): Les chemins contenant l'une de ces chaînes seront exclus
    
    Renvois:
        pd.DataFrame: un dataframe ayant une ligne par fichier csv, donnant le chemin, l'expérience (date),
            la scene et le logiciel (IMARIS ou FIJI)
    """

    # Chemins de tous les csv du dossier de l'expérience
    csv_paths = [os.path.join(dir, f) for dir,_,filenames in os.walk(xp_dir) for f in filenames if '.csv' in f]

    # Crétaion du dataframe
    c = ["Experience", "Scene", "Software", "Path"]
    df = pd.DataFrame(columns = c)

    # Recherche des chemins indesirables et élimination des doublons
    l = list(set([p for p in csv_paths for i in unwanted if i in p]))

    # Supprimmer les indésirables
    for i in l :
        csv_paths.remove(i)

    # Remplir le dataframe
    df["Path"] = csv_paths
    df["Experience"] = [first_match(DATE_REGEX, p) for p in df["Path"]]
    df["Scene"] = [first_match(SCENE_REGEX, p) for p in df["Path"]]
    df["Software"] = ["IMARIS" if "Statistics" in p else "FIJI" for p in df["Path"]]
    
    return df

def get_fiji_pos_dataframe(scene_dir: str, xp: str, scene: int) -> pd.DataFrame: 
    """
    Créée un dataframe en concaténant les csv du dossier FIJI fourni, en vérifiant les problèmes
    de doublons et saut dans le temps, et en ajoutant les colonnes de positions relatives.

    Arguments:
        scene_dir (str): chemin vers le dossier de la scène contenant les csv
        xp (str): date de l'expérience
        scene (int): numéro de la scène

    Renvois: 
        pd.DataFrame: le dataframe décri ci-dessus
    """

    list_df, list_paths = [], []
    for f in os.listdir(scene_dir):
        # Obtenir la liste des csv du répertoire scene_dir
        path = os.path.join(scene_dir, f)
        if isfile(path) and path.endswith(".csv"):

            # Lecture du fichier csv (éventuellement avec le séparateur européen)
            df_csv = pd.read_csv(path, encoding="latin")
            df_csv = pd.read_csv(path, encoding = 'latin', sep = ";") if len(df_csv.columns) != 7 else df_csv
            df_csv.columns = ['TrackID', 'Time', 'PositionX', 'PositionY', 'Distance_µm', 'Speed_µm/min', 'PixelValue']

            # Ajuste les valeurs de TrackID
            if "à" in path:
                diff_trackID = int(re.findall(r'\d+', path)[-1]) - df_csv['TrackID'].max()
                df_csv['TrackID'] += diff_trackID

            # Remet à 0 des premiers temps de chaque cellule (-1 à 0)
            df_csv.loc[(df_csv['Distance_µm'] == -1) & (df_csv['Time'] == 1), 'Distance_µm'] = 0
            df_csv.loc[(df_csv['Speed_µm/min'] == -1) & (df_csv['Time'] == 1), 'Speed_µm/min'] = 0

            list_df.append(df_csv)
            list_paths.append(path)

    # Affiche un message d'avertissement si DOUBLONS ou SAUT DE TEMPS
    check_issues(list_df, list_paths, xp, scene, fiji=True)

    # On forme qu'un seul fichier pour une scène et on renomme les index
    df = pd.concat(list_df)
    df = df.sort_values(by = ['TrackID', 'Time'])
    df.reset_index(level = None, drop = True, inplace=True)

    # Ajout de la position relative
    df = relative_positions(df, unnamed = False, changeUnit = True)

    return df

def get_imaris_pos_dataframe(scene_dir: str, xp: str, scene: int) -> pd.DataFrame: 
    """
    Ouvre le csv IMARIS du paramètre 'Position' de scene_dir en vérifiant les problèmes
    de doublons et saut dans le temps, et en ajoutant les colonnes de positions relatives.
    
    Arguments:
        scene_dir (str): chemin vers le dossier de la scène contenant les csv
        xp (str): date de l'expérience
        scene (int): numéro de la scène

    Renvois: 
        pd.DataFrame: le dataframe décri ci-dessus
        """

    # Ouvre le fichier et calcule les positions relatives
    df_pos, file_path = open_imaris_csv('Position', scene_dir)
    df_pos = relative_positions(df_pos)

    # Affiche un message d'avertissement si DOUBLONS ou SAUT DE TEMPS
    check_issues([df_pos], file_path, xp, scene) 
    return df_pos

def get_fiji_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Créée un dataframe synthétisant le dataframe des données d'une scène Fiji.
    
    Arguments:
        df (pd.DataFrame): dataframe des données fiji (sortie de get_fiji_scene_dataframe)

    Renvois:
        pd.DataFrame: dataframe résumant ces données
    """

    # Initialise le dataframe résumé
    df_sum = df[['TrackID']].drop_duplicates().copy()

    df_init = (
        df.sort_values(['TrackID', 'Time'])
          .drop_duplicates(subset='TrackID', keep='last')
    )

    # Vitesse
    grouped_speed = df.groupby('TrackID')['Speed_µm/min']
    df_sum['TrackSpeedMean_µm/min'] = grouped_speed.mean().values
    df_sum['TrackSpeedMin_µm/min'] = grouped_speed.min().values
    df_sum['TrackSpeedMax_µm/min'] = grouped_speed.max().values
    df_sum['TrackSpeedStdDev_µm/min'] = grouped_speed.std(ddof=0).values

    # Variation = sd / mean sauf si mean=0 → 0
    mean_vals = df_sum['TrackSpeedMean_µm/min'].values
    sd_vals = df_sum['TrackSpeedStdDev_µm/min'].values
    df_sum['TrackSpeedVariation'] = np.where(mean_vals == 0, 0, sd_vals / mean_vals)

    # Distance totale
    grouped_dist = df.groupby('TrackID')['Distance_µm']
    df_sum['Dist_Totale_µm'] = grouped_dist.sum().values
    df_sum['Dist_instant_Moyenne_µm'] = grouped_dist.mean().values

    # Distance à l'origine
    df_sum['Dist_Origine_µm'] = np.sqrt(
        df_init['PositionX_Relative'] ** 2 +
        df_init['PositionY_Relative'] ** 2
    ).values

    # Persistance
    tot = df_sum['Dist_Totale_µm'].values
    orig = df_sum['Dist_Origine_µm'].values
    df_sum['Persistance'] = np.where(tot == 0, np.nan, orig / tot)

    # Nombre de spots
    df_sum['TrackNumberofSpots'] = df.groupby('TrackID')['Time'].count().values

    return df_sum

def get_imaris_summary_dataframe(scene_dir: str, xp: str, scene: int)  -> pd.DataFrame:
    """
    Créée un dataframe résumé en concaténant les csv IMARIS du dossier fourni, 
    en vérifiant les problèmes de doublons et saut dans le temps.

    Arguments:
        scene_dir (str): chemin vers le dossier de la scène contenant les csv
        xp (str): date de l'expérience
        scene (int): numéro de la scène

    Renvois: 
        pd.DataFrame: le dataframe décri ci-dessus
    """

    # Liste des paramètres à récupérer (1 csv par paramètre)
    parameters = ['Acceleration', 'Speed.', 'Track_Speed_Mean', 'Track_Speed_Min', 'Track_Speed_Max', 
                 'Track_Speed_StdDev', 'Track_Speed_Variation', 'Displacement_Delta_Length', 
                 'Track_Displacement_Length', 'Track_Straightness', 'Track_Number_of_Branches', 
                 'Track_Ar1_Mean', 'Track_Duration', 'Track_Number_of_Spots', 'Displacement^2']
    
    list_df, to_check, to_check_fnames = [], [], []
    for p in parameters:

        # Ouvre le csv qui contient le nom du paramètre
        df_csv, file_path = open_imaris_csv(p, scene_dir)
        if df_csv is None:
            continue

        # Nettoyage du fichier
        df_csv = clean(df_csv)
        df_csv2 = None

        # Pour ces 4 là, on vérifiera les doublons et saut de temps
        if p in ['Acceleration', 'Speed.', 'Displacement_Delta_Length', 'Displacement^2']:
            to_check.append(df_csv.copy())
            to_check_fnames.append(file_path)

        # Calcul de la moyenne et conversion en minute
        if p == 'Acceleration':
            df_csv = aggregate_by_id(df_csv, p, 'Acc_Moyenne_µm/m2', factor=3600)
            
        # Calcul de la moyenne et conversion en minute
        elif p == 'Speed.':
            speed_col =  next(c for c in df_csv.columns if "Speed" in c)
            df_csv = aggregate_by_id(df_csv, speed_col, 'Speed_Moyenne_µm/min', factor=60)

        # Calcul de la moyenne et de la moyenne sur la première heure (les 20 premiers temps)
        elif p == 'Displacement^2':
            df_csv2 = df_csv[df_csv['Time']<21].copy()
            df_csv2 = aggregate_by_id(df_csv2, p, 'Displacement^2_60min_Moyenne_µm2')
            df_csv = aggregate_by_id(df_csv, p, 'Displacement^2_Moyenne_µm2')

        # Calcul des distances moyennes et totales
        elif p == 'Displacement_Delta_Length':
            ddl_col = 'DisplacementDeltaLength'
            df_csv2 = aggregate_by_id(df_csv, ddl_col, 'Dist_instant_Moyenne_µm')
            df_csv = aggregate_by_id(df_csv, ddl_col, 'Dist_Totale_µm', agg='sum')

        # Pour les autres, on supprime juste les colonnes inutiles
        else: 
            col = p.replace("_","")
            df_csv = df_csv[['ID', col]].rename(columns= {'ID': 'TrackID'})

            # Conversion en minute pour les vitesses
            if 'Speed' in col and not 'Variation' in col:
                df_csv[col] = 60*df_csv[col]
                df_csv = df_csv.rename(columns={col: col+"_µm/min"})
                
            # Noms de colonnes qui matchent avec Fiji
            df_csv = df_csv.rename(columns={
                'TrackStraightness': 'Persistance',
                'TrackDisplacementLength': 'Dist_Origine_µm'
            })

        # Ajout du/des dataframe(s) à la liste
        list_df.append(df_csv)
        if df_csv2 is not None:
            list_df.append(df_csv2)
        

    # Affichage d'un message d'avertissement en cas de doublon ou saut de temps
    # Uniquement pour 'Acceleration', 'Speed.', 'Displacement_Delta_Length' et 'Displacement^2'
    check_issues(to_check, to_check_fnames, xp, scene)

    # Merge dataframes
    df_merged = list_df[0]
    for df in list_df[1:]:
        df_merged = df_merged.merge(df, on="TrackID", how='outer')

    return df_merged

#Création d'un mega DF comprenant toutes les cellules d'une expérience ayant pour chemin de dossier path
def get_mega_dataframe(xp_res_dir: str, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Concatène les dataframe Résumé par Celule d'une expérience donnée en un 'mega' dataframe.
    Ajoute des colonnes d'informations complémentaires contenues dans le fichier PATHINFOTABLE
    
    Arguments
        xp_res_dir (str): le chemin du répertoire des résultats l'expérience (sortie du code précédent)
        df_meta (pd.Dataframe): dataframe d'informations complémentaires

    Renvois:
        pd.DataFrame: le 'mega' dataframe de l'expérience
    """

    list_df = []

    for filename in os.listdir(xp_res_dir):
        if filename.endswith("Par_Cellule.csv"):

            # Ouvre le DataFrame Résumé par cellule
            df_sum = pd.read_csv(os.path.join(xp_res_dir, filename))
            filename = filename[:-4]

            # Ajoute la colonne CLASSE
            res_regex = re.search(CLASS_REGEX, filename)
            class_name = res_regex.group(0) if res_regex else np.nan
            df_sum['CLASSE'] = class_name

            # Ajoute la colonne SCENE
            res_regex = re.search(SCENE_REGEX[1], filename)
            scene = int(res_regex.group(0)) if res_regex else np.nan
            df_sum['SCENE'] = scene

            # Ajout de la date
            date = os.path.basename(xp_res_dir)
            df_sum['DATE'] = date

            # Ajout de l'age avec conversion en semaines
            age = df_meta.loc[(df_meta["Date"] == date ) & (df_meta['Scene'] == scene), ['Age']]
            res_regex = re.search(AGE_REGEX, str(age))
            if res_regex:
                age, unit = res_regex.group(1), res_regex.group(2)
                factor =  4.34524 if unit=='mois' else 52.1429 if unit=='an' else 1
                age = round(factor*int(age))

            # Si l'age n'est pas spécifié ou mal spécifié, affiche un avertissement
            else:
                age = np.nan
                print("_________________________________________________________________")
                print("            /!\\ ATTENTION PROBLEME D'AGE' /!\\")
                print(f"Age non spécifié dans le fichier {os.path.basename(PATHINFOTABLE)}")
                print(f"Expérience {date} | Scène {scene} ")
                print("_________________________________________________________________")
            df_sum['AGE'] = age

            # Ajout du sexe, IMC et Traitement
            df_sum['SEXE'] = df_meta.loc[(df_meta["Date"] == date ) & (df_meta['Scene'] == scene), ['Sexe']].values[0][0]
            df_sum['ESPECE'] = df_meta.loc[(df_meta["Date"] == date ) & (df_meta['Scene'] == scene), ['Espèce']].values[0][0]
            df_sum['IMC'] = df_meta.loc[(df_meta["Date"] == date ) & (df_meta['Scene'] == scene), ['IMC']].values[0][0]
            df_sum['TRAITEMENT'] = df_meta.loc[(df_meta["Date"] == date ) & (df_meta['Scene'] == scene), ['Traitement']].values[0][0]
            

            list_df.append(df_sum)
    
    # Fusion des dataframe et ajout d'un ID unique car le TrackID revient à 1 à chaque nouvelle xp
    df_mega = pd.concat(list_df).reset_index(level = None, drop = True)
    df_mega['IDunique'] = range(1, df_mega.shape[0]+1)

    # On enleve toutes les cellules suivies pendant moins de 20*3 minutes (1h)
    df_mega = df_mega[df_mega['TrackNumberofSpots'] >= 20]

    # IMARIS rajoute des trackID non existants, supression ici
    if 'Dist_Totale_µm' in list(df_mega.columns):
        df_mega = df_mega[df_mega['Dist_Totale_µm'].notna()]

    return df_mega

#####################################################################################################################
# 3. FONCTIONS DE SAUVEGARDE DES DONNEES
#####################################################################################################################

def save_fiji_scene(df: pd.DataFrame, scene_dir: str, xp: str, scene: int, summary: bool = False):
    """
    Sauvegarde le dataframe fiji dans le répertoire de sortie.
    
    Arguments:
        df (pd.DataFrame): dataframe à sauvegarder
        scene_dir (str): chemin vers le dossier de la scène contenant les csv
        xp (str): date de l'expérience 
        scene (int): numéro de la scène
        summary (bool, default False): True s'il s'agit du dataframe résumé   
    """
    dir = os.path.join(OUTDATAPATH_FIJI, xp)
    
    # On s'assure que le répertoire existe
    os.makedirs(dir, exist_ok=True)

    # Création du nom du fichier
    filename = xp + "_Scene_" + str(scene)
    if "scatj" in scene_dir.lower().replace(" ", ""):
        filename += "_SCATj"
    elif "scatv" in scene_dir.lower().replace(" ", ""):
        filename += "_SCATv"
    elif "pgatv" in scene_dir.lower().replace(" ", ""):
        filename += "_PGATv"
    elif "pgatj" in scene_dir.lower().replace(" ", ""):
        filename += "_PGATj"
    elif "fapj" in scene_dir.lower().replace(" ", ""):
        filename += "_FAPj"
    elif "fapv" in scene_dir.lower().replace(" ", ""):
        filename += "_FAPv"
    elif "cab" in scene_dir.lower().replace(" ", ""):
        filename += "_" + re.findall('CAB[0-9]+', scene_dir)[0]
    elif "dla" in scene_dir.lower().replace(" ", ""):
        filename += "_" + re.findall('DLA[0-9]+', scene_dir)[0]
    if summary:
        filename += "_Résumé_Par_Cellule"
    else:
        filename += "_Position_Relative"

    # Enregistrement du fichier CSV
    file_path = os.path.join(dir, filename + ".csv")
    df.to_csv(file_path, index = False)

def save_imaris_scene(df: pd.DataFrame, scene_dir: str, xp: str, summary: bool = False):
    """
    Sauvegarde le dataframe IMARIS dans le répertoire de sortie.
    
    Arguments:
        df (pd.DataFrame): dataframe à sauvegarder
        scene_dir (str): chemin vers le dossier de la scène contenant les csv
        xp (str): date de l'expérience 
        summary (bool, default False): True s'il s'agit du dataframe résumé   
    """

    # On s'assure que le répertoire existe
    dir = os.path.join(OUTDATAPATH_IMARIS, xp)
    os.makedirs(dir, exist_ok=True)

    # Création du nom du fichier
    filename = os.path.basename(scene_dir)
    filename = filename.replace("dla", "DLA").replace("cab", "CAB")
    filename = filename.replace("DLA ", "DLA").replace("CAB ", "CAB")
    if summary:
        filename += "_Résumé_Par_Cellule"
    else:
        filename += "_Position_Relative"
    
    # Enregistrement du fichier CSV
    file_path = os.path.join(dir, filename + ".csv")
    df.to_csv(file_path, index = False)

def save_mega_dataframe(df_mega: pd.DataFrame, xp_res_dir: str):
    """Sauvegarde le 'mega' dataframe dans le répertoire xp_res_dir"""
    xp = first_match(DATE_REGEX, xp_res_dir)
    print("DataFrame de l'expérience ", xp," en cours de traitement ...")
    df_mega.to_csv(os.path.join(xp_res_dir, "MEGA_" + xp + ".csv"), index = False)
    print("DONE !")

#####################################################################################################################
# 4. FONCTIONS PRINCIPALES
#####################################################################################################################

def process_fiji_scene(scene_dir: str, xp: str, scene: int): 
    """
    Génère les dataframe de données d'une scène IMARIS et les sauvegarde.
    
    Arguments:
        scene_dir (str): chemin vers le dossier de la scène contenant les csv
        xp (str): date de l'expérience
        scene (int): numéro de la scène
    """

    # Rassemble les differents csv qui forment qu'une scene et applique des vérifications aux données brutes
    df = get_fiji_pos_dataframe(scene_dir, xp, scene)

    # Construit le dataset résumé
    df_sum = get_fiji_summary_dataframe(df)

    # Sauvegarde les fichiers
    save_fiji_scene(df, scene_dir, xp, scene)
    save_fiji_scene(df_sum, scene_dir, xp, scene, summary = True)

def process_imaris_scene(scene_dir: str, xp: str, scene: int): 
    """
    Génère les dataframe de données d'une scène IMARIS et les sauvegarde.
    
    Arguments:
        scene_dir (str): chemin vers le dossier de la scène contenant les csv
        xp (str): date de l'expérience
        scene (int): numéro de la scène
    """

    # Récupère le fichier des positions relatives
    df = get_imaris_pos_dataframe(scene_dir, xp, scene)

    # Construit le dataset résumé
    df_sum = get_imaris_summary_dataframe(scene_dir, xp, scene)

    # Sauvegarde les fichiers
    save_imaris_scene(df, scene_dir, xp)
    save_imaris_scene(df_sum, scene_dir, xp, summary = True)

def process_data(xp_dirs: List[str], unwanted: List[str]):
    """
    Génère les deux csv de statistiques Résumé par Cellules et Positions Relatives
    pour toutes les expériences données et les sauvegarde dans le répertoire de sortie.

    Arguments:
        xp_dirs (List[str]): la liste des répertoires des expériences (se termine par la date)
        unwanted (List[str]): les fichiers ayant ces mots dans leur nom seront ignorés.
    """
    for dir in xp_dirs:
        print("Recherche des fichiers csv...")
        df = create_df_paths(dir, unwanted)
        xp_list = list(df['Experience'].drop_duplicates())

        print("Organisation en cours...")
        for xp in xp_list:
        
            # On récupère la scene et le logiciel de l'expérience
            scenes = list(df[df["Experience"] == xp]["Scene"].drop_duplicates())
            software = list(df[df['Experience'] == xp ]['Software'].head())[0]

            # Parcours des scenes (nombre entier)
            for scene in scenes:

                # On récupère le répertoire qui contient tous le 1er fichier de la scène (et supposément tous les autres)
                scene_path = df.loc[(df["Experience"] == xp) & (df["Scene"] == scene), "Path"].iloc[0]
                scene_dir = os.path.dirname(scene_path)

                print(f"Expérience: {xp},  Scène {scene} ...")
                if software == "FIJI":
                    process_fiji_scene(scene_dir, xp, scene)
                elif software == "IMARIS":
                    process_imaris_scene(scene_dir, xp, scene)
    print("DONE !!")

def process_mega(xp_dirs: List[str]): 
    """
    Génère les 'mega' dataframes de toutes les expériences souhaitées et les sauvegarde
    
    Arguments:
        xp_dirs (List[str]): liste des chemins d'accès aux expériences (données brutes) à traiter
    """

    # Récupère la liste des dossiers de chaque expérience demandée dans le répertoire de sortie
    xp_list = [first_match(DATE_REGEX, p) for p in xp_dirs] # Liste des xp demandées par l'utilisateur
    xp_results_dirs = []
    for sub in [OUTDATAPATH_FIJI, OUTDATAPATH_IMARIS]:
        xp_results_dirs += [os.path.join(sub, xp) for xp in xp_list if os.path.isdir(os.path.join(sub, xp))]

    # Lis le dataframe récap rempli par Sylvie qui contient les information à ajouter
    df_meta = open_metadata()

    # Génère le mega dataframe (résumant toute l'expérience) et le sauvegarde dans le répertoire de sortie
    for xp_res_dir in xp_results_dirs:
        df_mega = get_mega_dataframe(xp_res_dir, df_meta)
        save_mega_dataframe(df_mega, xp_res_dir)


#####################################################################################################################
# 5. FONCTIONS D'INTERACTION AVEC L'UTILISATEUR (Menu)
#####################################################################################################################

def ask_path_to_user(path: str, where: str) -> str:
    """
    Demande à l'utilisateur le chemin d'accès aux données brutes, ou bien 
    le chemin du répertoire de sortie des données produites par ce programme.

    Arguments:
        path (str): Chemin par défaut
        where (str, 'output' ou 'input'): spécifie s'il s'agit du chemin des données brutes
            (input) ou de sortie (output) 

    Renvois:
        (str): le chemin choisi

    """

    print("Voulez-vous garder ce chemin: ", path)
    rep = str(input("y/n ?"))
    if rep == "y":
        return path
    
    #L'utilisateur veut rentrer son propre chemin
    if where == 'input':
        path = str(input("Chemin du dossier qui contient toutes les données (imaris + fiji): "))
        while not os.path.isdir(path):
            print("Chemin NON VALIDE")
            path = str(input("Chemin du dossier qui contient toutes les données (imaris + fiji): "))

    elif where == 'output':
        path = str(input("Chemin du dossier de destination: "))
        print( "Valider ?")
        sur = str(input("y/n ?"))
        while sur != "y":
            path = str(input("Chemin du dossier de destination: "))
            print( "Valider ?")
            sur = str(input("y/n ?"))

    return path

#Fonction qui laisse le choix à l'utilisateur de transformer toutes ou une selection d'xp
def ask_xp_to_user(datapath: str) -> List[str]:
    """
    Demande à l'utilisateur s'il veut traiter toutes les expériences ou une sélection.

    Arguments:
        datapath (str): Le chemin vers les données brutes où sont stockées les expériences
    
    Renvois:
        List[str]: La liste des chemins d'accès aux expériences.
    """

    print("Voulez-vous 1,2,... ou toutes les expériences ?")
    n_xp = int(input("0 pour TOUTES sinon combien? "))

    # Toutes les expériences -> renvoie le chemin total
    if n_xp == 0:
        return [datapath]
    
    # Sinon, demande la date des expériences à traiter
    else :
        xp_dirs = []
        for i in range(n_xp):
            nb = "1ère" if i == 0 else f"{i+1}ème"
            while True:

                # Demande la date
                date = input(f"Date de la {nb} expérience ? Année-Mois-Jour: ").strip()
                xp_dir = os.path.join(datapath, date[:4], date)

                # Vérifie format, validité réelle et existence du dossier
                if re.fullmatch(DATE_REGEX[0], date) is None:
                    print(f"Format de date invalide, écrivez Année-Mois-Jour.")
                    continue
                elif not os.path.isdir(xp_dir):
                    print("Expérience inexistante, Réessayez.")
                    continue
                break

            xp_dirs.append(xp_dir)
        return xp_dirs
    

#####################################################################################################################
# VARIABLES GLOBALES et MAIN
#####################################################################################################################
load_dotenv()

# Chemins des dossiers IN et OUT
IN = os.getenv("RAW_DATA_DIR")
OUT = os.getenv("ORGANIZED_DATA_DIR")
PATHINFOTABLE = os.getenv("METADATA_TABLE_PATH")

# Dossier de sortie par défaut
OUTDATAPATH_FIJI = os.path.join(OUT, "Data_Fiji_ORGANISED")
OUTDATAPATH_IMARIS = os.path.join(OUT, "Data_Imaris_ORGANISED")

# Tous les fichiers contenant un des éléments de la liste unwanted dans son nom seront ignorés
UNWANTED = os.getenv("UNWANTED").split(",")
UNWANTED = [u.strip() for u in UNWANTED]

DATE_REGEX = [r"\d{4}[-.]\d{2}[-.]\d{2}"]
SCENE_REGEX = [r'(?<=IMAGE ).+?(?=(?![0-9]))', r'(?<=Scene[-_]).+?(?=[-_])']
CLASS_REGEX = r"(CAB[0-9]+|DLA[0-9]+|SCAT[jv]|PGAT[jv]|FAP[jv])"
AGE_REGEX = r'(\d+)\s*(sem|mois|an)'

####################################################################################################################
# BOUCLE PRINCIPALE
####################################################################################################################
if __name__ == "__main__":

    # Affichage de la version
    print("\nScript OrganizeAllData: Version du 10/12/2025\n")

    # On verifie l'existance du chemin donné par l'utilisateur
    print("CHEMIN D'ORIGINE")
    DATAPATH = ask_path_to_user(IN, where='input')
    print("CHEMIN DE DESTINATION")
    OUTDATAPATH = ask_path_to_user(OUT, where='output')

    # On différencie les dossier destination par leur logiciel d'origine
    OUTDATAPATH_FIJI = OUTDATAPATH + "\\Data_Fiji_ORGANISED"
    OUTDATAPATH_IMARIS = OUTDATAPATH + "\\Data_Imaris_ORGANISED"

    # L'utilisateur choisit si il organise toutes les XP ou un nombre précis
    xp_dirs= ask_xp_to_user(DATAPATH)

    # xp_dirs peut ne contenir qu'un element 
    # Créée les dataframes Résumé par Cellule et Position Relatives dans le répertoire de sortie
    process_data(xp_dirs, UNWANTED)

    # Créée les 'mega' dataframes dans le répertoire de sortie
    print("CALCUL DES MEGA DATAFRAMES")
    process_mega(xp_dirs)

    print("Terminé !")