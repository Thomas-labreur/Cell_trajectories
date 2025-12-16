#Librairie de manipulation d'expression régulière
import re
# Librairie pour la manipulation de DataFrames
import pandas as pd
# Librairie pour la manipulation des répertoires
import os
from os.path import isfile, join
from fnmatch import fnmatch
#Librairie pour sqrt, mean, and std
import numpy as np
#####################################################################################################################


#####################################################################################################################
#FONCTIONS ANNEXES
#####################################################################################################################

# Récupère le trackID max contenu dans un csv à partir de son nom
# Renvoi le derneier nombre dans le nom du fichier,
# Appellé si "à" est dans le nom
def how_big_trackID_should_be(txt):
    return int(re.findall(r'\d+', txt)[-1])

# renvoie un dataframe contenant la moyenne de la variable 'colonne' classée par la variable 'valeur' du dataframe df
def moy_sachant(df, colonne, valeur, nom, dropAll = True):
    #df = cleaning(df)
    # Création d'un dataframe ne comportant que le trackID de chaque cellule
    dfc = df.copy()
    if dropAll:
        dfc = drop_all_but(dfc, [valeur])
    dfc = dfc.drop_duplicates(subset = valeur)
    #On vérifie que les valeurs sont transformables en float si non:
    if "," in str(df[colonne][0]):
        df[colonne] = [float(l.replace(",",".")) for l in df[colonne]]
    dfc[nom] = [df[df[valeur]==x][colonne].mean() for x in dfc[valeur]]
    return dfc
#IMARIS
# Fonction qui supprime tous les espaces dans les noms des colonnes et suprimme la colonne rajoutée systématiquement à la fin
def cleaning(df, unnamed = True):
    df.columns = [x.replace(' ', '') for x in df.columns]
    # liste de boolean qui verifie si une colonne s'appelle Unnamed...
    b = ["Unnamed" in c for c in df.columns]
    if True in b:
        c_to_delete = "Unnamed:"+ str(len(df.columns) - 1)
        df = df.drop(columns = c_to_delete)
    return df

# Drop toutes les colonnes du dataframe df sauf celles indiquées en liste l_colonnes
def drop_all_but(df, l_colonnes):
    n_colonnes = list(df.columns.values)
    for c in l_colonnes:
        n_colonnes.remove(c)
    df = df.drop(columns = n_colonnes)
    return df

# Fonction pour retrouver le nom du premier fichier qui contient 'name' dans son nom
def find(name, path):
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if name in file_name:
                return os.path.join(root, file_name)

#####################################################################################################################
#FONCTIONS DE MANIPULATIONS DE DATAFRAME ET FEATURES
# Renvoi la fusion sur la clé key des dataframes de la liste l_df
def merge_l_df(l_df, key):
    # initialise le premier dataframe comportant seulement les clés sur lesquelles la fusion se fait
    result = l_df[0].copy()
    result = drop_all_but(result, [key])
    # Fusionne les dataframes
    for df in l_df:
        result = result.merge(df, on = key, how = 'outer')
    return result

#FIJI
# Fonction de nettoyage pour les données sortant du logiciel FIJI
# Renomme les colonnes et réajuste la valeur des TrackIDs
def cleaning_fiji(df, name):
    df.columns = ['TrackID', 'Time', 'PositionX', 'PositionY', 'Distance_µm', 'Speed_µm/min', 'PixelValue']
    # Si il faut réajuster les valeurs de TrackID
    if "à" in name:
        diff_trackID = how_big_trackID_should_be(name) - df['TrackID'].max()
        df['TrackID'] += diff_trackID
    # Remise à zero des premiers temps de chaque cellule (-1 à 0)
    df['Distance_µm'] = [remettreAzero(d,time) for d, time in zip(df['Distance_µm'], df['Time'])]
    df['Speed_µm/min'] = [remettreAzero(d,time) for d, time in zip(df['Speed_µm/min'], df['Time'])]
    return df

# IMARIS
# Renvoie un dataFrame avec chaque cellule et son acceleration moyenne en µm/m2 et drop celle en µm/s2
def acceleration_moyenne(df_a):
    # Cleaning
    df_a = cleaning(df_a)
    # Moyenne de l'acceleration sachant trackID
    df_a = moy_sachant(df_a,'Acceleration','TrackID','Acc_Moyenne_µm/s2')
    df_a['Acc_Moyenne_µm/m2'] = [3600 * a for a in df_a['Acc_Moyenne_µm/s2']]
    # Supression des colonnes inutiles
    df_a = df_a.drop(columns = 'Acc_Moyenne_µm/s2')
    # df_a = drop_all_but(df_a, ['Acc_Moyenne_µm/s2', 'Acc_Moyenne_µm/m2', 'TrackID'])
    return df_a


# IMARIS
# Renvoie un dataFrame avec chaque cellule et sa vitesse moyenne en µm/m et drop celle en µm/s
def vitesse_moyenne(df_s):
    df_s = cleaning(df_s)
    # Moyenne de la vitesse sachant trackID
    #Certain csv n'ont pas les même nom de colonnes
    if "Speed(µm/s)" in df_s.columns:
        df_s = moy_sachant(df_s,'Speed(µm/s)','TrackID','Speed_Moyenne_µm/s')
    else:
        df_s = moy_sachant(df_s,'Speed','TrackID','Speed_Moyenne_µm/s')
    df_s['Speed_Moyenne_µm/min'] = [60 * a for a in df_s['Speed_Moyenne_µm/s']]
    df_s = df_s.drop(columns = 'Speed_Moyenne_µm/s')
    return df_s

#ALL
def sec_to_min(df, colonne):
    df[colonne] = df[colonne] * 60
    return df

def distance2(df_d, nom):
    # Cleaning
    df_d = moy_sachant(df_d, 'Displacement^2', 'TrackID', nom)
    return df_d

def distance2_60min(df_d2_60min):
    # On garde seulement les 20 premiers clichés 20 x 3 minutes = 1 heure
    df_d2_60min = df_d2_60min[df_d2_60min['Time']<21]
    df_d2_60min = distance2(df_d2_60min,'Displacement^2_60min_Moyenne_µm2')
    return df_d2_60min

def distance_parcourue(df_ddl):
    df_ddl = cleaning(df_ddl)
    df_ddlm = df_ddl.copy()
    df_ddlm = drop_all_but(df_ddlm, ['TrackID'])
    df_ddlm = df_ddlm.drop_duplicates(subset = 'TrackID')

    df_ddlm['Dist_Totale_µm'] = [df_ddl[df_ddl['TrackID']==x]['DisplacementDeltaLength'].sum() for x in df_ddlm['TrackID']]
    return df_ddlm

def moyenne_distance_origin(df):
    df = cleaning(df)
    df = moy_sachant(df, 'DistancefromOrigin', 'TrackID', 'DistancefromOrigin_Moyenne_µm')
    return df

# INUTILISE
def generation(df):
    df = moy_sachant(df, 'Generation','TrackID', 'Generation')
    return df
# INUTILISE
def velocity_angle(df):
    df = cleaning(df)
    df = moy_sachant(df, "VelocityAngleX", "TrackID", "VelocityAngleX_Moyenne", dropAll = False)
    df = moy_sachant(df, "VelocityAngleY", "TrackID", "VelocityAngleY_Moyenne", dropAll = False)
    df = drop_all_but(df, ['VelocityAngleX_Moyenne', 'VelocityAngleY_Moyenne', 'TrackID'])
    return df


# Renvoie le dataframe df une fois nettoyé, 
# on enlève les espaces des noms de colonnes et suprimme la derniere colonne ajoutée systématiquement 'unamed:x'
# Renome la colonne 'ID' en 'TrackID' pour pouvoir faire la jointure ultérieurement
# Suprimme toutes les colonnes sauf 'TrackID' et le paramètre d'entrée colonne
def cleaning2(df, colonne, leg = '', add_leg = False):
    df = cleaning(df)
    df = df.rename(columns = {'ID':'TrackID'})
    df = drop_all_but(df, ['TrackID', colonne])
    if add_leg:
        df = df.rename(columns = {colonne:colonne+"_"+leg})
    return df


# Rajoute à df_p 2 colonnes indiquant la position relative à celle de départ à tout instant pour toutes les cellules
def position_relative(df_p, unnamed = True, changeUnit = False):
    # Cleaning
    df_p = cleaning(df_p, unnamed)
    df_p = df_p.dropna()
    #df_p = df_p.drop(columns=['Category', 'Unit', 'PositionZ', 'Collection'])
    # DataFrame intermédiaire
    df_init = df_p.copy()
    df_init = df_init.sort_values(by = ['TrackID', 'Time'])
    df_init = df_init.drop_duplicates(subset = 'TrackID', keep = 'first')
    # Création des colonnes de Position Relative
    # Logiciel Fiji donne la position en pixel: 1 pixel = 0.454micrometre
    if changeUnit :
        df_p['PositionX_Relative'] = [float(x - df_init[df_init['TrackID']==tID]['PositionX'])*0.454 for x, tID in zip(df_p.PositionX, df_p.TrackID)]
        df_p['PositionY_Relative'] = [float(y - df_init[df_init['TrackID']==tID]['PositionY'])*0.454 for y, tID in zip(df_p.PositionY, df_p.TrackID)]
    else:
        df_p['PositionX_Relative'] = [float(x - df_init[df_init['TrackID']==tID]['PositionX']) for x, tID in zip(df_p.PositionX, df_p.TrackID)]
        df_p['PositionY_Relative'] = [float(y - df_init[df_init['TrackID']==tID]['PositionY']) for y, tID in zip(df_p.PositionY, df_p.TrackID)]
        # Supression des colonnes inutiles
        df_p = drop_all_but(df_p,['PositionX_Relative', 'PositionY_Relative', 'TrackID', 'Time'])
    return df_p


# Renvoie la proportion des cellules qui se sont divisées dans le df
def proportion_division(df):
    df = cleaning2(df, 'TrackNumberOfGenerations')
    p = df[df['TrackNumberOfGenerations']>1]['TrackNumberOfGenerations'].count()/df['TrackNumberOfGenerations'].size * 100
    return p


# Renvoie le nombre de divisions apparues dans le df
def number_of_division(df):
    df = cleaning2(df, 'TrackNumberOfGenerations')
    p = df[df['TrackNumberOfGenerations']>1]['TrackNumberOfGenerations'].count()
    return p

# INUTILISE
# Renvoie un dataFrame avec chaque cellule et sa vitesse moyenne en µm/m2
def vitesse_moyenne_fiji(df_s):
    # Moyenne de la vitesse sachant trackID
    df_s = moy_sachant(df_s,'Speed_µm/min','TrackID','Speed_Moyenne_µm/min', dropAll = True)
    df_s['Speed_Moyenne_µm/min']
    return df_s


#####################################################################################################################
#Fonctions de sauvegarde des CSV
#####################################################################################################################

#Sauvegarde le df organisé d'une scene en format csv en créant un nouveau nom de fichier
def save_fiji_scene(df, path, xp, scene, resume = False):
    chemin = OUTDATAPATH_FIJI + xp + "/"
    
    # On s'assure que le répertoire existe
    os.makedirs(chemin, exist_ok=True)
    # Création du nom du fichier
    filename = "Results_from_XP_" + xp + "_Scene_" + str(scene)
    if True in [ x in path for x in ["scatj", "scat j", "SCATj", "SCATJ"]]:
        filename += "_SCATj"
    elif True in [ x in path for x in ["scatv", "scat v", "SCATv", "SCATV"]]:
        filename += "_SCATv"
    elif True in [ x in path for x in ["pgatv", "pgat v","PGATv","PGATV"]]:
        filename += "_PGATv"
    elif True in [ x in path for x in ["pgatj","pgat j","PGATj","PGATJ"]]:
        filename += "_PGATj"
    elif True in [ x in path for x in ["fapj","fap j", "FAPj"]]:
        filename += "_FAPj"
    elif True in [ x in path for x in ["fapv","fap v","FAPv"]]:
        filename += "_FAPv"
    elif True in [ x in path for x in ["CAB", "cab"]]:
        filename += "_" + re.findall('CAB[0-9]+', path)[0]
    elif True in [ x in path for x in ["DLA", "dla"]]:
        filename += "_" + re.findall('DLA[0-9]+', path)[0]
    
    if resume:
        filename += "_Résumé_Par_Cellule"
    else:
        filename += "_Position_Relative"
        
    filename +=  ".csv"
    # Enregistrement du fichier CSV
    df.to_csv(chemin + filename, index = False)
    #print('CSV enregistré à:', chemin+filename)
    return 0


#IMARIS
#good updated
def ecrire_csv(df, path, xp, name):
    xp += "/"
    f_name = os.path.basename(path) + name
    if is_cabdla(f_name):
        f_name = maj_dla_cab(f_name)
        f_name = del_space(f_name)
    os.makedirs(OUTDATAPATH_IMARIS + xp, exist_ok=True)
    df.to_csv(OUTDATAPATH_IMARIS+xp+f_name, index = False)
    #print("CSV enregistré à: ", OUTDATAPATH_IMARIS+xp+f_name)
    return 0

def is_cabdla(f_name):
    possib =['dla', "DLA", "cab", "CAB"]
    if True in [x in f_name for x in possib]:
        return True
    else:
        return False
#S'assure que DLA et CAB soient en majuscules sans espace avant le nombre
def maj_dla_cab(f_name):
    if 'dla' in f_name:
        return f_name.replace('dla', 'DLA')
    elif 'cab' in f_name:
        return f_name.replace('cab','CAB')
    else:
        return f_name
# Efface l'espace qui pourrait etre entre [CAD, DLA] et son numero
def del_space(f_name):
    typ = 'CAB'
    if 'DLA' in f_name:
        typ = 'DLA'
    i = f_name.index(typ)
    if ' ' == f_name[i+3]:
        f_name = f_name[:i+3] + f_name[i+4:]
    return f_name

#####################################################################################################################
#FIJI
#good
# INPUT: liste des chemin des fichiers de la scène
# Renvoie un df organisé à partir du chemin dossier d'origine de la scène comportant plusieurs csv
def organise_fiji_scene(paths, xp, scene):
    # On ouvre chaque fichier
    l_df = [pd.read_csv(p, encoding = 'latin') for p in paths]
    # On vérifie la dimension si non on réouvre avec le séparateur européen
    if len(l_df[0].columns)!=7:
        l_df = [pd.read_csv(p, encoding = 'latin', sep = ";") for p in paths]
    # renommage des colonnes et des TrackID erronés, remise à zero des premiers temps de chaque cellule
    l_df = [cleaning_fiji(df, p) for df, p in zip(l_df, paths)]
    #VERIFICATION DOUBLOUS ou SAUT DE TEMPS
    affiche_verif(l_df, scene, xp)
    # On forme qu'un seul fichier pour une scène et on renomme les indexes
    df = pd.concat(l_df)
    df = df.sort_values(by = ['TrackID', 'Time'])
    df.reset_index(level = None, drop = True, inplace=True)
    #print('calcul de la position relative...')
    # On y ajoute la position relative
    df = position_relative(df, unnamed = False, changeUnit = True)
    #print("position relative calculée")
    return df

#FIJI
# Gère le cas des cellules qui n'ont pas bougé t = 0
def persistance(o,t):
    if t == 0:
        return "NaN"
    else:
        return o/t
#FIJI
# change le -1 en 0 pour toutes les cellules au temps t=1
def remettreAzero(d, time):
    if time == 1 and d == -1:
        return 0
    else:
        return d

#Fix division by zero rare case, used to get the TrackSpeedVariation , sometimes TrackSpeedMean=0
def division_zero_proof(n,d):
    if d==0:
        return 0
    else:
        return n/d

#FIJI
#GOOD
# INPUT : dataframe d'une scène déjà organisé 
# Renvoie un df qui résume la scène par cellule
def resume_fiji_scene(df):
    # Construction d'un df, on garde 1 fois chaque TrackID 
    rdf = df.copy()
    rdf = drop_all_but(rdf, ['TrackID'])
    rdf = rdf.drop_duplicates(subset = 'TrackID')
    # df intermédiaire qui contient la derniere position relative de chaque cellule: utilisé pour Dist_Origine et trackNumberOfSpots
    df_init = df.copy()
    df_init = df_init.sort_values(by = ['TrackID', 'Time'])
    df_init = df_init.drop_duplicates(subset = 'TrackID', keep = 'last')
    # Features suplémentaires sur vitesse
    rdf['TrackSpeedMean_µm/min'] = [np.mean(df[df['TrackID']==x]['Speed_µm/min']) for x in rdf['TrackID']]
    rdf['TrackSpeedMin_µm/min'] = [min(df[df['TrackID']==x]['Speed_µm/min']) for x in rdf['TrackID']]
    rdf['TrackSpeedMax_µm/min'] = [max(df[df['TrackID']==x]['Speed_µm/min']) for x in rdf['TrackID']]
    rdf['TrackSpeedStdDev_µm/min'] = [np.std(df[df['TrackID']==x]['Speed_µm/min']) for x in rdf['TrackID']]
    rdf['TrackSpeedVariation'] = [division_zero_proof(sd,moy) for sd, moy in zip(rdf['TrackSpeedStdDev_µm/min'], rdf['TrackSpeedMean_µm/min'])]
    # Feature Distance Totale Parcourue
    rdf['Dist_Totale_µm'] = [df[df['TrackID']==x]['Distance_µm'].sum() for x in rdf['TrackID']]
    # Feature moyenne des distances instantanés
    rdf['Dist_instant_Moyenne_µm'] = [np.mean(df[df['TrackID']==x]['Distance_µm']) for x in rdf['TrackID']]
    # Feature Distance vol d'oiseau
    rdf['Dist_Origine_µm'] = [np.sqrt(x**2+y**2) for x, y in zip(df_init['PositionX_Relative'], df_init['PositionY_Relative'])]
    # Feature Persistance
    rdf['Persistance'] = [persistance(o,t) for o, t in zip(rdf['Dist_Origine_µm'], rdf['Dist_Totale_µm'])]
    rdf['TrackNumberofSpots'] = [int(df_init[df_init['TrackID']==i]['Time']) for i in rdf['TrackID']]
    return rdf

#IMARIS
# Fonction qui ouvre les anciens csv pour les réorganiser
# Renvoie 1 dataFrame d'une scène 
def by_TrackID(path, Scene, xpname):
    # Chargement des données
    df_a = get_dirty_csv("Acceleration", path)
    df_ddl = get_dirty_csv("Displacement_Delta_Length", path)
    df_s = get_dirty_csv("Speed.", path)
    df_d2 = get_dirty_csv("Displacement^2", path)
    #df_dfo = get_dirty_csv("Distance_from_Origin", path)
    #df_g = get_dirty_csv("Generation", Scene, path)
    #df_nog = get_dirty_csv('Track_Number_Of_Generations', path)
    df_nob = get_dirty_csv('Track_Number_of_Branches', path)
    #df_va = get_dirty_csv("Velocity_Angle", path)
    df_ts = get_dirty_csv("Track_Straightness", path)
    df_sv = get_dirty_csv('Track_Speed_Variation', path)
    df_ssd = get_dirty_csv('Track_Speed_StdDev', path)
    df_smin = get_dirty_csv('Track_Speed_Min', path)
    df_smax = get_dirty_csv('Track_Speed_Max', path)
    df_smean = get_dirty_csv('Track_Speed_Mean', path)
    df_nos = get_dirty_csv('Track_Number_of_Spots', path)
    df_tdl = get_dirty_csv('Track_Displacement_Length', path)
    #Track duration inutile peut etre
    df_td = get_dirty_csv('Track_Duration', path)
    df_ar = get_dirty_csv('Ar1', path)
    #VERIFICATION DES DONNEES BRUTES
    l_ddf = [df_a, df_ddl, df_s, df_d2]
    affiche_verif(l_ddf, Scene, xpname)
    # Modification des dataframes
    df_a = acceleration_moyenne(df_a)
    # Calcul de la moyenne des distances instantanés
    didf = moy_sachant(df_ddl, 'Displacement Delta Length', 'TrackID', 'Dist_instant_Moyenne_µm', True)
    df_ddl = distance_parcourue(df_ddl)
    df_s = vitesse_moyenne(df_s)
    df_d2_60min = df_d2.copy()
    df_d2_60min = distance2_60min(df_d2_60min)
    df_d2 = distance2(df_d2,'Displacement^2_Moyenne_µm2')
    #df_dfo = moyenne_distance_origin(df_dfo)
    # VARIABLE Génération remplacé par number of gen et number of branches
    #df_g = generation(df_g)
    df_ar = cleaning2(df_ar, 'TrackAr1Mean')
    #df_nog = cleaning2(df_nog, 'TrackNumberOfGenerations')
    df_nob = cleaning2(df_nob, 'TrackNumberofBranches')
    #df_va = velocity_angle(df_va)
    df_ts = cleaning2(df_ts,'TrackStraightness')
    # Change de nom de colonne pour matcher les données de FIJI 
    df_ts = df_ts.rename(columns = {'TrackStraightness':'Persistance'})
    df_sv = cleaning2(df_sv, 'TrackSpeedVariation')
    df_ssd = cleaning2(df_ssd, 'TrackSpeedStdDev', leg = "µm/min", add_leg = True)
    df_smin = cleaning2(df_smin, 'TrackSpeedMin', leg = "µm/min", add_leg = True)
    df_smax = cleaning2(df_smax, 'TrackSpeedMax', leg = "µm/min", add_leg = True)
    df_smean = cleaning2(df_smean, 'TrackSpeedMean', leg = "µm/min", add_leg = True)
    # Convertir les données en µm/min
    df_ssd = sec_to_min(df_ssd, 'TrackSpeedStdDev_µm/min')
    df_smin = sec_to_min(df_smin, 'TrackSpeedMin_µm/min')
    df_smax = sec_to_min(df_smax, 'TrackSpeedMax_µm/min')
    df_smean = sec_to_min(df_smean, 'TrackSpeedMean_µm/min')
    
    df_nos = cleaning2(df_nos, 'TrackNumberofSpots')
    df_tdl = cleaning2(df_tdl, 'TrackDisplacementLength')
    df_tdl = df_tdl.rename(columns = {'TrackDisplacementLength':'Dist_Origine_µm'})
    #Track duration inutile peut etre
    df_td = cleaning2(df_td, 'TrackDuration')
    # Les rassembler en 1 dataframe , df_dfo, df_va, df_nog enlevés
    l_df = [df_a, df_s, df_smean, df_smin, df_smax, df_ssd, df_sv, df_ddl, didf, df_tdl, df_ts, df_nob, df_ar, df_td, df_nos, df_d2, df_d2_60min]
    #print(len(l_df),' variables pour chaque cellule')
    result = merge_l_df(l_df,'TrackID')
    return result
#####################################################################################################################
#FIJI
#GOOD
#prend le chemin du dossier de la scène, la date de l'xp et le numéro de la scène
def organise_resume_save_scene_fiji(path, xp, scene):
    scene_files = get_fiji_scene_files_list(path)
    #Rassemble les differents csv qui forment qu'une scene et applique des vérifications aux données brutes
    df = organise_fiji_scene(scene_files, xp, scene)
    rdf = resume_fiji_scene(df)
    save_fiji_scene(df, path, xp, scene)
    save_fiji_scene(rdf, path, xp, scene, resume = True)
    return 0

#IMARIS
#GOOD
# input path :chemin du dossier de la scène 
def organise_save_scene_imaris(path, xp, scene):
    # CSV par cellule
    r = by_TrackID(path, scene, xp)
    ecrire_csv(r, path, xp, "_Par_Cellule.csv")
    # CSV position relative
    df = get_dirty_csv("Position", path)
    r = position_relative(df , unnamed = True, changeUnit = False)
    ecrire_csv(r, path, xp, "_Position_Relative.csv")
    return 0


#####################################################################################################################
# Fonctions pour automatiser la recherche de fichiers
#####################################################################################################################
#ALL?
def extension(path):
    filename, file_extension = os.path.splitext(path)
    return file_extension
# IMARIS
# Renvoi le dataframe du paramètre para de l'expérience scene contenu dans le dossier wpath
# Bug si find() ne trouve pas (pd.read_csv(None))
def get_dirty_csv(para, spath):
    f_path = find(para, spath)
    #tous les csv n'ont pas la meme structure
    #, skiprows = 3
    df = pd.read_csv(find(para, spath), on_bad_lines = 'skip', encoding = "latin", sep = ";")
    # Si le csv a une en-tête bizarre
    if len(df.columns) < 2:
        df = pd.read_csv(find(para, spath), on_bad_lines = 'skip', skiprows = 3, encoding = "latin", engine = 'python')
    return df
# FIJI
# Renvoie la liste des chemins de tous les fichiers csv d'une scène
def get_file_path(path):
    l = [path +"/"+ f for f in os.listdir(path) if (isfile(join(path, f)) and (".csv" == extension(join(path,f))))]
    return l
#FIJI
#good
# renvoie la liste des chemins des fichiers d'une scène fiji à partir du dossier de la scène
def get_fiji_scene_files_list(s_path):
    return get_file_path(s_path)

#ALL
#Recupère tous les chemins des fichiers csv dans le dossier ayant pour chemin root
def get_all_dirty_paths(root):
    pattern = "*.csv"
    l = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files if fnmatch(name, pattern)]
    print("CSV trouvés, Tri en cours...")
    l = [ p.replace("\\","/") for p in l ]
    return l
def get_all_absFP(dossier):
    l=[os.path.join(dirpath, f) for dirpath,_,filenames in os.walk(dossier) for f in filenames if '.csv' in f]
    print("CSV trouvés, Tri en cours...")
    l = [ p.replace("\\","/") for p in l ]
    return l
#ALL
# A partir du chemin d'un fichier renvoie la date de l'xp
# un point ou un tiret peut separer les dates d'xp
def path_to_xp(path):
    if re.findall("[0-9][0-9][0-9][0-9][-.][0-9][0-9][-.][0-9][0-9]", path) == []:
        return "NaN"
    else:
        xp = re.findall("[0-9][0-9][0-9][0-9][-.][0-9][0-9][-.][0-9][0-9]", path)[0]
        return xp
#ALL
# A partir du chemin d'un fichier renvoie le numéro de la scène
def path_to_scene(path, logiciel):
    #print(path, logiciel)
    if (re.findall(r'(?<=IMAGE ).+?(?=(?![0-9]))', path) == []) and (re.findall(r'(?<=Scene[-_]).+?(?=[-_])', path) == []) :
        return "NaN"
    if (re.findall(r'(?<=IMAGE ).+?(?= )', path) != []):
        scene = int(re.findall(r'(?<=IMAGE ).+?(?=(?![0-9]))', path)[0])
    else:
        scene = int(re.findall(r'(?<=Scene[-_]).+?(?=[-_])', path)[0])
    return scene
#ALL
# A partir du chemin d'un fichier renvoie le logiciel utilisé
def path_to_logiciel(path):
    if "Statistics" in path:
        return "IMARIS"
    else:
        return "FIJI"

#ALL
# Fonction qui trie tous les fichiers csv 
# Renvoi un dataframe qui contient tous les chemins de fichiers csv pour ensuite les trier etc
def df_paths(liste_chemins, indesirable):
    c = ["Experience","Scene","Logiciel","Chemin"]
    df = pd.DataFrame(columns = c)
    # Recherche des chemins indesirables
    l = [p for p in liste_chemins for i in indesirable if i in p]
    # On supprime les doublons parmis les chemins à supprimer
    l = list(dict.fromkeys(l))
    # Supprimmer les indésirables
    for i in l :
        liste_chemins.remove(i)
    df["Chemin"] = [p for p in liste_chemins]
    # df["Chemin"] = [p for i in indesirable for p in liste_chemins if i not in p]
    
    df["Logiciel"] = [ path_to_logiciel(p) for p in df["Chemin"]]
    df["Experience"] = [ path_to_xp(p) for p in df["Chemin"]]
    df["Scene"] = [ path_to_scene(p, l) for p, l in zip(df["Chemin"], df["Logiciel"])]
    return df
#ALL
# Renvoi la liste de toutes les xp
# A partir du df renvoyé par df_paths 
def get_liste_xp(df):
    df2 = df.copy()
    df2 = df2.drop_duplicates(subset = "Experience")
    return list(df2['Experience'])

#ALL
# Renvoi la liste des numéros de scènes d'une xp
def get_scenes_of(df, xp):
    df2 = df[df["Experience"] == xp].copy()
    df2 = df2.drop_duplicates(subset = "Scene")
    return list(df2['Scene'])

#ALL

def select_logiciel_from_xp(df, xp):
    return list(df[df['Experience'] == xp ]['Logiciel'].head())[0]

#ALL
# Renvoi le chemin du dossier d'une scène d'une xp à partir du dataframe de tous les chemins
# Cherche le chemin d'un des fichiers de la scène et prend son dossier parent
def select_scene_path(df, scene, xp):
    chem = df.loc[(df["Experience"] == xp ) & (df['Scene'] == scene), ['Chemin']]
    chem = os.path.dirname(list(chem['Chemin'].head())[0])
    return chem

#####################################################################################################################

#####################################################################################################################
#ALL
#input df : retour de la fonction df_paths()
# Parcourt toutes les expériences, données en input,
# les nettoie et les enregistre
def CLEAN_ALL(df, liste_xp):
    for xp in liste_xp:
        liste_scenes = get_scenes_of(df, xp)
        logiciel = select_logiciel_from_xp(df, xp)
        #ici on parcour une liste de nombre
        for scene in liste_scenes:
            path = select_scene_path(df, scene, xp)
            print("XP ", xp, ", scène ", scene," ...")
            if logiciel == "FIJI":
                organise_resume_save_scene_fiji(path, xp, scene)
            elif logiciel == "IMARIS":
                organise_save_scene_imaris(path, xp, scene)
    return 0


def DO(path, DontTake):
    print("Recherche des fichiers csv...")
    #RAJOUTER
    df = df_paths(get_all_absFP(path), DontTake)
    liste_xp = get_liste_xp(df)
    print("Organisation en cours...")
    #liste_xp = exclure2(liste_xp)
    #liste_xp.remove('NaN')
    CLEAN_ALL(df, liste_xp)
    print("DONE !!")
    return 0

#####################################################################################################################
# MEGA DATAFRAME et ses fonctions
# DataFrame qui réuni toutes les cellules d'une expérience mais pas à tout instant
# Fonction à appeler : save_all_mega_csv(chemin du dossier des xp)
#####################################################################################################################

#renvoie la liste des dossiers directement dans le dossier path
def get_folder_list(path):
    if os.path.isdir(path):
        l = next(os.walk(path))[1]
        return l
    else:
        return []
#NEW
# Renvoie la liste qui contient tous les chemins des fichiers qui contiennent name dans leur nom, du dossier ayant pour chemin path
def find_all_f_with(name, path):
    l = [os.path.join(root, file_name) for root, dirs, files in os.walk(path) for file_name in files if name in file_name]
    return l
#NEW
# Renvoie le nom du fichier à partir de son chemin complet sans son extension
# path doit etre un chemin de fichier
def path_to_filename(path):
    # On récupère seulement la fin du chemin 
    f_name = os.path.basename(path)
    # On enlève .csv
    f_name = f_name[:f_name.index(".csv")]
    return f_name
#NEW
# Renvoie la classe des cellules contenues dans ce fichier
# SCATv/SCATj ou CAB/DLA ou PGATv/ PGATj
def class_from(filename):
    if True in [ x in filename for x in ["CAB", "cab"]]:
        return re.findall('CAB[0-9]+', filename)[0]
    elif True in [ x in filename for x in ["DLA", "dla"]]:
        return re.findall('DLA[0-9]+', filename)[0]
    elif True in [ x in filename for x in ["scatj", "scat j", "SCATj"]]:
        return "SCATj"
    elif True in [ x in filename for x in ["scatv", "scat v", "SCATv"]]:
        return "SCATv"
    elif True in [ x in filename for x in ["pgatj", "pgat j", "PGATv"]]:
        return "PGATj"
    elif True in [ x in filename for x in ["pgatv", "pgat v", "PGATv"]]:
        return "PGATv"
    elif True in [ x in filename for x in ["fapj","fap j", "FAPj"]]:
        return "FAPj"
    elif True in [ x in filename for x in ["fapv","fap v","FAPv"]]:
        return "FAPv"
    else:
        return "NaN"
    
#NEW
# Ajoute la colonne classe au df par rapport au nom de fichier
def ajouter_classe(df, filename):
    classe = class_from(filename)
    df['CLASSE'] = classe
    return df


#NEW
#Ajoute la colonne qui spécifie de quelle scène provient la cellule
def ajouter_scene(df, filename):
    scene = int(re.findall(r'(?<=\Scene[-_]).+?(?=[-_])',filename)[0])
    df['SCENE'] = scene
    return df
#ALL
# Ajoute la colonne de la date de l'xp, pour s'y retrouver qd on assemble plusieurs xp
def ajouter_date(df, filename):
    date = re.findall("[0-9][0-9][0-9][0-9][-.][0-9][0-9][-.][0-9][0-9]", filename)[0]
    df['DATE'] = date
    return df

#AGE
# Ajoute la colonne de l'age en semaine
# Prend le df d'une xcène, récup la date et le numero de la scène qui vient d'y etre ajouté
def ajouter_age(df, df_age):
    xp_date = df['DATE'][0]
    scene = df['SCENE'][0]
    df['AGE'] = age(xp_date, scene, df_age)
    return df
#AGE
# Extrait le nombre associé à l'unité donné en paramètre
# renvoi 3 pour txt='3 mois et 2 semaines' et unit='mois'
def extract(txt, unit):
    #Si l'unité n'est pas présente alors 0
    if re.findall('(\d+)(?=\s*' + unit + ')', txt) ==[]:
        nb = '0'
    else:
        nb = re.findall('(\d+)(?=\s*' + unit + ')', txt)[0]
    return int(nb)
#AGE
# Prend un age en chaine de caractère mélangeant sem, mois et année pour le convertir en nombre de semaines uniquement
def en_semaine(txt):
    sem = 0
    sem += extract(txt, 'sem')
    sem += 4.34524 * extract(txt, 'mois')
    sem += 52.1429 * extract(txt, 'an')
    return round(sem)
#AGE
# Renvoie un age en fonction de la scène et de l'xp
# df est le tableau recapitulatif fait par sylvie
def age(xp, sc, df):
    a = df.loc[(df["Date"] == xp ) & (df['Scene'] == sc), ['Age']]
    return en_semaine(str(a))
# utilisé pour nettoyer le fichier de sylvie 
# renvoie seulement le numéro de la scène à partir de 'Image x'
def get_nb_image(txt):
    i = re.findall('(\d+)$', txt)[0]
    return i
# prend le fichier recap de sylvie et en créer un autre près à etre utilisé
def prepare_age():
    #recuperer le fichier remplie par Sylvie
    df = pd.read_excel(TableauTTPath)
    #change de nom de colonne
    df = df.rename(columns={'Scene/Image':'Scene'})
    #suppression des eventuels saut de ligne
    df = df.drop(df.index[df['Scene'].isna()])
    # on garde seulement le numéro de la scène    
    df['Scene'] = [get_nb_image(i) for i in df['Scene']]
    # enregistre où est le script
    df.to_csv('recap_age_pour_script.csv', index = False)

#IMARIS rajoute des trackID non existants, supression ici
def clean_mega(df):
    # On enleve toutes les cellules suivies pendant moins de 20*3 minutes
    df = df.drop(df.index[df['TrackNumberofSpots']<20])
    # On vérifie qu'on est sur IMARIS (faux)
    if 'Dist_Totale_µm' in list(df.columns):
        df = df.drop(df.index[df['Dist_Totale_µm'].isna()])
    return df
def verif_age(df):
    age_nul = np.array(df[df['AGE']==0][['DATE', 'SCENE']].drop_duplicates())
    if len(age_nul)>0:
        print("=================================================================")
        print("            /!\\ ATTENTION PROBLEME D'AGE' /!\\")
        print("Age non spécifié dans le fichier recap_age_Theotime.csv")
        for xp, sc in age_nul:
            print("Age NUL pour l'xp ", xp,", scene ", sc )
#NEW
#Création d'un mega DF comprenant toutes les cellules d'une expérience ayant pour chemin de dossier path
def get_mega_df(path):
    recap_age = pd.read_csv('recap_age_pour_script.csv')
    # Recupere la liste des chemins des scenes de l'xp ayant path pour chemin et les ouvre
    l_path_scenes = find_all_f_with("Par_Cellule.csv", path)
    #print('l_path_scenes', l_path_scenes)
    l_df = [pd.read_csv(x) for x in l_path_scenes]
    # Recupere la liste des noms des fichiers
    l_filename_scenes = [path_to_filename(x) for x in l_path_scenes]
    #print('l_filename_scene',l_filename_scenes)
    # Ajout de la colonne classe, scene et date pour tous les df
    l_df = [ajouter_classe(df, filename) for df, filename in zip(l_df, l_filename_scenes)]
    l_df = [ajouter_scene(df, filename) for df, filename in zip(l_df, l_filename_scenes)]
    l_df = [ajouter_date(df,filename) for df, filename in zip(l_df, l_path_scenes)]
    l_df = [ajouter_age(df, recap_age) for df in l_df]
    #print(l_df)
    # Concatenation des df
    result = pd.concat(l_df)
    result.reset_index(level = None, drop = True, inplace=True)
    # Ajout d'une colonne d'identification unique (certain TrackID y sont plusieurs fois)
    result['IDunique'] = [x for x in range(1, result.shape[0]+1)]
    result = clean_mega(result)
    #Verifier si l'age n'est pas nul
    verif_age(result)
    return result


#NEW
# enregistre tous les mega csv au bon endroit
def save_all_mega_csv(l_path):
    #récupère la liste des dossiers dans le dossier ayant pour chemin path
    #l_folder = get_folder_list(path)
    # dossier est le nom du dossier et l'identifiant de l'xp
    prepare_age()
    for path in l_path:
        dossier = path_to_xp(path)
        print("DataFrame de l'expérience ", dossier," en cours de traitement ...")
        mega_df = get_mega_df(path)
        mega_df.to_csv(path + "/" + "MEGA_" + dossier + ".csv", index = False)
        #print("Enregistré à: ",path + dossier + "/" + "MEGA_" + dossier + ".csv")
    print("DONE !")
    
#MEGA FETCH
# renvoi la liste des chemins des xp
# Path : chemin de destination qui comprend IMARIS et FIJI une fois organisés
def get_xp_folder_path(path, l_p):
    l_xp = [path_to_xp(p) for p in l_p]
    l_all_xp_folder_fiji = get_folder_list(path + 'Data_Fiji_ORGANISED')
    l_f = [path + 'Data_Fiji_ORGANISED/' + xp for xp in l_all_xp_folder_fiji]
    l_all_xp_folder_imar = get_folder_list(path + 'Data_Imaris_ORGANISED')
    l_i = [path + 'Data_Imaris_ORGANISED/' + xp for xp in l_all_xp_folder_imar]
    l = l_f + l_i
    if l_xp[0] == 'NaN':
        return l
    else:
        l_f = [path + 'Data_Fiji_ORGANISED/' + xp for xp in l_xp if xp in l_all_xp_folder_fiji]
        l_i = [path + 'Data_Imaris_ORGANISED/' + xp for xp in l_xp if xp in l_all_xp_folder_imar]
        l = l_f + l_i
        print("liste path pour mega",l)
        return l


#####################################################################################################################
#FONCTIONS DE VERIFICATION
#####################################################################################################################
#VERIF
# Pour une cellule donnée vérifie si des temps sont en doubles
# Une cellule peut avoir plusieurs meme instant t => BUG
def no_double(i, df):
    l_time = df[df['TrackID']==i]['Time']
    l_one_time = l_time.drop_duplicates()
    return len(l_time) == len(l_one_time)

#VERIF
# Pour une cellule donnée vérifie si des temps sont manquants
# Il laisse une marge d'erreur de 20 temps
def no_hole(i, df):
    l_time = list(df[df['TrackID']==i]['Time'])
    # taille théorique de la liste 
    t = l_time[-1] - (l_time[0]-1)
    return len(l_time) >= t - 10
#VERIF
# Renvoie le nombre d'instants manquants
def size_hole(i, df):
    l_time = list(df[df['TrackID']==i]['Time'])
    t = l_time[-1] - (l_time[0]-1)
    return t-len(l_time)

#VERIF
# Verifie si il y a des doublons de temps dans les données brutes pour toutes les cellules d'une scène
def verif_dirty(df):
    l_id = df['TrackID'].copy()
    l_id = l_id.drop_duplicates()
    double = [i for i in l_id if not no_double(i, df)]
    hole = [i for i in l_id if not no_hole(i, df)]
    hole_size = [size_hole(i, df) for i in hole]
    return double, hole, hole_size


#VERIF AFFICH
# Verifie tous les df
def affiche_verif(l_df, scene, xp):
    for df in l_df:
        d, h, hs = verif_dirty(df)
        for i in d:
            print("_________________________________________________________________")
            print("            /!\\ ATTENTION PROBLEME DE DOUBLONS /!\\")
            print("Expérience ", xp, " | Scène ", scene, " | Cellule ", i)
            print("Fichier ", list(df.columns)[0], "(nom du fichier pas forcement exact)")
            print("_________________________________________________________________")
        for i,s in zip(h,hs):
            print("_________________________________________________________________")
            print("            /!\\ ATTENTION PROBLEME TIME MANQUANT /!\\")
            print("Expérience ", xp, " | Scène ", scene, " | Cellule ", i," | ", s," Temps manquants")
            print("Fichier ", list(df.columns)[0], "(nom du fichier pas forcement exact)")
            print("_________________________________________________________________")

#####################################################################################################################
#FONCTIONS DE MENU
#####################################################################################################################

#Chemin est un chemin enregistré pour faciliter la saisi
#bool_origine indique si cette fonction est appellé pour le chemin d'origine
def get_chemin_from_user(chemin, bool_origine):
    print("Voulez-vous garder ce chemin: ", chemin)
    rep = str(input("y/n ?"))
    if rep == "y":
        return chemin
    #L'utilisateur veut rentrer son propre chemin
    if bool_origine:
        path = str(input("Chemin du dossier qui contient toutes les données (imaris + fiji): "))
        while not os.path.isdir(path):
            print("Chemin NON VALIDE")
            path = str(input("Chemin du dossier qui contient toutes les données (imaris + fiji): "))
    else:
        path = str(input("Chemin du dossier de destination: "))
        print( "Valider ?")
        sur = str(input("y/n ?"))
        while sur != "y":
            path = str(input("Chemin du dossier de destination: "))
            print( "Valider ?")
            sur = str(input("y/n ?"))
    return path

def exclure(liste_xp):
    print("Expériences à organiser: ", liste_xp)
    rep = str(input("Voulez-vous exclure une ou plusieurs expérience?  y/n "))
    if rep == "y":
        todel = str(input("Date de l'expérience à exclure: (ecrire 'fin' pour arreter d'exclure)"))
        while todel != "fin":
            liste_xp.remove(todel)
            print("Expériences à organiser: ", liste_xp)
            todel = str(input("Date de l'expérience à exclure: (ecrire 'fin' pour arreter d'exclure)"))
    return liste_xp

#Fonction pour faciliter le debuggage
def exclure2(liste_xp):
    print("Expériences à organiser: ", liste_xp)
    rep = str(input("Voulez-vous exclure une ou plusieurs expérience?  y/n "))
    if rep == "y":
        todel = str(input("Date de l'expérience à partir de laquelle la transfo s'applique: (ecrire 'fin' pour arreter d'exclure)"))
        i=0
        while liste_xp[i] != todel :
            liste_xp = liste_xp[1:]
        print("Expériences à organiser: ", liste_xp)
    return liste_xp

#Fonction qui laisse le choix à l'utilisateur de transformer toutes ou une selection d'xp
def choix(datapath):
    print("VOULEZ-VOUS TRANSFORMER 1,2,... OU TOUTES LES EXPERIENCES ?")
    rep = int(input("0 pour TOUTES sinon combien? "))
    if rep == 0:
        return [datapath]
    else :
        l = [xp_path(datapath) for x in range(rep)]
        return l

#Fonction qui renvoi le chemin complet d'une xp
def xp_path(path):
    date = str(input("Date de l'experience? Année-Mois-Jour "))
    annee = date[0:4]
    xp_p = path + annee + "/" + date
    return xp_p
#####################################################################################################################
# VARIABLES GLOBALES et MAIN
#####################################################################################################################
# Chemins des dossiers IN et OUT

# # Variables debuggage ordi de Thomas
IN = "P:/Public/Imagerie/Videomicroscopie pièce 038/sylvie/"
OUT = "./DATA_ORGANISED/"
TableauTTPath = "./tableautheotime.xlsx"
#####################################################################################################################
# Variables du vrai environement d'application

"""
IN = "Z:/Public/Imagerie/videomicroscopie pièce 038/sylvie/"
OUT = "X:/02 PROJETS/sylviemimi/DATA_ORGANISED/"
TableauTTPath = "X:/02 PROJETS/sylviemimi/tableautheotime.xlsx"
"""
#####################################################################################################################
#####################################################################################################################
# RAJOUTER un element à cette liste pour que le script IGNORE tout fichier comportant cet element dans son chemin
# exemple tous fichiers csv contenus dans le dossier "xp2021-09-06_NePasPrendre_le nom peut continuer" seront ignorés
NePasPrendre = ["aranger","Rajouter", "ORGANISED", "NePasPrendre", "Caroline"]
######################################################################################################################

OUTDATAPATH_FIJI = OUT + "/Data_Fiji_ORGANISED/"
OUTDATAPATH_IMARIS = OUT + "/Data_Imaris_ORGANISED/"

if __name__=="__main__":
    #Affichage de la version
    print(" ")
    print("Script OrganiseAllData: Version du 13/09/2022")
    print(" ")

    # On verifie l'existance du chemin donné par l'utilisateur
    print("CHEMIN D'ORIGINE")
    DATAPATH = get_chemin_from_user(IN, bool_origine = True)
    print("CHEMIN DE DESTINATION")
    OUTDATAPATH = get_chemin_from_user(OUT, bool_origine = False)
    # On différencie les dossier destination par leur logiciel d'origine
    OUTDATAPATH_FIJI = OUTDATAPATH + "/Data_Fiji_ORGANISED/"
    OUTDATAPATH_IMARIS = OUTDATAPATH + "/Data_Imaris_ORGANISED/"

    # L'user choisi si il Organise toutes les XP ou un nombre précis
    liste_xp_path = choix(DATAPATH)

    #liste-xp_paths peut ne contenir qu'un element 
    #Transformation des données
    for xp in liste_xp_path:
        DO(xp, NePasPrendre)

    # MegaDataFrame
    print("CALCUL DES MEGA DATAFRAMES")
    liste_path_clean = get_xp_folder_path(OUT, liste_xp_path)

    save_all_mega_csv(liste_path_clean)

    print("Terminé !")