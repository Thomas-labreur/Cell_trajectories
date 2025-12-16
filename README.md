# Trajectoires des cellules

Ce projet est constitué de deux programme d'analyse de trajectoires de cellules produites avec les logiciels Fiji et Imaris. L'un organise les données en CSV uniformisés et l'autre est une application streamlit qui permet de générer des graphiques.

Avant d'utiliser les programmes de ce projet, veillez à installer python sur votre ordinateur. 

Ouvrez alors un terminal de commande comme powershell par exemple. Puis déplacez-vous vers le dossier où sont stockés:
    - Ce fichier README.md
    - Le fichier .env
    - Le fichier requirements.txt
    - Les deux fichiers Organize_All_Data_v2.py et trajectoires.py
Par exemple, si ces 5 fichiers sont stockés dans ""C:\\Users\\thomas.labreur\\Workspace\\Trajectoires", écrivez la commande:
```bash
cd "C:\\Users\\thomas.labreur\\Workspace\\Trajectoires"
```

## Installer les librairies

Si vous lancez le programme sur votre ordinateur pour la première fois, il faudra installer de nombreuses librairies python. Ces librairies sont listées dans le fichier 'requirements.txt' et peuvent toutes être isntallées d'un coup avec la commande suivante.
```bash
pip install -r requirements.txt
```

## Modifier le .env

Le fichier .env contient les chemins d'accès aux fichiers csv bruts, au tableau de Théotime, ainsi qu'à l'endroit où vous souhaitez sotcker les csv organisés. N'hésitez pas à ouvrir le fichier (avec bloc-notes par exemple) et modifier ces chemins d'accès pour les adapter à votre usage. Pensez à enregistrer avant de fermer !

## Lancer les programmes

Lancer le programme pour organiser les données
```bash
python .\Organize_All_Data_v2.py
```

Lancer le programme qui génère l'outil graphique pour tracer les trajectoires
```bash
streamlit run trajectoires.py
```