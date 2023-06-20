import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.neighbors import NearestNeighbors
# TITRE
st.markdown("<h1 style='text-align: center;color: black;'>Cinéma de la Creuse</h1>", unsafe_allow_html=True)
# Image de couverture
image_url = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/901370a7-ec10-4044-901c-82a189cc8036/d38rnod-2f6cb815-011d-47a8-bb9f-448c20f994c9.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzkwMTM3MGE3LWVjMTAtNDA0NC05MDFjLTgyYTE4OWNjODAzNlwvZDM4cm5vZC0yZjZjYjgxNS0wMTFkLTQ3YTgtYmI5Zi00NDhjMjBmOTk0YzkuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.jARAKDZXyoFndPQ0PbPD1rt54hFDhvHtXpm5rY-aQJE"
st.image(image_url, caption='IMDB Movies')
# Affiche le DataFrame avec 6100 films
title = ''
df = pd.read_csv("/Users/julienpeneau/Desktop/WILD CODE SCHOOL/Projet2/Nettoyage/table_filtré_final.tsv", sep='\t')
# Mise en place du filtre
df1 = df.loc[(df['averageRating'] >= 7.0) & (df['runtimeMinutes'] >= 60.0) & (df['startYear'] >= 1970.0) & (df['numVotes'] > 3000)]
# Filtrer les films dont le titre contient la chaîne de caractères (insensible à la casse)
matching_films = df1[df1['primaryTitle'].str.contains(title, case=False)].drop_duplicates(subset='tconst')
# On affiche notre DataFrame avec tous les films (titre, année et tconst)
affichage = (matching_films[['primaryTitle', 'startYear', 'tconst']])
st.dataframe(affichage)
# Couleur du background de notre streamlit
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {background-color: black;
opacity: 1;
background-image:  linear-gradient(to right top, #1655B4, #0082E1, #00A8D4, #00C790, #1CDD16);;}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# Sidebar de streamlit
with st.sidebar:
    # Ajouter le titre 'Choisissez votre film' avant le 'input'
    st.markdown("<h1 style='text-align: center;'>Choisissez votre film</h1>", unsafe_allow_html=True)
    title = st.text_input('')
    # Charger votre dataframe contenant les titres des films
    df = pd.read_csv("/Users/julienpeneau/Desktop/WILD CODE SCHOOL/Projet2/Nettoyage/table_filtré_final.tsv", sep='\t')
    # Mise en place du filtre
    df1 = df.loc[(df['averageRating'] >= 7.0) & (df['runtimeMinutes'] >= 60.0) & (df['startYear'] >= 1970.0) & (df['numVotes'] > 3000)]
    # Filtrer les films dont le titre contient la chaîne de caractères (insensible à la casse)
    matching_films = df1[df1['primaryTitle'].str.contains(title, case=False)].drop_duplicates(subset='tconst')
    # Afficher les noms des films correspondants / to_string améliore affichage et supprime l'affichage de index
    affichage = (matching_films[['tconst','primaryTitle', 'startYear']])
    st.write(affichage)
    # Input pour entrer 'tconst'
    tconst = st.text_input('Entrez le "tconst" du film que vous avez choisi')
    # Lire le csv utilisé pour la recommandation de film
    df = pd.read_csv("/Users/julienpeneau/Desktop/WILD CODE SCHOOL/Projet2/Nettoyage/table_filtré_final.tsv", sep='\t')
    # Filtrer le dataframe pour n'avoir que les conditions souhaitées : (film d'au moins 1 heure, après les années 70, et ayant plus de 3000 votes)
    df1 = df.loc[(df['averageRating'] >= 7.0) & (df['runtimeMinutes'] >= 60.0) & (df['startYear'] >= 1970.0) & (df['numVotes'] > 3000)]
    # Concaténer le nouveau dataframe avec une nouvelle colonne pour chaque genre avec le dataframe initial
    df_ml = pd.concat([df1 , df1['genres'].str.get_dummies(sep=',')], axis=1)
    # Factoriser (transformer les valeurs non numériques en valeurs numériques) la colonne "nconst" représentant les acteurs/actrices/réalisateurs, de manière à pouvoir l'utiliser pour l'apprentissage automatique
    df_ml["nconst"] = df_ml["nconst"].factorize()[0]
    # Supprimer les colonnes numériques qui ne nous intéressent pas, afin de se concentrer uniquement sur les colonnes pertinentes (acteurs, genre, etc.)
    df_colonne_sup = df_ml[['averageRating', 'numVotes', 'startYear', 'runtimeMinutes']]
    # Créer notre X pour entraîner le modèle avec les colonnes souhaitées, c'est-à-dire uniquement les colonnes numériques pertinentes
    X = df_ml.select_dtypes('number').drop(df_colonne_sup, axis=1)
    # Créer le modèle puis l'entraîner (ex : 3 plus proches voisins de James Cameron, 3 PPV de Leo...)
    model = NearestNeighbors(n_neighbors=3).fit(X)
    # Obtenir l'index du/des film(s) pour lesquels nous voulons faire des recommandations
    index_films = X[X.index.isin(df_ml.loc[df_ml["tconst"] == tconst].index)]
    # On cherche les voisins les plus proches des "index_films" que l'on souhaite
    # le model retourne 2 tableaux (distances et indices)
    # Pour que le resltat soit plus simple à utiliser (accèder aux films recommandés),
    # on utilise .flatten() qui permet qui convertit le tableau multidimensionnel en tableau unidimensionnel
    # La 2ème ligne  sélectionne la colonne "primaryTitle" du DataFrame pour obtenir les titres des films correspondants
    distances, indices = model.kneighbors(index_films)
    recommandations = df_ml.iloc[indices.flatten()][["tconst","primaryTitle",'startYear','averageRating','numVotes']].values.tolist()
    recommandations_df = pd.DataFrame(recommandations, columns=[ "tconst",'films_proches','startYear','averageRating','numVotes'])
    # Enlever les doublons des films dans les films proches voisins + enlever le film lui même car c'est lui le plus proche voisin
    recommandations_df = recommandations_df.drop_duplicates(subset=['films_proches'], keep='first')
    recommandations_df = recommandations_df.drop(recommandations_df.index[0])
    # Afficher les 10 premiers voisins trouvés
    st.write(recommandations_df.head(10))