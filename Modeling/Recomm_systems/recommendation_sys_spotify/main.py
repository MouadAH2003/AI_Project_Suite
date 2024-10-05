import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import seaborn as sns
# import plotly.express as px


from sklearn.cluster import k_means
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("./data/data.csv")
genre_data = pd.read_csv("./data/data_by_genres.csv")
year_data = pd.read_csv("./data/data_by_year.csv")

# print(data.info())
# print(genre_data.info())
# print(year_data.info())

#from yellowbrick.target import FeatureCorrelation

feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']

X, y = data[feature_names], data["popularity"]

# Create a list of the features names
features = np.array(feature_names)


corr_matrix = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()


## Implementing some EDA


def get_decade(year):
    period_start = int(year/10)*10
    decade = '{}s'.format(period_start)
    return decade

data["decade"]  = data["year"].apply(get_decade)
sns.get(rc={"figure.figsize":(11,6)})

sns.countplot(data["decade"])

plt.show()