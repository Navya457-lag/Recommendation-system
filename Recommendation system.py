"""
Created on Tue Sep 24 12:47:52 2024
"""

import pandas as pd
df = pd.read_csv("D:\\My Classes\\ExcelR\\Sessions ExcelR\\Latest DS Material\\Day 21- Recommendation systems\\Recommendation Engine\\Movie.csv")
df

df["userId"].unique()
len(df["userId"].unique())
len(df["movie"].unique())


df["movie"].value_counts()

df["rating"].value_counts()
df["rating"].hist()


user_df = df.pivot(index='userId',
                                 columns='movie',
                                 values='rating')


user_df


user_df.fillna(value = 0, inplace=True)
user_df


from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df,metric='cosine')
user_sim

import numpy as np
np.fill_diagonal(user_sim,0)

user_sim  = pd.DataFrame(user_sim)
user_sim

user_sim.index = df["userId"].unique()
user_sim.columns = df["userId"].unique()
user_sim

user_sim.max()

user_sim.idxmax()


df[(df['userId'] == 3) | (df['userId'] == 11)]
df[(df['userId'] == 6) | (df['userId'] == 168)]

#=-----------------------------------------------












