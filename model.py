import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def check_game(title):
    if title in all_games:
        return True
    return False

def get_recommendation(title,n):
    global names
    names=[]
    global distances,indices
    index_num=game_features_df.reset_index()[game_features_df.index==title].index.values[0]
    distances, indices = model_knn.kneighbors(game_features_df.iloc[index_num,:].values.reshape(1, -1), n_neighbors = n)
    for i in range(0, len(distances.flatten())):
        names.append(game_features_df.index[indices.flatten()[i]])
    
    return names


game_info=pd.read_csv("./DATA/metacritic_game_info.csv")
game_5= pd.read_csv("./DATA/game_5.csv")
game_6= pd.read_csv("./DATA/game_6.csv")
game_7= pd.read_csv("./DATA/game_7.csv")
game_8= pd.read_csv("./DATA/game_8.csv")
game_9= pd.read_csv("./DATA/game_9.csv")
game_10= pd.read_csv("./DATA/game_10.csv")
game_11= pd.read_csv("./DATA/game_11.csv")
game_comm=pd.concat([game_5,game_6,game_7,game_8,game_9,game_10,game_11],0)
game_info=game_info.iloc[:,1:]
game_comm=game_comm.iloc[:,2:]
df=game_comm.merge(game_info,on='Title')
df=df.dropna()
df=df[['Title','Username','Metascore','Userscore']]
mean_data=pd.DataFrame(df.groupby('Title')['Userscore'].mean().reset_index())
mean_dict=dict(sorted(mean_data.values.tolist())) 
df['Userscore_mean']=df['Title'].map(mean_dict)
score_count=pd.DataFrame(df.groupby('Title')['Userscore_mean'].count().sort_values(ascending=False).reset_index().rename(columns = {'Userscore_mean': 'Userscore_count'}))
rating_with_count = df.merge(score_count, left_on = 'Title', right_on = 'Title', how = 'left')
UserCountplusScore = df.merge(score_count, left_on = 'Title', right_on = 'Title', how = 'left')
count_threshold = 100
rating_popular_game= UserCountplusScore.query('Userscore_count >= @count_threshold')
game_features_df=rating_popular_game.pivot_table(index='Title',columns='Username',values='Userscore_mean').fillna(0)
game_features_df_matrix = csr_matrix(game_features_df.values)
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(game_features_df_matrix)
all_games = game_features_df.index.to_list()

    

