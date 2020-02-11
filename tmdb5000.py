#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_m = pd.read_csv("/Users/seangao/Desktop/tmdb_5000_movies.csv")

# PART 1: Movie Recommendation

# JSON objects are surrounded by curly braces {}. 
# JSON objects are written in key/value pairs.

# Convert json to string for column 'genres' and 'keywords'
df_m['genres'] = df_m['genres'].apply(json.loads)
for index,i in zip(df_m.index,df_m['genres']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))
    df_m.loc[index,'genres']=str(list1)

df_m['keywords'] = df_m['keywords'].apply(json.loads)
for index,i in zip(df_m.index,df_m['keywords']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))
    df_m.loc[index,'keywords']=str(list1)   

# Remove unwanted part in the column 'genres' and 'keywords'
df_m['genres'] = df_m['genres'].str.strip('[]').str.replace("'",'').str.replace('u','').str.replace(' ','')
df_m['keywords'] = df_m['keywords'].str.strip('[]').str.replace("u'",'').str.replace("'",'').str.replace(' ','')

# Do the same for 'production_companies', 'production_countries', and 'spoken_languages'
df_m['production_companies'] = df_m['production_companies'].apply(json.loads)
for index,i in zip(df_m.index,df_m['production_companies']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))
    df_m.loc[index,'production_companies']=str(list1)

df_m['production_companies'] = df_m['production_companies'].str.strip('[]').str.replace("u'",'').str.replace("'",'').str.replace(' ','')
    
df_m['production_countries'] = df_m['production_countries'].apply(json.loads)
for index,i in zip(df_m.index,df_m['production_countries']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))
    df_m.loc[index,'production_countries']=str(list1)

df_m['production_countries'] = df_m['production_countries'].str.strip('[]').str.replace("u'",'').str.replace("'",'').str.replace(' ','') 
    
df_m['spoken_languages'] = df_m['spoken_languages'].apply(json.loads)
for index,i in zip(df_m.index,df_m['spoken_languages']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))
    df_m.loc[index,'spoken_languages']=str(list1)

df_m['spoken_languages'] = df_m['spoken_languages'].str.strip('[]').str.replace("u'",'').str.replace("'",'').str.replace(' ','')


# In[2]:


# Do the same for column 'cast' in dataset df_c
df_c = pd.read_csv("/Users/seangao/Desktop/tmdb_5000_credits.csv")

df_c['cast'] = df_c['cast'].apply(json.loads)
for index,i in zip(df_c.index,df_c['cast']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))
    df_c.loc[index,'cast']=str(list1)

df_c['cast'] = df_c['cast'].str.strip('[]').str.replace("u'",'').str.replace('u"','').str.replace('"','').str.replace("'",'').str.replace(' ','')

# Extract key crew members from column 'crew'
df_c['crew'] = df_c['crew'].apply(json.loads)
def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
df_c['director'] = df_c['crew'].apply(director)

df_c['director'] = df_c['director'].str.replace(' ','')

def composer(x):
    for i in x:
        if i['job'] == 'Original Music Composer':
            return i['name']
df_c['original_music_composer'] = df_c['crew'].apply(composer)

df_c['original_music_composer'] = df_c['original_music_composer'].str.replace(' ','')

def screenplay(x):
    for i in x:
        if i['job'] == 'Screenplay':
            return i['name']
df_c['screenplay'] = df_c['crew'].apply(screenplay)

df_c['screenplay'] = df_c['screenplay'].str.replace(' ','')

def editor(x):
    for i in x:
        if i['job'] == 'Editor':
            return i['name']
df_c['editor'] = df_c['crew'].apply(editor)

df_c['editor'] = df_c['editor'].str.replace(' ','')

df_c.rename(columns={'movie_id':'id'},inplace=True)

df = pd.merge(df_m[['id','budget','genres','keywords',
                    'original_language','popularity','production_companies',
                    'production_countries','revenue','runtime','spoken_languages',
                    'title','vote_average','vote_count']],
              df_c[['id','cast','director','original_music_composer','screenplay',
                    'editor']],how='left')


# In[3]:


df.fillna(value = 'NaN', inplace = True)


# In[4]:


df['keycrews'] = df["director"] + ',' + df["original_music_composer"] + ',' + df["screenplay"] + ',' + df["editor"]

df['allfeatures'] = df['keywords'] + ',' + df['genres'] + ',' +  df['production_companies']  + ',' + df['production_countries']  + ',' + df['spoken_languages']  + ',' + df['cast']  + ',' + df['keycrews']

count = CountVectorizer(stop_words = 'english')
count_matrix = count.fit_transform(df['allfeatures'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index, index = df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim = cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]


# In[5]:


get_recommendations('Avatar', cosine_sim)


# In[6]:


# PART 2: Box Office Prediction
# Normalize the Budget Column

normbud = df['budget']
df['budget_norm'] = (normbud - normbud.min()) / (normbud.max() - normbud.min())

# Normalize the runtime

df['runtime'] = df['runtime'].astype(float)
normrt = df['runtime']
df['runtime_norm'] = (normrt - normrt.min()) / (normrt.max() - normrt.min())

# Import IMDB dataset & Compute Average IMDB Score for each director

imdb = pd.read_csv("/Users/seangao/Desktop/movie_metadata.csv",encoding='utf-8')

imdb['director'] = imdb['director_name'].str.replace(' ','')
dscore = imdb.groupby('director', as_index = False)['imdb_score'].mean()

# Merge with main dataset
df1 = pd.merge(df,dscore[['director','imdb_score']],how='left')
df1 = df1.rename(index=str, columns={'imdb_score': 'director_score'})
df1['director_score'].fillna(0, inplace=True)

# Split first 3 major actors/actresses
big3 = df1['cast'].str.split(',', n = 3, expand = True)
df1['major1']= big3[0]
df1['major2']= big3[1] 
df1['major3']= big3[2]

top = pd.read_csv("/Users/seangao/Desktop/imdb_top_actors_actresses.csv",encoding='utf-8')
top['name'] = top['name'].str.replace(' ','')

df1 = pd.merge(df1, top, left_on=  ['major1'],
                   right_on= ['name'], 
                   how = 'left')
del df1['name']
df1 = df1.rename(index=str, columns={'mark': 'major1_mark'})
df1 = pd.merge(df1, top, left_on=  ['major2'],
                   right_on= ['name'], 
                   how = 'left')
del df1['name']
df1 = df1.rename(index=str, columns={'mark': 'major2_mark'})
df1 = pd.merge(df1, top, left_on=  ['major3'],
                   right_on= ['name'], 
                   how = 'left')
del df1['name']
df1 = df1.rename(index=str, columns={'mark': 'major3_mark'})
df1.fillna(value = 0, inplace = True)

# Further Cleaning
df1 = df1[df1.title != 'Avatar'] # Remove Outlier
df1 = df1[df1.revenue != 0]


# In[7]:


# create dataset for regression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# df2 = df1[['title','revenue','director_score','major1_score','major2_score','major3_score',
           # 'genres1_score','genres2_score','genres3_score','k1_score','k2_score',
           # 'k3_score','runtime_norm','budget_norm']]
        
df2 = df1[['title','revenue','director_score','major1_mark','major2_mark','major3_mark',
           'runtime_norm','budget_norm']]

# Normalize the Revenue
normrev = df2['revenue']
df2['revenue_norm'] = (normrev - normrev.min()) / (normrev.max() - normrev.min())

X = df2[['director_score','major1_mark','major2_mark','major3_mark',
         'runtime_norm','budget_norm']]
y = df2[['revenue_norm']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

X_train.fillna(value = 0, inplace = True)
X_test.fillna(value = 0, inplace = True)
y_train.fillna(value = 0, inplace = True)
y_test.fillna(value = 0, inplace = True)


# Linear Regression
lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

Linear_train_score = lm.score(X_train,y_train)
Linear_test_score = lm.score(X_test, y_test)

MSE = mean_squared_error(y_test, predictions)
R_squared = r2_score(y_test, predictions)

for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, lm.coef_[0][idx]))

print('Linear Regression R-Squared Score: ' + str(R_squared))
print('Linear Regression MSE: ' + str(MSE))

# Ridge Regression

rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)

Ridge_R_squared = rr.score(X_test, y_test)

print('Ridge Regression R-Squared Score with Small Alpha: ' + str(Ridge_R_squared))

rr10 = Ridge(alpha=10)
rr10.fit(X_train, y_train)

Ridge_R_squared = rr10.score(X_test, y_test)

print('Ridge Regression R-Squared Score with medium Alpha: ' + str(Ridge_R_squared))

rr100 = Ridge(alpha=100) # Compare Different Alpha
rr100.fit(X_train, y_train)

Ridge_R_squared_1 = rr100.score(X_test, y_test)

print('Ridge Regression R-Squared Score with Large Alpha: ' + str(Ridge_R_squared_1))

# Lasso Regression

lasso = Lasso()
lasso.fit(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)

print('Lasso Regression R-Squared Score: ' + str(test_score))
print('Number of features used: ' + str(coeff_used))

lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)

print('Lasso Regression R-Squared Score with 0.01 Alpha: ' + str(test_score001))
print('Number of features used: ' + str(coeff_used001))

lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)

print('Lasso Regression R-Squared Score with 0.0001 Alpha: ' + str(test_score00001))
print('Number of features used: ' + str(coeff_used00001))


# In[8]:


# PCA Analysis & KMeans
from sklearn import decomposition
from sklearn.cluster import KMeans

pca = decomposition.PCA(n_components='mle')
pc = pca.fit_transform(X)
pc_df = pd.DataFrame(data = pc)
pca.explained_variance_ratio_

pca_2 = decomposition.PCA(n_components=2)
pc_2 = pca_2.fit_transform(X)
pc_2_df = pd.DataFrame(data = pc_2)
pca_2.explained_variance_ratio_

pc_2_df['PC1']=pc_2_df[0]
pc_2_df['PC2']=pc_2_df[1]

plt.scatter(pc_2_df['PC1'],pc_2_df['PC2'], label='True Position')

kmeans = KMeans(n_clusters=2)  
kmeans.fit(pc_2_df[['PC1','PC2']])
print(kmeans.cluster_centers_)
print(kmeans.labels_)
plt.scatter(pc_2_df['PC1'],pc_2_df['PC2'], c=kmeans.labels_, cmap='coolwarm')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='red',marker='*',s=200)
plt.xlabel("PC1")
plt.ylabel("PC2")


# In[10]:


# SVR
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from numpy import ravel

clf = SVR()
clf.fit(X_train,y_train)
SVR_score = clf.score(X_test, y_test)

# RFR

X_train_rfr = X_train.values
X_test_rfr = X_test.values
y_train_rfr = y_train.values
y_train_rfr = ravel(y_train_rfr)
y_test_rfr = y_test.values
y_test_rfr = ravel(y_test_rfr)

rfr = RandomForestRegressor(n_estimators = 100)
rfr.fit(X_train_rfr,y_train_rfr)
RFR_score = rfr.score(X_test_rfr,y_test_rfr)

# Decision Tree

y_train_dt = y_train.values
y_test_dt = y_test.values

dtreg = DecisionTreeRegressor()
dtreg.fit(X_train_rfr,y_train_dt)
DTR_score = dtreg.score(X_test_rfr, y_test_dt)

print(SVR_score)
print(RFR_score)
print(DTR_score)


# In[56]:


# Deep Learning

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

dlX = X_train.values
dly = y_train.values
dlX_test = X_test.values
dly_test = y_test.values

# define model
def model():
    model = Sequential()
    model.add(Dense(10, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

seed = 8
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=30, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(pipeline, dlX, dly, cv=kfold)
print("Standardized: %.5f (%.5f) MSE" % (results.mean(), results.std()))


# In[50]:


pca.explained_variance_ratio_

