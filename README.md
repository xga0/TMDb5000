# TMDb5000
The first part of this project is a movie recommendation system based on cosine similarity. The Jupyter notebook for this part is: https://github.com/xga0/tmdb5000/blob/master/moviePrediction.ipynb.

Then, we conducted multiple algorithms to train a model to predict the box office of a movie with giving features.

We used 4 datasets in total in this project. For the first part, we used the TMDb 5000 datasets (https://www.kaggle.com/tmdb/tmdb-movie-metadata) from Kaggle. The datasets have 2 subsets: TMDb 5000 Movies and TMDb 5000 Credits. TMDb 5000 Movies has 4804 observations and 20 variables, including id, budget, genres, keywords, etc. There are four variables, movie_id, title, cast, and crew, in the TMDb 5000 Credits with 4814
observations. Similar as IMDb, The Movie Database (TMDb) is a popular, user editable database for movies and TV shows. Since 2008, with over 200,000 developers and companies using their platform, TMDb has become a premiere source for metadata. 

In order to prepare the dataset for second part, we combined 3 extra datasets: IMDb 5000 Movie Dataset (https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset), The Most Popular
Actors (https://www.imdb.com/list/ls022928819/), and The Most Popular Actresses (https://www.imdb.com/list/ls022928836/). From 27 variables in IMDb 5000 Movie Dataset, we mainly used director_name and imdb_score. Moreover, we used top 300 names in both The Most Popular Actors and The Most Popular Actresses.
