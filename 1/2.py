import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df1 = pd.read_csv(rf"C:\Users\ab\Downloads\RS-A2_A3_movie.csv")
df2 = pd.read_csv(rf"C:\Users\ab\Downloads\RS-A2_A3_tag.csv")

movies = df1.merge(df2, on='movieId').fillna('')
movies['context'] = movies['genres'] + " " + movies['tag']

# IMPORTANT: reset index after sampling
movies = movies.sample(10000).reset_index(drop=True)
print(movies.head())


tfidf = TfidfVectorizer(stop_words='english')
mat = tfidf.fit_transform(movies['context'])
sim = linear_kernel(mat, mat)

def reco(id):
    # row index in the sampled dataset
    i = movies[movies['movieId'] == id].index
    i = i[0]
    score = list(enumerate(sim[i]))
    top = sorted(score, key=lambda x: x[1], reverse=True)[1:4]
    for j, _ in top:
        print(movies.loc[j, 'title'])