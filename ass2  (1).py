import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Data
movies = pd.DataFrame({
 'title':['The Shawshank Redemption','The Godfather','The Dark Knight','Pulp Fiction','The Lord of the Rings: The Return of the King'],
 'genres':['Drama','Crime, Drama','Action, Crime, Drama','Crime, Drama','Action, Adventure, Fantasy'],
 'desc':['Two imprisoned men bond over years.','Patriarch transfers crime dynasty.','Joker wreaks havoc on Gotham.','Mob hitmen tales of violence.','Gandalf and Aragorn fight Sauron.']
})
movies['content']=movies['genres']+' '+movies['desc']

# TF-IDF + Similarity
tfidf=TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])
sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Recommendation Function
def recommend(title):
    i = movies[movies['title'] == title].index[0]
    scores = sorted(enumerate(sim[i]), key=lambda x: x[1], reverse=True)[1:4]
    return movies['title'].iloc[[j for j, _ in scores]]


# Output
print("Recommendations for 'The Godfather':\n", recommend('The Godfather'))
