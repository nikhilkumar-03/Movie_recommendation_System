import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movies_500.csv")
df['description'] = df['description'].fillna("")

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description'])

similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

index_map = pd.Series(df.index, index=df['title'].str.lower())

def recommend_movies(title, top_n=5):
    title = title.lower()
    if title not in index_map:
        return ["Movie not found in the database."]
    idx = index_map[title]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_matches = sorted_scores[1:top_n + 1]
    recommended_indices = [i[0] for i in top_matches]
    return df['title'].iloc[recommended_indices].tolist()

def main():
    print("Movie Recommendation System")
    movie_name = input("Enter a movie title: ")
    results = recommend_movies(movie_name)
    print("\nRecommended Movies:")
    for movie in results:
        print("-", movie)

if __name__ == "__main__":
    main()
