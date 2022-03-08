# This is a sample Python script.

import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie_data = pd.read_csv('movie_dataset.csv')
print(movie_data)

selected_features = ['genre', 'crew']
print(selected_features)

combined_features = movie_data['genre'] + ' ' + ['crew']
print(combined_features)

vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
print("this is feature vectore")
print(feature_vector)

print("now getting similarity")
similarity = cosine_similarity(feature_vector)
print(similarity)
print(similarity.shape)

movie_name = input('Enter your movie name')

print("taking all list of movies")
list_of_all_title = movie_data['title'].to_list()
print(list_of_all_title)

find_close_match = difflib.get_close_matches(movie_name, list_of_all_title)
print(find_close_match)

close_match = find_close_match[0]
print(close_match)

print("finding index of movie")
index_of_movie = movie_data[movie_data.title == close_match].index.values[0]
print(index_of_movie)
if index_of_movie ==0:
    print("movie not found in database")
else:

    print("finding similarity of the best match movies")
    somilarity_score = list(enumerate(similarity[index_of_movie]))
    print(somilarity_score)

    print("sorting the movie based on similarity score")
    sorted_similar_movies = sorted(somilarity_score, key=lambda x: x[1], reverse=True)
    display =sorted_similar_movies[2:6]
    #print(sorted_similar_movies)

    print("movie suggested for you are:\n")
    i = 1
    for movie in display:
        index = movie[0]
        title_from_index = movie_data[movie_data.index == index]['title'].values[0]
        if i <= 6:
            print(title_from_index)
        else:
            print("not found in database")
        

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
