# dsc478_final

This project is a content-based recommender system that uses the Spotify WebAPI, Spotipy, and KNN Search to build a playlist given an existing Spotify playlist. The new playlist will contain songs that are similar to the songs in the original playlist in terms of audio features, artists, & genre. The original playlist is used to construct a user profile by summarizing the features of each song into a summarized feature vector & comparing the values of those features in other songs in the same feature space. The system was implemented using PCA transformed data, but it was ultimately decided tha TF-IDF transformed data performed better when creating playlists & when measuring cosine similarities.

**Running the Application**

DISCLAIMER: A Spotify account is needed to run the application (primarily to create the final playlist with your suggested songs). Spotify playlists are FREE to create and use. Sign up for spotify here: https://www.spotify.com/us/signup?forward_url=https%3A%2F%2Fopen.spotify.com%2F

1. clone the repo 
2. create config.py file setting variables SPOTIFY_CLIENT_ID, SPOTIFY_SECRET, and REDIRET_URI
3. save the config.py file in this repo
4. navigate to the repo using the command prompt

~~~
marvazqu8 ~ % cd mv/.../.../.../dsc478_final
~~~

5. run the following command and follow along! (note: your command for python may be different)

~~~
marvazqu8 dsc478_final % python3 app.py
~~~
