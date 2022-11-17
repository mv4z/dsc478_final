#IMPORTS
import numpy as np
import pandas as pd
import re
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import pairwise

class Playlist():
    '''Class to represent a Spotify Playlist
        -------------------------------------------
        Parameters:
        - link: HTTP link to the playlist
        - spotify_object: A spotify Object
        -------------------------------------------
        Attributes:
        - link: HTTP link to the playlist
        - sp: A Spotify object to interact with the Spotify API
        - audio_features: DataFrame containing audio features of songs in the playlist
        - normalized_audio_features: DataFrame containing normalized audio features of songs in the playlist
        - numeric_features: DataFrame containing numeric features of all songs in the playlist
        - normalized_numeric_features: DataFrame containing normalized numeric features of all songs in the playlist
        - pca_transformed_features: DataFrame containing PCA transformed features of the playlist    
        - df_playlist: DataFrame containing all features of songs in the playlist
    '''

    def __init__(self, link, spotify_object, scaler=MinMaxScaler) -> None:
        self.link = link
        self.sp = spotify_object
        playlist = self.sp.playlist(self.link)
        # this is the base dataset. next datasets use this one in some sort of way
        df_playlist = self.build_playlist_df(playlist)

        #get the audio features for all of the songs in df_playlist
        self.audio_features = pd.concat([df_playlist.track_id, self.get_audio_features(df_playlist)], axis=1)
        cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
        self.normalized_audio_features = pd.concat([df_playlist.track_id, self.normalize_audio_features(scaler, self.audio_features.loc[:, cols])], axis=1) 
        
        # audio features + popularity = numeric_features
        cols.append('popularity')
        self.numeric_features = pd.concat([self.audio_features, df_playlist.popularity], axis=1)
        self.normalized_numeric_features = pd.concat([df_playlist.track_id, self.normalize_audio_features(scaler, self.numeric_features.loc[:,cols])], axis=1) 

        # can do PCA transformation on normalized_numeric_features
        self.pca_transformed_features = None
        self.df_playlist = pd.concat([df_playlist, self.numeric_features], axis=1)
        

    def build_playlist_df(self, playlist):
        '''Create a DataFrame with track information about songs that are on the playlist
           -------------------------------------------
           parameters:
           - playlist: the playlist for which to build a DataFrame
           -------------------------------------------
           returns:
           - playlist_df
        '''

        playlist = playlist["tracks"]["items"]
        playlist_df = pd.DataFrame(columns=['song_title', 'artist','artist_id' ,'album', 'track_id', 'popularity', 'genres'], index=None)
        track_position = 1

        for song in playlist:
            try:
                track = song["track"]
            except:
                track = None
            
            try:
                name = track["name"]
            except:
                name = None

            try:
                # artist_list = track["artists"]
                album = track["album"]["name"]
            except:
                album = None

            try:    
                track_id = track["id"]
            except:
                track_id = None

            try:
                # track popularity
                track_pop = track["popularity"]
            except:
                track_pop = None
            
            # some songs have more than 1 artist
            # create a list of all artists on the song
            # artists = []
            # for a in range(len(artist_list)):
            #     artists.append((artist_list[a]['name'], artist_list[a]['id']))
                
            # artist_list = [x[0] for x in artists]
            # artist_id_list = [x[1] for x in artists]

            # artist_genres = [self.sp.artist(artist_id)["genres"] for artist_id in artist_id_list]
            # artist_genres = [*set([item for sublist in artist_genres for item in sublist])]

            if track_id != None:
                # to get only one artist per song, get the artist from the ALBUM
                album_artist_list = track["album"]['artists']
                artist = album_artist_list[0]["name"]
                artist_id = album_artist_list[0]["id"]

                # get genres for this artist and put into a list to be processed later
                artist_genres = self.sp.artist(artist_id)["genres"]
                if len(artist_genres) == 0:
                    artist_genres =['unknown']
                else:
                    artist_genres = ",".join([re.sub(' ','_',i) for i in artist_genres])
                    artist_genres = list(artist_genres.split(','))

            # if track_id == None that means that the audio features for this song could not be found
            # the track is omitted from analysis & use when creating a new playlist
            if track_id != None:
                playlist_df.loc[track_position] = {'song_title' : name,
                                            'artist' : artist, #artist_list,
                                            'artist_id' : artist_id, #artist_id_list,
                                            'album' : album, 
                                            'track_id' : track_id,
                                            'popularity':track_pop,
                                            'genres' : artist_genres
                                            }
                
                track_position+=1
            elif track_id == None:
                print('\nThese are the songs which could not be analyzed. (this is probably because theyre not on spotify):\n', name)
                print('The application will not use these songs when curating your new playlist.')

        playlist_df = playlist_df.reset_index()
        return playlist_df

    def get_audio_features(self, playlist):
        '''Return a DataFrame for the audio features in a playlist
           -------------------------------------------
           parameters:
            - playlist: DataFrame containing track_ids to get audio features for
           -------------------------------------------
           returns
           - audio_features_df: DF containing only the audio features of a playlist
        '''

        # columns for audio features
        playlist_audio_features = ["danceability", "energy", "key",
            "loudness", "mode", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]

        # get track audio features using Spotify API
        track_list = playlist.track_id.ravel()
        audio_features_df = pd.DataFrame()
        # track_audio_features = self.sp.audio_analysis(track_list[0])['track']
        track_audio_features = self.sp.audio_features(track_list)

        #drop songs that arent found in spotify
        track_list = [x for x in track_list if (x != None)]
        track_audio_features = self.sp.audio_features(track_list)

        ind = 1
        for audio_features in track_audio_features:
            song_features = pd.DataFrame(audio_features, index=[ind])
            audio_features_df = pd.concat([audio_features_df,song_features], axis=0)
            ind+=1
        
        audio_features_df = audio_features_df.reset_index()

        return audio_features_df.loc[:, playlist_audio_features]

    # def get_artist_counts(self):
    #     '''Get occurence of artists in a playlist
    #        -------------------------------------------
    #        parameters:
    #        -------------------------------------------
    #        returns:
    #        - artist_counts: occurrences of artists in playlist (song features included)
    #        - artist_id_counts: occurrences of artist_ids in playlist (song features included)
    #     '''
    #     artist_counts_in_playlist = {} #count the occurrences of each artist in the playlist (including artist features)
    #     artist_id_counts_in_playlist = {}

    #     artists = self.df_playlist['artist']
    #     artist_ids = self.df_playlist['artist_id']

    #     for i in artists.index:
            
    #         track_artists_lst = artists[i]
    #         track_artist_ids_lst = artist_ids[i]
            
    #         for a in track_artists_lst:
    #             if a not in artist_counts_in_playlist.keys():
    #                 artist_counts_in_playlist[a] = 1
    #             else:
    #                 artist_counts_in_playlist[a] += 1
                    
    #         for a_id in track_artist_ids_lst:
    #             if a_id not in artist_id_counts_in_playlist.keys():
    #                 artist_id_counts_in_playlist[a_id] = 1
    #             else:
    #                 artist_id_counts_in_playlist[a_id] += 1

    #     artist_counts = pd.DataFrame(artist_counts_in_playlist, index=range(0,1))
    #     artist_id_counts = pd.DataFrame(artist_id_counts_in_playlist, index=range(0,1))

    #     return artist_counts, artist_id_counts

    def get_artist_counts2(self):
        '''Get occurence of artists in a playlist
           -------------------------------------------
           parameters:
           -------------------------------------------
           returns:
           - artist_counts: occurrences of artists in playlist (song features included)
           - artist_id_counts: occurrences of artist_ids in playlist (song features included)
        '''
        artist_counts = pd.DataFrame(self.df_playlist.artist.value_counts())
        artist_id_counts = pd.DataFrame(self.df_playlist.artist_id.value_counts())

        return artist_counts, artist_id_counts

    def normalize_audio_features(self, scaler, data):
        '''Normalize audio features using scaler
           -------------------------------------------
           parameters:
           - scaler: scaling method to use
           - data: data to scale
           -------------------------------------------
           returns:
           scaled_audio_features: data normalized using scaler
        '''

        scaler = scaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data), index=None)

        return scaled_data

    # def get_recommendations(self, search_res, comparison_values, tf_idf=False, pca_components=-1, k=5):
    #     ''' Get song song recommendations for a list of artists
    #         -------------------------------------------
    #         parameters:
    #         - search_res: a dictionary of artists & artist search result songs
    #         - comparison_values: values used for comparison in knn search
    #         - pca_components: the number of components to choose when performing PCA transformation
    #         - k: number of neighbors to find in KNN search. (default = 5)
    #         -------------------------------------------
    #         returns:
    #         - recommended_songs: DataFrame containing Artists, their k most similar songs (from search result) & accompanying info, and similarity to comparison_values
    #     '''

    #     recommended_songs = pd.DataFrame(index=None, columns=['artist', 'track', 'track_id', 'similarity'])

    #     # iterate over artists & their songs
    #     for key in search_res.keys():
    #         songs_search_res = search_res[key]
            
    #         # get audio features for all of the artists songs from search
    #         search_res_audio_features = self.get_audio_features(songs_search_res)
    #         search_res_numeric_features = pd.concat([search_res_audio_features, songs_search_res['popularity']], axis=1)
            
    #         # scale audio features
    #         # minmax scaling used instead of standardscaler to make values between 0 nd 1
    #         scaled_search_res_numeric_features = self.normalize_audio_features(MinMaxScaler, search_res_numeric_features)
    #         # scaled_search_res_numeric_features = self.normalize_audio_features(StandardScaler, search_res_numeric_features)

    #         scaled_numeric_features =  np.array(scaled_search_res_numeric_features)
    #         #perform tf_idf transformation on the DF with songs from search results for each artist
    #         if tf_idf:
    #             tfidf = TfidfVectorizer()
    #             tfidf_matrix =  tfidf.fit_transform(songs_search_res.genres.apply(lambda x: " ".join(x)))
    #             genre_df = pd.DataFrame(tfidf_matrix.toarray())
    #             genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
    #             genre_df.reset_index(drop = True, inplace=True)
    #             # print(pd.concat([scaled_search_res_numeric_features, genre_df], axis=1).columns)
    #             scaled_numeric_features = np.array(pd.concat([scaled_search_res_numeric_features, genre_df], axis=1))

    #         if pca_components > 0:
    #             #pca transform the data
    #             pca = decomposition.PCA(svd_solver='randomized')
    #             scaled_numeric_features = pca.fit_transform(scaled_numeric_features)
    #             scaled_numeric_features = scaled_numeric_features[:, :pca_components]

                
    #         # KNN search for k most similar songs from search results
    #         idxs, sims = self.knn_search(comparison_values, scaled_numeric_features, k, pairwise.cosine_similarity)
    #         # add all songs to recommended songs DF
    #         tmp = pd.DataFrame({'artist': key,
    #             'track' : songs_search_res.iloc[idxs,:].track,
    #             'track_id' : songs_search_res.iloc[idxs,:].track_id,
    #             'similarity' : [sims[x] for x in idxs]}, index=None)
    #         recommended_songs = pd.concat([recommended_songs, tmp], axis=0, ignore_index=True)

    #     return recommended_songs

    def get_recommendations2(self, search_res, comparison_values, tf_idf=True, pca_components=-1, k=5):
        ''' Get song song recommendations for a list of artists
            -------------------------------------------
            parameters:
            - search_res: a dictionary of artists & artist search result songs
            - comparison_values: values used for comparison in knn search
            - pca_components: the number of components to choose when performing PCA transformation
            - k: number of neighbors to find in KNN search. (default = 5)
            -------------------------------------------
            returns:
            - recommended_songs: DataFrame containing Artists, their k most similar songs (from search result) & accompanying info, and similarity to comparison_values
        '''

        recommended_songs = pd.DataFrame(index=None, columns=['artist', 'track', 'track_id', 'similarity'])

        # iterate over artists & their songs
        for key in search_res.keys():
            songs_search_res = search_res[key]
            
            # get audio features for all of the artists songs from search
            search_res_audio_features = self.get_audio_features(songs_search_res)
            search_res_numeric_features = pd.concat([search_res_audio_features, songs_search_res['popularity']], axis=1)
            
            # scale audio features
            # minmax scaling used instead of standardscaler to make values between 0 nd 1
            scaled_search_res_numeric_features = self.normalize_audio_features(MinMaxScaler, search_res_numeric_features)
            # scaled_search_res_numeric_features = self.normalize_audio_features(StandardScaler, search_res_numeric_features)

            scaled_numeric_features =  np.array(scaled_search_res_numeric_features)
            #perform tf_idf transformation on the DF with songs from search results for each artist
            if tf_idf:
                tfidf = TfidfVectorizer()
                tfidf_matrix =  tfidf.fit_transform(songs_search_res.genres.apply(lambda x: " ".join(x)))
                genre_df = pd.DataFrame(tfidf_matrix.toarray())
                genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
                genre_df.reset_index(drop = True, inplace=True)
                # print(pd.concat([scaled_search_res_numeric_features, genre_df], axis=1).columns)
                scaled_numeric_features = pd.concat([scaled_search_res_numeric_features, genre_df], axis=1)

            if pca_components > 0:
                #pca transform the data
                pca = decomposition.PCA(svd_solver='randomized')
                scaled_numeric_features = pca.fit_transform(scaled_numeric_features)
                scaled_numeric_features = scaled_numeric_features[:, :pca_components]

            # find overlapping columns in comparison_values & scaled_numeric_features
            # assume that scaled_numeric_features will always have smaller number of features (because its just 1 song)
            overlap = [column for column in scaled_numeric_features.columns if column in comparison_values.index]
                
            # KNN search for k most similar songs from search results
            idxs, sims = self.knn_search(comparison_values.loc[overlap,:], scaled_numeric_features.loc[:,overlap], k, pairwise.cosine_similarity)
            # add all songs to recommended songs DF
            tmp = pd.DataFrame({'artist': key,
                'track' : songs_search_res.iloc[idxs,:].track,
                'track_id' : songs_search_res.iloc[idxs,:].track_id,
                'similarity' : [sims[x] for x in idxs]}, index=None)
            recommended_songs = pd.concat([recommended_songs, tmp], axis=0, ignore_index=True)

        return recommended_songs

    def knn_search(self, instance, data, K, measure):
        """ Find K nearest neighbors of an instance x among the instances in D 
            -------------------------------------------
            parameters:
            - instance: vector used to compare data against
            - data: vector to search through for K neighbors
            - K: number of neighbors to find
            - measure: simlarity measure
            -------------------------------------------
            returns:
            - idx[:K]: the index of the first K most similar records to isntance in data
            - dists: similarities of rows in data to instance
        """
        data = np.array(data)
        n_songs = data.shape[0]
        sims = []
        for j in range(0,n_songs) :
            try:
                sims.append(measure(np.array(instance).reshape(1, -1), np.array(data)[j,:].reshape(1, -1)))
            except Exception as e:
                # depending on measure() there can be NaN values generated
                # still not qute sure why/how they are generated
                print(e)
                sims.append([[-1]])
                
        sims = [x[0][0] for x in sims]
        idx = np.argsort(sims)[::-1] # sorting in descending order bc were dealing with similarity
        
        # return the indexes of K most similar neighbors
        return idx[:K], sims

    def pca_transform(self, data):
        ''' Returns a DatatFrame with PCA transformed data
            -------------------------------------------
            parameters:
            - data: DataFrame with ONLY numeric values
            -------------------------------------------
            returns:
            - pca: decomposition.PCA object which contains information (i.e. explained_variance_ratio)
            - pca_transformed_data: DataFrame with PCA transformed data
        '''

        pca = decomposition.PCA(svd_solver='randomized')
        pca_transformed_data = pd.DataFrame(pca.fit_transform(data))

        return pca, pca_transformed_data

    def tf_idf_transform(self):
        ''' Returns a DatatFrame with TF-IDF transformed data
            -------------------------------------------
            parameters:
            - data: 
            -------------------------------------------
            returns:
            - tf_idf: TF-IDF transformed data
        '''

        # tf-idf transform genre data
        tfidf = TfidfVectorizer()
        tfidf_matrix =  tfidf.fit_transform(self.df_playlist.genres.apply(lambda x: " ".join(x)))
        genre_df = pd.DataFrame(tfidf_matrix.toarray())
        genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
        genre_df.reset_index(drop = True, inplace=True)

        num_features_scaled = self.normalized_numeric_features.copy()
        num_features_scaled = pd.concat([num_features_scaled, genre_df], axis=1)

        self.normalized_numeric_features = num_features_scaled



