import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import spotipy

class Playlist():
    '''Class to represent a Spotify Playlist
    -------------------------------------------
    Parameters:
        - link: HTTP link to the playlist
        - spotify_object: A spotify Object
    -------------------------------------------
    Attributes:
        - link: HTTP link to the playlist
        - audio_features: DataFrame containing audio features of songs in the playlist
        - playlist_df: DataFrame containing all features of songs in the playlist
        - sp: A Spotify object to interact with the Spotify API'''

    def __init__(self, link, spotify_object) -> None:
        self.link = link
        self.sp = spotify_object
        playlist = self.sp.playlist(self.link)

        df_playlist = self.build_playlist_df(playlist)
        self.audio_features = self.get_audio_features(df_playlist)
        self.normalized_audio_features = self.normalize_audio_features(MinMaxScaler, self.audio_features)
        # self.normalized_audio_features = self.normalize_audio_features(StandardScaler)
        self.df_playlist = pd.concat([df_playlist, self.audio_features], axis=1)
        

    def build_playlist_df(self, playlist):
        '''Create a DataFrame with track information about songs that are on the playlist
           -------------------------------------------
           parameters:
           - playlist: the playlist for which to build a DataFrame
           -------------------------------------------
           returns:
           - playlist_df'''

        playlist = playlist["tracks"]["items"]
        playlist_df = pd.DataFrame(columns=['song_title', 'artist','artist_id' ,'album', 'track_id', 'popularity'], index=None)
        track_position = 1

        for song in playlist:
            artists = []
            track = song["track"]
            name = track["name"]
            artist_list = track["artists"]

            # some songs have more than 1 artist
            # create a list of all artists on the song
            for a in range(len(artist_list)):
                artists.append((artist_list[a]['name'], artist_list[a]['id']))
                
            album = track["album"]["name"]
            track_id = track["id"]
            artist_list = [x[0] for x in artists]
            artist_id_list = [x[1] for x in artists]
            
            # track popularity
            track_pop = track["popularity"]

            # if track_id == None that means that the audio features for this song could not be found
            # the track is omitted from analysis & use when creating a new playlist
            if track_id != None:
                playlist_df.loc[track_position] = {'song_title' : name,
                                            'artist' : artist_list,
                                            'artist_id' : artist_id_list,
                                            'album' : album, 
                                            'track_id' : track_id,
                                                'popularity':track_pop}
                
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
           - audio_features_df.loc[: , playlist_audio_features]: DF containing only the audio features of a playlist'''

        # columns for audio features
        playlist_audio_features = ["danceability", "energy", "key",
            "loudness", "mode", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]

        # get track audio features using Spotify API
        track_list = playlist.track_id.ravel()
        audio_features_df = pd.DataFrame()
        track_audio_features = self.sp.audio_analysis(track_list[0])['track']

        #drop songs that arent found in spotify
        track_list = [x for x in track_list if (x != None)]
        track_audio_features = self.sp.audio_features(track_list)

        ind = 1
        for audio_features in track_audio_features:
            song_features = pd.DataFrame(audio_features, index=[ind])
            audio_features_df = pd.concat([audio_features_df,song_features], axis=0)
            ind+=1
        
        audio_features_df = audio_features_df.reset_index()

        return audio_features_df.loc[: , playlist_audio_features]

    def get_artist_counts(self):
        '''Get occurence of artists in a playlist
           -------------------------------------------
           -------------------------------------------
           returns:
           - artist_counts: occurrences of artists in playlist (song features included)
           - artist_id_counts: occurrences of artist_ids in playlist (song features included)
        '''
        artist_counts_in_playlist = {} #count the occurrences of each artist in the playlist (including artist features)
        artist_id_counts_in_playlist = {}

        artists = self.df_playlist['artist']
        artist_ids = self.df_playlist['artist_id']

        for i in artists.index:
            
            track_artists_lst = artists[i]
            track_artist_ids_lst = artist_ids[i]
            
            for a in track_artists_lst:
                if a not in artist_counts_in_playlist.keys():
                    artist_counts_in_playlist[a] = 1
                else:
                    artist_counts_in_playlist[a] += 1
                    
            for a_id in track_artist_ids_lst:
                if a_id not in artist_id_counts_in_playlist.keys():
                    artist_id_counts_in_playlist[a_id] = 1
                else:
                    artist_id_counts_in_playlist[a_id] += 1

        artist_counts = pd.DataFrame(artist_counts_in_playlist, index=range(0,1))
        artist_id_counts = pd.DataFrame(artist_id_counts_in_playlist, index=range(0,1))

        return artist_counts, artist_id_counts

    def normalize_audio_features(self, scaler, data):
        '''Normalize audio features using scaler
           -------------------------------------------
           parameters:
           - scaler: scaling method to use
           - data: data to scale
           -------------------------------------------
           returns:
           scaled_audio_features: data normalized using scaler'''

        scaler = scaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data), index=None)

        return scaled_data


