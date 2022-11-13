#IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spotipy
import playlist
import config
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise
from spotipy.oauth2 import SpotifyOAuth

#create spotipy object to interact with spotify web API
from spotipy.oauth2 import SpotifyClientCredentials
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials
        (client_id=config.SPOTIFY_CLIENT_ID,
        client_secret=config.SPOTIFY_SECRET
        ))

def get_user_playlist():
    '''Ask the user for a link to a Spotify playlist and for the desired length of generated playlist
    -------------------------------------------
    parameters:
    -------------------------------------------
    returns:
    - playlist_url: link to user provided playlist
    - playlist_len: the desired length of the generated playlist'''
    
    ask = True
    playlist_len = 0

    playlist_url = input('Enter a link to a Spotify playlist:')
    
    #make sure that desired length is actually an integer
    while ask==True:
        try:
            playlist_len = int(input('How long would you like your final playlist to be? (please enter an integer):'))
            ask = False
        except:
            print('Please be sure to use an INTEGER when specifying how long you woant your final playlist to be.')
        
    return (playlist_url, playlist_len)


def artist_search_results(artist_list):
    '''Searches Spotify for songs by artists that are in artist_list
        -------------------------------------------
        parameters:
        - artist_list: a list of artist names that will be searched for on Spotify
        -------------------------------------------
        returns:
        - artist_search_res: a dictionary with
            - key: artist_name
            - value: DataFrame{'track': Top 50 songs in artist search results
                      'track_id': Top 50 song ids in artist search results}'''
                      
    artist_search_res = {}

    for artist in artist_list:
        
        song_names = []
        song_ids = []

        #search spotify for artist name
        result = sp.search(artist, type=['track'], limit = 50)

        #add the song and its id to list of songs & list of song ids
        for song in result['tracks']['items']:
            song_names.append(song['name'])
            song_ids.append(song['id'])
            
        d = {'track':song_names,
            'track_id':song_ids}
        songs_search_res = pd.DataFrame(d, index=None)

        artist_search_res[artist] = songs_search_res

    return artist_search_res

def knn_search(instance, data, K, measure):
    """ find K nearest neighbors of an instance x among the instances in D 
    -------------------------------------------
    parameters:
    - instance: vector used to compare data against
    - data: vector to search through for K neighbors
    - K: number of neighbors to find
    - measure: simlarity measure
    -------------------------------------------
    returns:
    - idx[:K]: the index of the first K most similar records to isntance in data
    - dists: similarities of rows in data to instance"""

    n_songs = data.shape[0]
    sims = []
    for j in range(0,n_songs) :
        try:
            sims.append(measure(np.array(instance).reshape(1, -1), np.array(data)[j,:].reshape(1, -1)))
        except Exception as e:
            # depending on measure() there can be NaN values generated
            # still not qute sure why/how they are generated
            # print(e)
            sims.append([[-1]])
            
    sims = [x[0][0] for x in sims]
    idx = np.argsort(sims)[::-1] # sorting in descending order bc were dealing with similarity
    
    # return the indexes of K most similar neighbors
    return idx[:K], sims

def get_recommendations(search_res, comparison_values, k=5):
    '''Get song song recommendations for a list of artists
       -------------------------------------------
       parameters:
       - search_res: a dictionary of artists & artist search result songs
       - comparison_values: values used for comparison in knn search
       - k: number of neighbors to find in KNN search. (default = 5)
       -------------------------------------------
       returns:
       - recommended_songs: DataFrame containing Artists, their k most similar songs (from search result) & accompanying info, and similarity to comparison_values'''

    recommended_songs = pd.DataFrame(index=None, columns=['artist', 'track', 'track_id', 'similarity'])

    # iterate over artists & their songs
    for key in search_res.keys():
        songs_search_res = search_res[key]
        
        # get audio features for all of the artists songs from search
        search_res_audio_features = playlist1.get_audio_features(songs_search_res)
        
        # scale audio features
        # minmax scaling used instead of standardscaler to make values between 0 nd 1
        scaled_search_res_audio_features = playlist1.normalize_audio_features(MinMaxScaler, search_res_audio_features)
        scaled_audio_features =  np.array(scaled_search_res_audio_features)
            
        # KNN search for k most similar songs from search results
        idxs, sims = knn_search(comparison_values, scaled_audio_features, k, pairwise.cosine_similarity)
        
        # add all songs to recommended songs DF
        tmp = pd.DataFrame({'artist': key,
            'track' : songs_search_res.iloc[idxs,:].track,
            'track_id' : songs_search_res.iloc[idxs,:].track_id,
            'similarity' : [sims[x] for x in idxs]}, index=None)
        recommended_songs = pd.concat([recommended_songs, tmp], axis=0, ignore_index=True)

    return recommended_songs

def get_playlist_id(playlist_name, spotify):
    '''Return playlist ID of playlist_name
       -------------------------------------------
       parameters:
       - playlist_name: name of playlist whos ID you want
       - spotify: Spotify object to use the API
       -------------------------------------------
       returns:
       - playlist['id']
    '''

    playlists = spotify.current_user_playlists()
    for playlist in playlists['items']:
        if playlist['name'] == playlist_name:
            print(playlist.keys())
            return playlist['id']

def build_playlist(tracks, playlist_id, final_len, spotify):
    '''Uses the spotify API to put songs onto a new playlist
       -------------------------------------------
       parameters: 
       - tracks: tracks which can be added to the playlist
       - playlist_id: playlist_id to add tracks to
       - final_len: length of playlist created,
       spotify: Spotify object to use the API'''
    
    #obtain user info
    user_id = sp.me()['id']

    #add tracks to playlist
    spotify.user_playlist_add_tracks(user_id, playlist_id, tracks.track_id)
    
def create_new_playlist(recs, playlist_length):
    '''Creates a new playlist by sorting recommendations in descending order by distance & choosing the first playlist_length songs
       -------------------------------------------
       paremeters:
       - recs: recommended songs dictionary
       - playlist_length: desired length of recommended playlist'''
    
    # authorization
    SCOPE = "playlist-modify-public"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPE,
                                                   client_id=config.SPOTIFY_CLIENT_ID,
                                                   client_secret=config.SPOTIFY_SECRET,
                                                   redirect_uri=config.REDIRECT_URI)
                        )
    
    # sort recs based on similarity & grab as many as are wanted in the final playlist
    recs = recs.sort_values(by='similarity', ascending=False)
    rec_tracks_idx = recs.index[:final_playlist_len, ]
    recommendations = recs.loc[rec_tracks_idx, :]

    # ask the user if they want to name the playlist & change its description
    playlist_name = input('Choose a name for your new playlist:')
    playlist_desc = input('Add a description for your new playlist (OPTIONAL):')
                          
    #name & description defaults
    name = playlist_name if (len(playlist_name) > 0) else f'New Playlist (DSC 478 APP)_{int(time.time())}'
    desc = playlist_desc if (len(playlist_desc) > 0) else f'Playlist generated using DSC 478 final project application (timestamp:{int(time.time())})'

    #create the playlist for the first time
    user_id = sp.me()['id']
    sp.user_playlist_create(user_id, name, public=True, description=desc)
        
    # build the playlist
    playlist_id = get_playlist_id(name, sp)
    build_playlist(recommendations, playlist_id, playlist_length, sp)
    
    # provide the new playlist URL
    playlist_url = sp.playlist(playlist_id)['external_urls']['spotify']
    print('DONE!')
    print(f'Here is the link to your new playlist: {playlist_url}')



playlist_link, final_playlist_len = get_user_playlist()
playlist1 = playlist.Playlist(playlist_link, sp)
artist_counts, artist_id_counts = playlist1.get_artist_counts() 
avg_audio_values = pd.DataFrame(playlist1.normalized_audio_features.mean(axis=0))
artists_search = artist_search_results(artist_counts.columns)
recs = get_recommendations(artists_search, avg_audio_values, k=5)
create_new_playlist(recs, final_playlist_len)


        