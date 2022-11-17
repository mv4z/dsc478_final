#IMPORTS
import pandas as pd
import re
import spotipy
import playlist
import config
import time
from sklearn.preprocessing import MinMaxScaler
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
    
    playlist_url = input('Enter a link to a Spotify playlist:')
        
    return playlist_url

def artist_search_results(artist_list, artist_id_list):
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
    num_artists = len(artist_list)

    for i in range(0, num_artists):

        artist = artist_list[i]
        artist_id = artist_id_list[i]
        
        song_names = []
        song_ids = []
        song_popularity = []
        song_genres = []

        #search spotify for artist name
        result = sp.search(artist, type=['track'], limit = 50)

        artist_genres = sp.artist(artist_id)["genres"]
        if len(artist_genres) == 0:
                artist_genres =['unkown']
        else:
            artist_genres = ",".join([re.sub(' ','_',i) for i in artist_genres])
            artist_genres = list(artist_genres.split(','))
        # artist_genres = [*set([item for sublist in artist_genres for item in sublist])]

        #add the song and its id to list of songs & list of song ids
        for song in result['tracks']['items']:
            song_names.append(song['name'])
            song_ids.append(song['id'])
            song_popularity.append(song['popularity'])
            song_genres.append(artist_genres)
        
            
        d = {'track':song_names,
            'track_id':song_ids,
            'popularity' : song_popularity,
            'genres' : song_genres}
        songs_search_res = pd.DataFrame(d, index=None)

        artist_search_res[artist] = songs_search_res

    return artist_search_res

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

def build_playlist(tracks, playlist_id, spotify):
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
    
def create_new_playlist(recs):
    '''Creates a new playlist by sorting recommendations in descending order by distance & choosing the first playlist_length songs
       -------------------------------------------
       paremeters:
       - recs: recommended songs dictionary
    '''
    
    # authorization
    SCOPE = "playlist-modify-public"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPE,
                                                   client_id=config.SPOTIFY_CLIENT_ID,
                                                   client_secret=config.SPOTIFY_SECRET,
                                                   redirect_uri=config.REDIRECT_URI)
                        )

    ask = True
    #make sure that desired length is actually an integer
    while ask==True:
        try:
            playlist_len = int(input('How long would you like your final playlist to be? (please enter an integer):'))
            ask = False
        except:
            print('Please be sure to use an INTEGER when specifying how long you woant your final playlist to be.')
    
    # sort recs based on similarity & grab as many as are wanted in the final playlist
    recs = recs.sort_values(by='similarity', ascending=False)
    rec_tracks_idx = recs.index[:playlist_len, ]
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
    build_playlist(recommendations, playlist_id, sp)
    
    # provide the new playlist URL
    playlist_url = sp.playlist(playlist_id)['external_urls']['spotify']
    print('DONE!')
    print(f'Here is the link to your new playlist: {playlist_url}')

    return playlist_url



# playlist_link = get_user_playlist()
# playlist_link = 'https://open.spotify.com/playlist/37i9dQZEVXcNDMTzEMNNE1', 10
# playlist1 = playlist.Playlist(playlist_link, sp)
# artist_counts, artist_id_counts = playlist1.get_artist_counts() 
# avg_audio_values = pd.DataFrame(playlist1.normalized_numeric_features.loc[:,0:].mean(axis=0))
# artists_search = artist_search_results(artist_counts.columns)
# recs = playlist1.get_recommendations(artists_search, avg_audio_values, k=5)
# create_new_playlist(recs)


        