import mod1
import playlist 
import pandas as pd

if __name__ == "__main__":
    playlist_link = mod1.get_user_playlist()
    p = playlist.Playlist(playlist_link, mod1.sp)
    p.tf_idf_transform()
    artist_counts, artist_id_counts = p.get_artist_counts2() 
    summarized_feature_vector = pd.DataFrame(p.normalized_numeric_features.loc[:,0:].mean(axis=0))
    artists_search = mod1.artist_search_results(artist_counts.index, artist_id_counts.index)
    recs = p.get_recommendations2(artists_search, summarized_feature_vector, tf_idf=True, k=5)
    mod1.create_new_playlist(recs)

