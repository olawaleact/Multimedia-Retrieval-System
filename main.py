import streamlit as st
from time import sleep

from recommender import RecommenderSystem

global id_

@st.cache_data
def load_recommender():
    rs = RecommenderSystem()
    return rs

with st.spinner('Loading...'):

    rs = load_recommender()

    try:
        id_
    except NameError:
        # id_, url = rs.random_url()
        id_ = rs.retrieve_id()
        url = rs.get_song_url(id_)
    else:
        pass

st.write("# Recommender System", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.write('### Search for a song', unsafe_allow_html=True)

    col11, col12 = st.columns(2)
    with col11:
        artist_name_query = st.text_input("Artiste's name")

    with col12:
        song_name_query = st.text_input("Song name")
    
    # search_text = st.text_input("Enter artist name, song name separated by comma.")

    if st.button('Search'):
        # if ',' not in search_text:
        #     st.error('Invalid format, use a comma to separate artist and song name.')
        #     sleep(10000000)

        # artist_name_query, song_name_query = search_text.split(',')
        
        artist_name_query = artist_name_query.strip()
        song_name_query = song_name_query.strip()

        id_ = rs.get_song_id_by_name_artist(song_name_query,
                                            artist_name_query)
        if id_ == None:
            st.error('Song not found.')
            sleep(10000000)
        else:
            rs.cache_id(id_)
            # st.rerun()

    artist_name, song_name, album_name = rs.get_song_info(id_)

    st.video(url)
    st.write(f"**{song_name}** by **{artist_name}**, from {album_name} Album",
            unsafe_allow_html=True)

with col2:
    st.write('### More like this')
    algorithm = st.selectbox('Recommendation algorithm.',
                        ['random', 
                            'jaccard', 
                            'tf_idf', 
                            'word2vec',
                            'bert',
                            'mfcc_bow',
                            'id_ivec256_mmsr', 
                            'id_blf_spectral_mmsr', 
                            'id_musicnn_mmsr',
                            'id_incp_mmsr',
                            'id_resnet_mmsr',
                            'id_vgg19_mmsr',
                            'early_fusion',
                            'late_fusion'])
    
    with st.spinner('Looking for recommendations...'):
        similar_songs = rs.find_all_similar_songs(id_, 
                                                algorithm=algorithm,
                                                top_n=10)
        similar_songs_ids = similar_songs['id']

    for ssid in similar_songs_ids:
        st.write('---')
        artist_n, song_n, album_n = rs.get_song_info(ssid)
        song_url = rs.get_song_url(ssid)

        st.video(song_url)

        st.write(f"**{song_n}** by **{artist_n}**, from {album_n} Album",
            unsafe_allow_html=True)