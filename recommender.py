from ast import literal_eval
import random

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderSystem:

    def __init__(self, dataset_folder='dataset'):

        ### Load the datasets with the 'id' column as the index
        # TF-IDF data
        self.tfidf_data = pd.read_csv(f'{dataset_folder}/id_lyrics_tf-idf_mmsr.tsv', sep='\t', index_col='id')

        # BERT data
        self.bert_data = pd.read_csv(f'{dataset_folder}/id_lyrics_bert_mmsr.tsv', sep='\t', index_col='id')

        # Word2Vec data
        self.word2vec_data = pd.read_csv(f'{dataset_folder}/id_lyrics_word2vec_mmsr.tsv', sep='\t', index_col='id')

        # Info data
        self.info_data = pd.read_csv(f'{dataset_folder}/id_information_mmsr.tsv', sep='\t')

        # MFCC Bow data
        data_mfcc_bow = pd.read_csv(f'{dataset_folder}/id_mfcc_bow_mmsr.tsv', sep='\t')
        self.data_mfcc_bow = data_mfcc_bow.set_index('id')

        # ivec256
        self.id_ivec256_mmsr = pd.read_csv(f'{dataset_folder}/id_ivec256_mmsr.tsv', sep='\t')
        self.id_ivec256_mmsr.set_index(self.id_ivec256_mmsr.columns[0], inplace=True)

        # ivec512
        self.id_ivec512_mmsr = pd.read_csv(f'{dataset_folder}/id_ivec512_mmsr.tsv', sep='\t')

        # BLF correlation
        self.id_blf_correlation_mmsr = pd.read_csv(f'{dataset_folder}/id_blf_correlation_mmsr.tsv', sep='\t')

        # BLF spectral
        self.id_blf_spectral_mmsr = pd.read_csv(f'{dataset_folder}/id_blf_spectral_mmsr.tsv', sep='\t')
        self.id_blf_spectral_mmsr.set_index(self.id_blf_spectral_mmsr.columns[0], inplace=True)

        # MusicNN
        self.id_musicnn_mmsr = pd.read_csv(f'{dataset_folder}/id_musicnn_mmsr.tsv', sep='\t')
        self.id_musicnn_mmsr.set_index(self.id_musicnn_mmsr.columns[0], inplace=True)

        # INCP
        id_incp_mmsr = pd.read_csv(f'{dataset_folder}/id_incp_mmsr.tsv', sep='\t')
        self.id_incp_mmsr = id_incp_mmsr.set_index('id')

        # ResNet
        id_resnet_mmsr = pd.read_csv(f'{dataset_folder}/id_resnet_mmsr.tsv', sep='\t')
        self.id_resnet_mmsr = id_resnet_mmsr.set_index('id')

        # URL
        self.id_url_mmsr = pd.read_csv(f'{dataset_folder}/id_url_mmsr.tsv', sep='\t')

        # VGG19
        id_vgg19_mmsr = pd.read_csv(f'{dataset_folder}/id_vgg19_mmsr.tsv', sep='\t')
        self.id_vgg19_mmsr = id_vgg19_mmsr.set_index('id')

        # ID genres
        self.data_id_genres = pd.read_csv(f'{dataset_folder}/id_genres_mmsr.tsv', sep='\t')

    
    # Random song id and url
    def random_url(self):
        id, url = tuple(self.id_url_mmsr.sample(1).values[0])
        return id, url
    
    # Get song url
    def get_song_url(self, id):
        url = self.id_url_mmsr[self.id_url_mmsr['id'] == id]['url'].values[0]
        return url
    
    # Get song info including artist, song and album name
    def get_song_info(self, id):
        row = self.info_data[self.info_data['id'] == id]
        artist = row['artist'].values[0]
        song = row['song'].values[0]
        album_name = row['album_name'].values[0]
        return artist, song, album_name
    
    # Get song id frm artist name and song name
    def get_song_id_by_name_artist(self, song_name, artist_name):
        matched_songs = self.info_data[
            (self.info_data['song'].str.lower().str.strip() == song_name.lower().strip()) &
            (self.info_data['artist'].str.lower().str.strip() == artist_name.lower().strip())
        ]
        if not matched_songs.empty:
            return matched_songs.iloc[0]['id']
        else:
            return None
        
    # Retrieve id_
    def retrieve_id(self):
        with open('id.txt', 'r') as f:
            id_ = f.read().strip()
        return id_
    
    # Cache id_
    def cache_id(self, id_):
        with open('id.txt', 'w') as f:
            f.write(id_)

    # Random Similarity
    def random_song(self, song_id, other_song_id, data):
        return random.random()

    # Jaccard Similarity
    def jaccard_similarity(self, id1, id2, tfidf_data):
        vec1 = tfidf_data.loc[id1].astype(bool).values
        vec2 = tfidf_data.loc[id2].astype(bool).values
        intersection = np.sum(vec1 & vec2)
        union = np.sum(vec1 | vec2)
        return intersection / union if union != 0 else 0

    # Cosine Similarity for Word2Vec and TF-IDF
    def cosine_similarity_between_songs(self, id1, id2, embedding_data):
        vec1 = embedding_data.loc[id1].values.reshape(1, -1)
        vec2 = embedding_data.loc[id2].values.reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]


    # Retrieval Function
    def retrieve_similar_songs(self, song_id, embedding_data, similarity_function, info_data, top_n=10):
        similarities = {}
        for other_song_id in embedding_data.index:
            if other_song_id != song_id:
                # Use the provided similarity function
                similarity = similarity_function(song_id, other_song_id, embedding_data)
                similarities[other_song_id] = similarity
        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_n]
        
        similar_songs_info = info_data[info_data['id'].isin([song_id for song_id, _ in sorted_similarities])]
        similar_songs_info = similar_songs_info.join(
            pd.DataFrame(sorted_similarities, columns=['id', 'similarity_score']).set_index('id'), 
            on='id'
        )
        
        return similar_songs_info
    
    # Genre hit func
    def genre_hit_func(song_id_genres: list, other_song_genres: str):
        for genre in song_id_genres:
            if genre in other_song_genres:
                return 1
        else:
            return 0
    
    # Finding all similar songs
    # def find_all_similar_songs(self,
    #                         song_id, 
    #                         info_data, 
    #                         word2vec_data, 
    #                         tfidf_data, 
    #                         data_mfcc_bow,
    #                         id_ivec256_mmsr, 
    #                         id_blf_spectral_mmsr, 
    #                         id_musicnn_mmsr,
    #                         id_incp_mmsr,
    #                         id_resnet_mmsr,
    #                         id_vgg19_mmsr,
    #                         data_id_genres=None,
    #                         output_genre=False,
    #                         output_genre_hit=False,
    #                         top_n=10):
        
    # Finding all similar songs
    def find_all_similar_songs(self, song_id, algorithm, top_n=10):
        
        # Random method
        if algorithm == 'random':
            random_songs = self.info_data.sample(n=top_n).assign(method='random').assign(similarity_score=0.5)
            random_songs = random_songs.sort_values('similarity_score', ascending=False)
            return random_songs

        # Jaccard similarity
        if algorithm == 'jaccard':
            jaccard_songs = self.retrieve_similar_songs(song_id, self.tfidf_data, self.jaccard_similarity, self.info_data, top_n).assign(method='jaccard')
            jaccard_songs = jaccard_songs.sort_values('similarity_score', ascending=False)
            return jaccard_songs
        
        # BERT similarity
        if algorithm == 'bert':
            bert_songs = self.retrieve_similar_songs(song_id, self.bert_data, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='bert')
            bert_songs = bert_songs.sort_values('similarity_score', ascending=False)
            return bert_songs

        # Cosine similarity with TF-IDF
        elif algorithm == 'tf_idf':
            tfidf_songs = self.retrieve_similar_songs(song_id, self.tfidf_data, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='tfidf')
            tfidf_songs = tfidf_songs.sort_values('similarity_score', ascending=False)
            return tfidf_songs

        # Cosine similarity with Word2Vec embeddings
        elif algorithm == 'word2vec':
            word2vec_songs = self.retrieve_similar_songs(song_id, self.word2vec_data, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='word2vec')
            word2vec_songs = word2vec_songs.sort_values('similarity_score', ascending=False)
            return word2vec_songs
        
        #Cosine similarity with MFCC Bow 
        elif algorithm == 'mfcc_bow':
            mfcc_bow_songs = self.retrieve_similar_songs(song_id, self.data_mfcc_bow, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='mfcc_bow')
            mfcc_bow_songs = mfcc_bow_songs.sort_values('similarity_score', ascending=False)
            return mfcc_bow_songs
        
        #Cosine similarity with id_ivec256_mmsr  
        elif algorithm == 'id_ivec256_mmsr':
            id_ivec256_mmsr_songs = self.retrieve_similar_songs(song_id, self.id_ivec256_mmsr, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='id_ivec256_mmsr')
            id_ivec256_mmsr_songs = id_ivec256_mmsr_songs.sort_values('similarity_score', ascending=False)
            return id_ivec256_mmsr_songs

        #Cosine similarity with id_blf_spectral_mmsr
        elif algorithm == 'id_blf_spectral_mmsr':
            id_blf_spectral_mmsr_songs = self.retrieve_similar_songs(song_id, self.id_blf_spectral_mmsr, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='id_blf_spectral_mmsr')
            id_blf_spectral_mmsr_songs = id_blf_spectral_mmsr_songs.sort_values('similarity_score', ascending=False)
            return id_blf_spectral_mmsr_songs
        
        #Cosine similarity with id_musicnn_mmsr
        elif algorithm == 'id_musicnn_mmsr':
            id_musicnn_mmsr_songs = self.retrieve_similar_songs(song_id, self.id_musicnn_mmsr, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='id_musicnn_mmsr')
            id_musicnn_mmsr_songs = id_musicnn_mmsr_songs.sort_values('similarity_score', ascending=False)
            return id_musicnn_mmsr_songs

        ### Video based
        #Cosine similarity with id_incp_mmsr
        elif algorithm == 'id_incp_mmsr':
            id_incp_mmsr_songs = self.retrieve_similar_songs(song_id, self.id_incp_mmsr, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='id_incp_mmsr')
            id_incp_mmsr_songs = id_incp_mmsr_songs.sort_values('similarity_score', ascending=False)
            return id_incp_mmsr_songs

        #Cosine similarity with id_resnet_mmsr
        elif algorithm == 'id_resnet_mmsr':
            id_resnet_mmsr_songs = self.retrieve_similar_songs(song_id, self.id_resnet_mmsr, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='id_resnet_mmsr')
            id_resnet_mmsr_songs = id_resnet_mmsr_songs.sort_values('similarity_score', ascending=False)
            return id_resnet_mmsr_songs

        #Cosine similarity with id_vgg19_mmsr
        elif algorithm == 'id_vgg19_mmsr':
            id_vgg19_mmsr_songs = self.retrieve_similar_songs(song_id, self.id_vgg19_mmsr, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='id_vgg19_mmsr')
            id_vgg19_mmsr_songs = id_vgg19_mmsr_songs.sort_values('similarity_score', ascending=False)
            return id_vgg19_mmsr_songs

        ### Early fusion: Merge data before you search
        elif algorithm == 'early_fusion':
            early_fusion_data = pd.concat([self.tfidf_data, self.word2vec_data], axis=1)
            early_fusion_index = early_fusion_data.index
            early_fusion_pca = PCA(n_components=300)
            early_fusion_data = early_fusion_pca.fit_transform(early_fusion_data)
            early_fusion_data = pd.DataFrame(early_fusion_data, index=early_fusion_index)

            early_fusion_mmsr_songs = self.retrieve_similar_songs(song_id, early_fusion_data, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='early_fusion')
            early_fusion_mmsr_songs = early_fusion_mmsr_songs.sort_values('similarity_score', ascending=False)
            return early_fusion_mmsr_songs

        ### Late fusion: Search individually first before you merge the results
        elif algorithm == 'late_fusion':
            tfidf_songs = self.retrieve_similar_songs(song_id, self.tfidf_data, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='tfidf')
            tfidf_songs = tfidf_songs.sort_values('similarity_score', ascending=False)

            word2vec_songs = self.retrieve_similar_songs(song_id, self.word2vec_data, self.cosine_similarity_between_songs, self.info_data, top_n).assign(method='word2vec')
            word2vec_songs = word2vec_songs.sort_values('similarity_score', ascending=False)

            late_fusion_scores = (tfidf_songs['similarity_score'] + word2vec_songs['similarity_score']) / 2
            late_fusion_data = tfidf_songs.drop(columns=['similarity_score'])
            late_fusion_mmsr_songs = pd.merge(late_fusion_data, late_fusion_scores, left_index=True, right_index=True)
            late_fusion_mmsr_songs['method'] = 'late_fusion'
            late_fusion_mmsr_songs = late_fusion_mmsr_songs.sort_values('similarity_score', ascending=False)
            return late_fusion_mmsr_songs