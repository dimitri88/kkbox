import pandas as pd
import numpy as np
import datetime
'''
Read train.csv file and convert it into a feature vector
'''

def getFeatureVecotrs(path,fileName):
    train = pd.read_csv(path + fileName, dtype={'msno': 'category',
                                                'source_system_tab': 'category',
                                                'source_screen_name': 'category',
                                                'source_type': 'category',
                                                'target': np.uint8,
                                                'song_id': 'category'})

    song_col = pd.read_csv(path + 'songs.csv',   dtype={'genre_ids': 'category',
                                                        'language': 'category',
                                                        'artist_name': 'category',
                                                        'composer': 'category',
                                                        'lyricist': 'category',
                                                        'song_id': 'category'})
    members = pd.read_csv(path + 'members.csv', dtype={'city': 'category',
                                                       'bd': np.uint8,
                                                       'gender': 'category',
                                                       'registered_via': 'category'})
    print('...Data preprocessing...')
    train_category_data = train[['source_system_tab', 'source_type']]
    train_category_data = pd.get_dummies(train_category_data)
    train = pd.concat([train[['msno', 'song_id']], train_category_data], axis=1)
    '''
    Read members.csv file and convert it into a feature vector
    'db' has 57% percentage of 0
    'gender' has 53% percentage null value
    '''

    members['registration_init_time'] = pd.to_datetime(members['registration_init_time'], format='%Y%m%d')
    members['expiration_date'] = pd.to_datetime(members['expiration_date'], format='%Y%m%d')
    members['length'] = members['expiration_date'] - members['registration_init_time']
    # convert timedelta object back to int
    members['length'] = (members['length'] / np.timedelta64(1, 'D')).astype(int)
    members['established'] = ((datetime.datetime.now() - members['registration_init_time']) / np.timedelta64(1, 'D')).astype(int)
    members['established'] = members['established'] // 365
    # get rid of skew data
    members.loc[members.length < 0, 'length'] = 0
    norm_range = members['length'].max() - members['length'].min()
    # normalize length to [0,1]
    #members['length'] = (members['length'] - members['length'].min()) / norm_range
    mean_age = members['bd'].mean()
    members.loc[members.bd == 0, 'bd'] = mean_age
    members = members[['msno', 'city', 'registered_via', 'length', 'bd']]
    #members = members.drop(['registration_init_time', 'bd', 'gender'], axis=1)
    #song_id_mat = song_col['song_id']

    len_mat = song_col['song_length']
    len_mode = (len_mat / 60000).astype(int)
    len_norm = len_mode

    #genre_id_mat.replace(np.inf, genre_id_mat.mode())
    #genre_id_mat.fillna(genre_id_mat.mode())
    # Assign the genre of the music with multiple genres to be its 1st one
    song_col['genre_ids'] = song_col['genre_ids'].apply(lambda genre: str(genre).split('|')[0])
    song_col['language'] = song_col['language'].apply(lambda language: 1 if float(language) == 52.0 or float(language) == -1.0 else 0).astype(np.int8)
    #Linzuo: commented genre_id
    #Steven: commented back after revising genre_id
    #songs_all_frame = pd.concat([song_id_mat, len_norm, genre_id_norm, song_cat_mat], axis=1)

    song_col = song_col[['song_id', 'genre_ids', 'language']]

    #songs_all_frame.to_csv('test.csv', index=False)
    train = pd.merge(train, members, on='msno', how='left')
   # train = pd.merge(train, song_col, on='song_id', how='left')
    # Replicate the rows with multiple genre ids to be multiple rows, each with a single genre id
    # train = train.set_index(train.columns.drop('genre_ids', 1).tolist()) \
    #                .genre_ids.str.split('|', expand=True).stack().reset_index() \
    #                .rename(columns={0: 'genre_ids'}).loc[:, train.columns]

    train = train.apply(lambda x: x.fillna(x.value_counts().index[0]))
    return train