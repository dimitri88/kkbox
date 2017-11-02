import pandas as pd
import numpy as np
'''
Read train.csv file and convert it into a feature vector
'''

def getFeatureVecotrs(path,fileName):
    train_data = pd.read_csv(path + fileName)
    # extract categorical data and construct a one hot encoder using pandas dummy function
    train_category_data = train_data[['source_system_tab','source_screen_name','source_type']]
    train_category_data = pd.get_dummies(train_category_data)
    #combine IDs with one hot encoded data
    train_data = pd.concat([train_data[['msno','song_id']], train_category_data ], axis=1)
    # write train_data to csv file
    print('Writing train_feature_vector.csv ')
    train_data.to_csv('./train_feature_vector.csv', mode='w+')

    '''
    Read members.csv file and convert it into a feature vector
    'db' has 57% percentage of 0
    'gender' has 53% percentage null value
    '''

    # read members.csv
    member_data = train_data = pd.read_csv(path+'members.csv')
    # convert date date into pandas date format
    member_data['registration_init_time'] = pd.to_datetime(member_data['registration_init_time'], format='%Y%m%d')
    member_data['expiration_date'] = pd.to_datetime(member_data['expiration_date'], format='%Y%m%d')
    member_data['length'] = member_data['expiration_date'] - member_data['registration_init_time']
    # convert timedelta object back to int
    print('Changing length to int')
    member_data['length'] = (member_data['length'] / np.timedelta64(1, 'D')).astype(int)
    # get rid of skew data
    member_data.loc[member_data.length < 0, 'length'] = 0
    norm_range = member_data['length'].max() - member_data['length'].min()
    # normalize length to [0,1]
    print('Normalizing length')
    member_data['length'] = (member_data['length'] - member_data['length'].min()) / norm_range
    member_data[['city', 'registered_via']] = member_data[['city', 'registered_via']].applymap(str)
    member_category_data = pd.get_dummies(member_data[['city', 'registered_via']])

    member_data = pd.concat([member_data['msno'], member_category_data, member_data['length']], axis=1)
    print('Writing member.csv')
    member_data.to_csv('./member_feature_vector.csv', mode='w+')

    song_col = pd.read_csv('../data/songs.csv')
    song_id_mat = song_col['song_id']

    len_mat = song_col['song_length']
    len_mode = len_mat % 120000
    len_norm = (len_mode - len_mode.min()) / (len_mode.max() - len_mode.min())

    genre_id_mat = song_col['genre_ids']
    #genre_id_mat = genre_id_mat.applymap(int)
    genre_id_mat = pd.to_numeric(genre_id_mat, errors='coerce')
    genre_id_norm = (genre_id_mat - genre_id_mat.min()) / (genre_id_mat.max() - genre_id_mat.min())

    song_category = song_col[['language']]
    song_category = song_category.applymap(str)
    song_cat_mat = pd.get_dummies(song_category)

    songs_all_frame = pd.concat([song_id_mat, len_norm, genre_id_norm, song_cat_mat], axis=1)
    songs_all_frame.to_csv('test.csv', index=False)
