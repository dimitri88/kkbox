import pandas as pd
from sklearn import tree
print('Reading files')
train = pd.read_csv('../train.csv')
label = train['target'].as_matrix()

# count how many times each songs appeard

songs_count = train[['song_id', 'target']].groupby(['song_id']).agg(['mean', 'count'])
songs_count.reset_index(inplace=True)
songs_count.columns = ['song_id', 'id_play_chance', 'plays']
songs_count['repeat_plays'] = songs_count['id_play_chance'] * songs_count['plays']
print('Merging tables')
train.merge(songs_count[['song_id', 'repeat_plays']], on='song_id')
print('Converting feature vectors')

feature_data = train[['source_system_tab', 'source_type']]

train_data = pd.get_dummies(feature_data).as_matrix();

print('Starting to train')
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, label)

print('Training finished')
print('Building Test vectors')

test = pd.read_csv('../test.csv')
test_feature = test[['source_system_tab', 'source_type']]
test_data = pd.get_dummies(test_feature).as_matrix();
print('Start predicting')
res = clf.predict_proba(test_data)
res_data = pd.read_csv('../sample_submission.csv')
res_data['target'] = res[:, 1]
res_data.to_csv('final.csv', index=False)







