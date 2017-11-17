import pandas as pd
from sklearn import linear_model,tree
from sklearn.ensemble import RandomForestClassifier
from features import getFeatureVecotrs

print('##############Create feature vectors based on training data##############')
# cross validation to findout best machine learning algorithm
train_data = getFeatureVecotrs('../data/', 'train.csv')
y = pd.read_csv('../data/train.csv')['target'].as_matrix()
x = train_data.drop(['msno', 'song_id'], axis=1)
x = x.as_matrix()
#clf = linear_model.SGDClassifier(loss="log", penalty="l2")
#rf_clf = Ran
clf = tree.DecisionTreeClassifier()
print('##############Create training data##############')
clf.fit(x, y)
test_data = getFeatureVecotrs('../data/', 'test.csv')
test_data = test_data.drop(['msno', 'song_id'], axis=1)
print('##############Create training data##############')
#res = clf.predict_proba(test_data)
res = clf.predict_proba(test_data)
output = pd.read_csv('../data/sample_submission.csv')
output['target'] = res[:, 1]
print('##############Writing output##############')
output.to_csv('out.csv', index=False)


