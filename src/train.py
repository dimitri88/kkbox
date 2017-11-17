import pandas as pd
from sklearn import linear_model,tree
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from features import getFeatureVecotrs


def main():
    print('##############Create feature vectors based on training data##############')
    #cross validation to findout best machine learning algorithm
    train_data = getFeatureVecotrs('../data/', 'train.csv')
    y = pd.read_csv('../data/train.csv')['target'].as_matrix()
    x = train_data.drop(['msno', 'song_id'], axis=1)
    x = x.as_matrix()
    kf = KFold(n_splits=4)

    f1_linear_reg = []
    f1_rf = []
    f1_sgd = []
    f1_tree = []
    print('##############Starting cross validation##############')
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]


        # print('##############Training with Logistic Regression##############')
        # reg = linear_model.LogisticRegression()
        # reg.fit(x_train, y_train)
        # y_res = reg.predict_proba(x_test)
        # f1_linear_reg.append(aucScore(y_test, y_res, 'Logistic Regression'))
        # print('##############Training with Random Forest##############')
        # rf_clf = RandomForestClassifier(max_depth=4, random_state=0)
        # rf_clf.fit(x_train, y_train)
        # y_res = rf_clf.predict_proba(x_test)
        # f1_rf.append(aucScore(y_test, y_res, 'Random Forest'))

        print('##############Training with Decision Tree ##############')
        tree_clf = tree.DecisionTreeClassifier()
        tree_clf.fit(x_train, y_train)
        y_res = tree_clf.predict_proba(x_test)
        f1_tree.append(aucScore(y_test, y_res, 'Decision Tree'))

        # print('##############Training with SGD ##############')
        # sgd_clf = linear_model.SGDClassifier(loss="log", penalty="l2")
        # sgd_clf.fit(x_train, y_train)
        # y_res = sgd_clf.predict_proba(x_test)
        # f1_sgd.append(aucScore(y_test, y_res, 'Stochastic Gradient Descent'))

   # avg_linear = sum(f1_linear_reg)/4.0
   # avg_rf = sum(f1_rf)/4.0
    #avg_sgd = sum(f1_sgd)/4.0
    avg_tree = sum(f1_tree)/4.0
   # print('The auc score for linear regression is : %f' % avg_linear)
    #print('The auc score for random forest is : %f' % avg_rf)
    #print('The auc score for sgd is : %f' % avg_sgd)
    print('The auc score for decision tree is : %f' % avg_tree)

def aucScore(y_test, y_res, model_name):
    print('The auc score for' + model_name)
    roc_acu = roc_auc_score(y_test, y_res[:, 1])
    print('Average auc score: {0:0.2f}'.format(roc_acu))
    return roc_acu


if __name__ == "__main__": main()





