import pandas as pd
from sklearn import linear_model, tree, svm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from features import getFeatureVecotrs


def main():
    print('##############Create feature vectors based on training data##############')
    #cross validation to findout best machine learning algorithm
    train_data = pd.read_csv('../data/train.csv')
    x = getFeatureVecotrs('../data/', 'train.csv');
    y = train_data['target'].as_matrix()
    kf = KFold(n_splits=5)

    f1_linear_reg = [];
    f1_svm = [];
    f1_sgd = [];
    print('##############Starting cross validation##############')
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('##############Training with Linear Regression##############')
        reg = linear_model.LinearRegression()
        reg.fit(x_train, y_train)
        y_res = reg.predict(x_test)
        f1_linear_reg.append(computeF1(y_test, y_res, 'Linear Regression'))
        print('##############Training with Support Vector Machine##############')
        svm_clf = svm.SVC()
        svm_clf.fit(x_train, y_train)
        y_res = svm_clf.predict(x_test)
        f1_svm.append(computeF1(y_test, y_res, 'Support Vector Machine'))
        print('##############Training with SGD ##############')
        sgd_clf = linear_model.SGDClassifier(loss="hinge", penalty="l2")
        sgd_clf.fit(x_train, y_train)
        y_res = svm_clf.predict(x_test)
        f1_sgd.append(computeF1(y_test, y_res, 'Stochastic Gradient Descent'))


def computeF1(y_test, y_res, model_name):
    print('The f1 score for' + model_name)
    f1 = f1_score(y_test, y_res)
    print('Average f1: {0:0.2f}'.format(f1))
    return f1

if __name__ == "__main__": main()





