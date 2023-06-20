import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import config
import classifiers
import results
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import functools
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

###base version without cross-validation, bagging, or data writing###

def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

def orconjunction(*conditions):
    return functools.reduce(np.logical_or, conditions)

def checkNegatives():
    #load data
    dataset = pd.read_csv(config.DATAFILE)

    #filter dates
    #time between blood test and results - 0 to 2 days before test
    c_1 = dataset.delta_days < 0
    c_2 = dataset.delta_days > -2
    dataset = dataset[conjunction(c_1, c_2)]

    #Group 1: culture positive sepsis - 1,sepsis
    #Group 2: culture negative - 0, no sepsis
    #Group 3: culture negative, clinically positive - 1, sepsis
    #add group 3 to 1
    dataset = dataset.replace({'SEPSIS_GROUP': {2: 0}})
    dataset = dataset.replace({'SEPSIS_GROUP': {3: 1}})
    c_1 = dataset.SEPSIS_GROUP == 0
    c_2 = dataset.SEPSIS_GROUP == 1
    dataset = dataset[orconjunction(c_1, c_2)]

    #dataset = dataset[selectedColumns]
    length = len(dataset.columns) - 3

    #remove nulls
    dataset = dataset.dropna()

    dataset.to_csv('result.csv')

    X = dataset.iloc[:, 0:length]
    y = dataset.iloc[:, length]
    #split data into sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TESTSIZE, random_state=config.SEED)

    #normalize data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


if __name__ == '__main__':
    #load data
    dataset = pd.read_csv(config.DATAFILE)

    #filter dates
    c_1 = dataset.delta_days < 0
    c_2 = dataset.delta_days > -2
    dataset = dataset[conjunction(c_1, c_2)]

    #add group 3 to 1
    dataset = dataset.replace({'SEPSIS_GROUP': {2: 0}})
    dataset = dataset.replace({'SEPSIS_GROUP': {3: 1}})
    c_1 = dataset.SEPSIS_GROUP == 0
    c_2 = dataset.SEPSIS_GROUP == 1
    dataset = dataset[orconjunction(c_1, c_2)]

    # selected features to use
    #selectedColumns = ['NE_SFL', 'NE_FSC', 'MO_WX', 'MO_Y', 'LY_X', 'MPV', 'LYMP_per_amper', 'MicroR', 'LYMPH_per',
    #                   'MCV', 'NE_WY', 'LYMP_num_amper', 'LYMPH_num', 'MO_X', 'LY_WY', 'LY_WZ', 'HCT', 'MO_WY', 'LY_Y',
    #                   'NEUT_per', 'NEUT_per_amper', 'HGB', 'MCH', 'MacroR', 'PCT', 'EO_num', 'PLT_I', 'PLT', 'RBC',
    #                   'EO_per', 'LY_Z', 'IG_per', 'RDW_CV', 'BA_N_per', 'MO_WZ', 'BASO_per', 'HFLC_per', 'IG_num',
    #                   'LY_WX', 'BASO_num', 'NEUT_num', 'NEUT_num_amper', 'MONO_num', 'BA_N_num', 'MONO_per',
    #                   'HFLC_num', 'SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SEPSIS_GROUP'
    #                   ]

    #dataset = dataset[selectedColumns]
    length = len(dataset.columns) - 2
    #remove nulls
    dataset = dataset.dropna()

    dataset.to_csv('result.csv')

    X = dataset.iloc[:, 0:length]
    y = dataset.iloc[:, length]
    #split data into sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TESTSIZE, random_state=config.SEED)

    #normalize data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #Logistic Regression
    logisticParameters = [{'clf__C': [0.001, 0.01, 0.1]}, {'clf__max_iter': [999999]},
                          {'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]
    classifier, bestParams, GSresults = classifiers.logistic(X_train, y_train, logisticParameters, config.CV_SCORING, SMOTE())
    #results.getResults(classifier, X_train, y_train, './results/logistic', bestParams, 'TRAIN', 0, GSresults)
    results.getResults(classifier, X_test, y_test, './results/logistic', bestParams, 'TEST', 0, GSresults)

    #SVM
    SVMParameters = [{'clf__C': [10, 100], 'clf__gamma': [0.1]}]
    classifier, bestParams, GSresults = classifiers.SVM(X_train, y_train, SVMParameters, config.CV_SCORING, SMOTE())
    #results.getResults(classifier, X_train, y_train, './results/SVM', bestParams, 'TRAIN', 0, GSresults)
    results.getResults(classifier, X_test, y_test, './results/SVM', bestParams, 'TEST', 0, GSresults)

    #Random forest
    randomForestParameters = [{'clf__n_estimators': [500, 750], 'clf__criterion': ['entropy']}]
    classifier, bestParams, GSresults = classifiers.randomForest(X_train, y_train, randomForestParameters, config.CV_SCORING, SMOTE())
    #results.getResults(classifier, X_train, y_train, './results/randomForest', bestParams, 'TRAIN', 0, GSresults)
    results.getResults(classifier, X_test, y_test, './results/randomForest', bestParams, 'TEST', 0, GSresults)

    #KNN - performance sucks unless you optimize for accuracy instead of roc_auc
    KNNParameters = [{'clf__n_neighbors': [1, 3, 5, 10], 'clf__p': [1, 2], 'clf__weights': ['uniform', 'distance']}]
    classifier, bestParams, GSresults = classifiers.KNN(X_train, y_train, KNNParameters, config.CV_SCORING, SMOTE())
    #results.getResults(classifier, X_train, y_train, './results/KNN', bestParams, 'TRAIN', 0, GSresults)
    results.getResults(classifier, X_test, y_test, './results/KNN', bestParams, 'TEST', 0, GSresults)

    #XGBoost
    model = XGBClassifier()
    xg = model.fit(X_train, y_train)
    XGParameters = {

                'clf__booster': ['gbtree'], 'clf__learning_rate': [0.1],
                 'clf__max_depth': [6], 'clf__min_child_weight': [11],
                'clf__subsample': [0.8], 'clf__colsample_bytree': [0.7],
                 'clf__gamma': [1], 'clf__n_estimators': [750]} #try other gamma values
    classifier, bestParams, GSresults = classifiers.XG(X_train, y_train, XGParameters, config.CV_SCORING, SMOTE())
    #results.getResults(classifier, X_train, y_train, './results/XGBoost', bestParams, 'TRAIN', 0, GSresults)
    results.getResults(classifier, X_test, y_test, './results/XGBoost', bestParams, 'TEST', 0, GSresults)

