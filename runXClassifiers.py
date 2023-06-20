import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import config
import classifiers
import results
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import functools
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from mlxtend.classifier import EnsembleVoteClassifier
import time

from config import config
from xgboost import XGBClassifier
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

def orconjunction(*conditions):
    return functools.reduce(np.logical_or, conditions)

if __name__ == '__main__':
    #load data
    dataset = pd.read_csv(config.DATAFILE, dtype = {'SURGICAL':str, 'CARDIAC':str, 'CLD': str, 'IVHSHUNT':str, 'NEC':str})

    # #filter dates - definition of sepsis
    c_1 = dataset.delta_days < 1
    c_2 = dataset.delta_days > -2
    dataset = dataset[conjunction(c_1, c_2)]
    print(len(dataset))
    # #add group 3 to 1
    dataset = dataset.replace({'SEPSIS_GROUP': {2: 0}})
    dataset = dataset.replace({'SEPSIS_GROUP': {3: 1}})

    c_1 = dataset.SEPSIS_GROUP == 0
    c_2 = dataset.SEPSIS_GROUP == 1
    dataset = dataset[orconjunction(c_1, c_2)]

    print(len(dataset))

    # selected features to use - dependent on dataset

    #selectedColumns = ['NE_SFL', 'NE_FSC', 'MO_WX', 'MO_Y', 'LY_X', 'MPV', 'LYMP_per_amper', 'MicroR', 'LYMPH_per',
                       'MCV', 'NE_WY', 'LYMP_num_amper', 'LYMPH_num', 'MO_X', 'LY_WY', 'LY_WZ', 'HCT', 'MO_WY', 'LY_Y',
                       'NEUT_per', 'NEUT_per_amper', 'HGB', 'MCH', 'MacroR', 'PCT', 'EO_num', 'PLT_I', 'PLT', 'RBC',
                       'EO_per', 'LY_Z', 'IG_per', 'RDW_CV', 'BA_N_per', 'MO_WZ', 'BASO_per', 'HFLC_per', 'IG_num',
                       'LY_WX', 'BASO_num', 'NEUT_num', 'NEUT_num_amper', 'MONO_num', 'BA_N_num', 'MONO_per',
                       'HFLC_num', 'SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SEPSIS_GROUP']


    #use only clinical variables here for a clinical variable only model for comparison - adjust titles of files below as necessary
    selectedColumns_clinical_var = ['GESTATIONAL_AGE', 'AGE_DAYS_ONSET', 'SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SUBJECT_ID', 'SEPSIS_GROUP']

    dataset = dataset[selectedColumns_mostdata]
    dataset = dataset.drop(['delta_days'], axis=1)

    #remove nulls
    print(len(dataset))
    dataset = dataset.dropna()
    print(len(dataset))

    dataset = dataset.replace(to_replace='TRUE', value=1)
    dataset = dataset.replace(to_replace='FALSE', value=0)
    dataset.index = range(len(dataset))

    #removing collinear variables, generating final dataset
    def calculate_vif(X, thresh=10):
        variables = list(range(X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
                   for ix in range(X.iloc[:, variables].shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                      '\' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True
        print('Remaining variables:')
        print(X.columns[variables])
        return X.iloc[:, variables]

    patient_splitting_var = ['SUBJECT_ID', 'SEPSIS_GROUP']

    # # Vif/dataset formation for non-imputed variables
    Non_collinear_var = calculate_vif(dataset.iloc[:, :-2])
    final_data_set = Non_collinear_var.join(dataset[patient_splitting_var])
    unique_patients = dataset[patient_splitting_var].drop_duplicates(subset="SUBJECT_ID")
    #dataset.to_csv('result.csv')
    X = unique_patients.iloc[:, 0]
    y = unique_patients.iloc[:, 1]
    print(y.mean())
    print(len(unique_patients))

    #define specific sens and NPV metrics
    def specific(values, predictions):
        tn, fp, fn, tp = confusion_matrix(values, predictions).ravel()
        specificity = tn / (tn + fp)
        return specificity
    specificity = make_scorer(specific, greater_is_better=True)

    def npv(values, predictions):
        tn, fp, fn, tp = confusion_matrix(values, predictions).ravel()
        npv = tn / (tn + fn)
        return npv
    neg_pred_val = make_scorer(npv, greater_is_better=True)

    # set up the k-fold process
    skf = StratifiedKFold(n_splits=config.NUMBER_OF_FOLDS, random_state=config.SEED, shuffle=True)
    # get the folds and loop over each fold
    fold_number = 0

    headers = ['classifier', 'fold', 'pred_values', 'test_labels','y_pred', 'accuracy', 'roc', 'sensitivity', 'specificity', 'ppv', 'npv', 'test_set_size', 'percent_sepsis', 'f1_score', 'parameters']
    logistic_results = [headers]
    random_forest_results = [headers]
    xgboost_results = [headers]
    svm_results = [headers]
    knn_results = [headers]

    for train_index, test_index in skf.split(X, y):
        fold_number = fold_number + 1

        unique_train_patients = X.iloc[train_index].to_frame()
        unique_test_patients = X.iloc[test_index].to_frame()

        #drop patients with records that are shared between the train and test sets to avoid training and testing on the same set
        train_patient_data = final_data_set[final_data_set.SUBJECT_ID.isin(unique_train_patients.SUBJECT_ID)].drop(['SUBJECT_ID'], axis=1)
        test_patient_data = final_data_set[final_data_set.SUBJECT_ID.isin(unique_test_patients.SUBJECT_ID)].drop(['SUBJECT_ID'], axis=1)
        length = len(train_patient_data.columns) - 1

        X_train, X_test = train_patient_data.iloc[:, 0:length], test_patient_data.iloc[:, 0:length]
        y_train, y_test = train_patient_data.iloc[:, length], test_patient_data.iloc[:, length]

        # normalize data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        #Non-bagging classifiers

        # #Logistic Regression
        # logisticParameters = [{'clf__C': [0.001, 0.01, 0.1]}, {'clf__max_iter': [9999999]},
        #                       {'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]
        # classifier, bestParams, GSresults = classifiers.logistic(X_train, y_train, logisticParameters, config.CV_SCORING, SMOTE())
        # logistic_results.append(results.StoreResults(classifier, X_test, y_test, 'logistic', bestParams, 'TEST', fold_number, GSresults))
        #
        # #SVM
        # SVMParameters = [{'clf__C': [10, 100, 200, 500, 750], 'clf__gamma': [0, 0.1, 0.5, 1]}]
        # classifier, bestParams, GSresults = classifiers.SVM(X_train, y_train, SVMParameters, config.CV_SCORING, SMOTE())
        # svm_results.append(results.StoreResults(SVM_classifier, X_test, y_test, 'SVM', bestParams, 'TEST', fold_number, GSresults))
        #
        # # Random forest
        # randomForestParameters = [{'clf__n_estimators': [250,500, 750, 1000], 'clf__criterion': ['entropy']}]
        # classifier, bestParams, GSresults = classifiers.randomForest(X_train, y_train, randomForestParameters, config.CV_SCORING, SMOTE())
        # random_forest_results.append(results.StoreResults(classifier, X_train, y_train, 'RF', bestParams, 'TEST', fold_number, GSresults))
        #
        # #KNN - performance sucks unless you optimize for accuracy instead of roc_auc
        # KNNParameters = [{'clf__n_neighbors': [1, 3, 5, 10, 20], 'clf__p': [1, 2, 8], 'clf__weights': ['uniform', 'distance']}]
        # classifier, bestParams, GSresults = classifiers.KNN(X_train, y_train, KNNParameters, config.CV_SCORING, SMOTE())
        # KNN_classifier, bestParams, GSresults = classifiers.KNN(X_train, y_train, KNNParameters,config.CV_SCORING, SMOTE())
        # knn_results.append(results.StoreResults(KNN_classifier, X_test, y_test, 'KNN', bestParams, 'TEST', fold_number, GSresults))
        #
        # #XGBoost
        # model = XGBClassifier()
        # xg = model.fit(X_train, y_train)
        # XGParameters = {
        #              'clf__booster': ['gbtree'], 'clf__learning_rate': [0.1, 0.01],
        #               'clf__max_depth': [6], 'clf__min_child_weight': [11],
        #              'clf__subsample': [0.8], 'clf__colsample_bytree': [0.7],
        #               'clf__gamma': [0, 0.1, 0.5, 1, 5, 20], 'clf__n_estimators': [100, 250, 500, 750, 1000]}
        # classifier, bestParams, GSresults = classifiers.XG(X_train, y_train, XGParameters, config.CV_SCORING, SMOTE())
        # xgboost_results.append(results.StoreResults(XGB_classifier, X_test, y_test, 'xgb', bestParams, 'TEST', fold_number, GSresults))
        #
        # #Bagging Versions Below
        #
        # #Logistic Regression Bagging
        # LogisticParameter_Grid = {
        #         'bootstrap': [True],
        #         'bootstrap_features': [False],
        #         'n_estimators': [5],
        #         'max_features': [0.5, 0.8, 1.0],
        #         'max_samples': [0.5, 0.8, 1.0],
        #         #'max_samples': [0.25, 0.5, 0.8, 1.0],
        #         'random_state': [config.BAGGING_SEED],
        #         'base_estimator__clf__C': [0.001, 0.01, 0.1],
        #         'base_estimator__clf__max_iter': [999999],
        #         'base_estimator__clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        # }
        # log_classifier, bestParams, GSresults = classifiers.Bagging_logistic(X_train, y_train, LogisticParameter_Grid, config.CV_SCORING, SMOTE())
        # logistic_results.append(results.StoreResults(log_classifier, X_test, y_test, 'logistic', bestParams, 'TEST', fold_number, GSresults))
        # #
        # #
        # # #SVM_Bagging
        # SVMParameter_Grid = {
        #     'bootstrap': [True],
        #     'bootstrap_features': [False],
        #     'n_estimators': [5],
        #     'max_features': [0.5, 0.8, 1.0],
        #     'max_samples': [0.5, 0.8, 1.0],
        #     #'max_samples': [0.25, 0.5, 0.8, 1.0],
        #     'random_state': [config.BAGGING_SEED],
        #     'base_estimator__clf__C':  [10, 100, 200, 500],
        #     #0 was invalid?
        #     'base_estimator__clf__gamma': [0.1, 0.5, 1]
        #     }
        # SVM_classifier, bestParams, GSresults = classifiers.Bagging_SVM(X_train, y_train, SVMParameter_Grid, config.CV_SCORING, SMOTE())
        # svm_results.append(results.StoreResults(SVM_classifier, X_test, y_test, 'SVM', bestParams, 'TEST', fold_number, GSresults))
        # #
        # # Random forest bagging
        # RandomForestParameter_Grid = {
        #     'bootstrap': [True],
        #     'bootstrap_features': [False],
        #     'n_estimators': [5],
        #     'max_features': [0.5, 0.8, 1.0],
        #     'max_samples': [0.5, 0.8, 1.0],
        #     #'max_samples': [0.25, 0.5, 0.8, 1.0],
        #     'random_state': [config.BAGGING_SEED],
        #     'base_estimator__clf__n_estimators': [250, 500, 750, 1000],
        #     'base_estimator__clf__criterion': ['entropy']
        # }
        # RF_classifier, bestParams, GSresults = classifiers.Bagging_randomForest(X_train, y_train, RandomForestParameter_Grid, config.CV_SCORING, SMOTE())
        # random_forest_results.append(results.StoreResults(RF_classifier, X_test, y_test, 'RF', bestParams, 'TEST', fold_number, GSresults))
        #
        # # #KNN Bagging
        # KNNParameter_Grid = {
        #     'bootstrap': [True],
        #     'bootstrap_features': [False],
        #     'n_estimators': [5],
        #     'max_features': [0.5, 0.8, 1.0],
        #     'max_samples': [0.5, 0.8, 1.0],
        #     #'max_samples': [0.25, 0.5, 0.8, 1.0],
        #     'random_state': [config.BAGGING_SEED],
        #     'base_estimator__clf__n_neighbors': [1, 3, 5, 10],
        #     'base_estimator__clf__p': [1, 2, 8],
        #     'base_estimator__clf__weights': ['uniform', 'distance']
        # }
        # KNN_classifier, bestParams, GSresults = classifiers.Bagging_KNN(X_train, y_train, KNNParameter_Grid, config.CV_SCORING, SMOTE())
        # knn_results.append(results.StoreResults(KNN_classifier, X_test, y_test, 'KNN', bestParams, 'TEST', fold_number, GSresults))
        # #
        # #
        # #XGBoost
        # XBGParameter_Grid = {
        #     'bootstrap': [True],
        #     'bootstrap_features': [False],
        #     'n_estimators': [5],
        #     'max_features': [0.5, 0.8, 1.0],
        #     'max_samples': [0.5, 0.8, 1.0],
        #     #'max_samples': [0.25, 0.5, 0.8, 1.0],
        #     'random_state': [config.BAGGING_SEED],
        #     'base_estimator__clf__booster': ['gbtree'],
        #     'base_estimator__clf__learning_rate': [0.1, 0.01],
        #     'base_estimator__clf__max_depth': [6],
        #     'base_estimator__clf__min_child_weight': [11],
        #     'base_estimator__clf__subsample': [0.8],
        #     'base_estimator__clf__colsample_bytree': [0.7],
        #     'base_estimator__clf__gamma': [0, 0.1, 0.5, 1, 5, 15],
        #     'base_estimator__clf__n_estimators': [100, 250, 500, 750, 1000]
        # }
        # XGB_classifier, bestParams, GSresults = classifiers.Bagging_XG(X_train, y_train, XBGParameter_Grid, config.CV_SCORING, SMOTE())
        # xgboost_results.append(results.StoreResults(XGB_classifier, X_test, y_test, 'xgb', bestParams, 'TEST', fold_number, GSresults))
        # #
        #
        # results combining the different models - didn't use these results in the end
        #
        # Manual voting classifier
        # estimators = ['logistic','SVM','Random_forest','KNN','XBG']
        # estimators_list = ['SVM', 'Random_forest', 'XBG']
        # estimator_combination_list = classifiers.param_gen(estimators_list)
        # estimators = [log_classifier, RF_classifier]
        # labels = classifiers.results_grid(X_test, y_test, estimators)
        # print(labels)
        # print(estimators_list)
        #
        # Voting_Classifier
        # estimators = []
        # #estimators.append(('logistic', log_classifier))
        # estimators.append(('SVM', SVM_classifier))
        # estimators.append(('Random_forest', RF_classifier))
        # #estimators.append(('KNN', KNN_classifier))
        # estimators.append(('XGB', XGB_classifier))
        # base_estimators = estimators
        #
        # estimators = classifiers.param_gen(estimators)
        #
        # Voting_Grid = {
        #     'clfs': estimators,
        #     'voting': ['hard', 'soft'],
        # }
        # voting_clf = EnsembleVoteClassifier(clfs=base_estimators, refit=False)
        # voting_clf_GS = GridSearchCV(voting_clf, param_grid=Voting_Grid, scoring=config.CV_SCORING, verbose=config.VERBOSE)
        # voting_clf_GS.fit(X_test, y_test)
        # voting_clf_classifier, bestParams, GSresults = voting_clf_GS.best_estimator_, voting_clf_GS.best_params_, voting_clf_GS.cv_results_
        # results.getResults(voting_clf_classifier, X_train, y_train, './results/Voting_classifier', bestParams, 'TRAIN', fold_number, GSresults)
        # results.getResults(voting_clf_classifier, X_test, y_test, './results/Voting_classifier', bestParams, 'TEST', fold_number, GSresults)
        #
        # voting_clf = VotingClassifier(estimators, voting='hard', weights=None)
        # voting_clf = EnsembleVoteClassifier(clfs=base_estimators, voting='soft', refit=False)
        # voting_clf = voting_clf.fit(X_train, y_train)
        # results.getResults(voting_clf, X_train, y_train, './results/Voting_classifier', bestParams, 'TRAIN', fold_number, GSresults)
        # results.getResults(voting_clf, X_test, y_test, './results/Voting_classifier', bestParams, 'TEST', fold_number, GSresults)

        print("Fold {number} complete.".format(number=fold_number))

    df1 = pd.DataFrame(logistic_results).transpose()
    df1.to_csv("10_fold_feat_logistic.csv", index=False)
    df2 = pd.DataFrame(random_forest_results).transpose()
    df2.to_csv("10_fold_feat_random_forest.csv", index=False)
    df3 = pd.DataFrame(xgboost_results).transpose()
    df3.to_csv("10_fold_feat_xgboost.csv", index=False)
    df4 = pd.DataFrame(svm_results).transpose()
    df4.to_csv("10_fold_feat_svm.csv", index=False)
    df5 = pd.DataFrame(knn_results).transpose()
    df5.to_csv("10_fold_feat_knn.csv", index=False)

    # for multiple runs
    # timestr = time.strftime("%Y%m%d-%H%M%S")

    # df1 = pd.DataFrame(logistic_results).transpose()
    # df1.to_csv("feat_bootstrap_logistic"+timestr+".csv", index=False)
    # df2 = pd.DataFrame(random_forest_results).transpose()
    # df2.to_csv("feat_bootstrap_random_forest" + timestr + ".csv", index=False)
    # df3 = pd.DataFrame(xgboost_results).transpose()
    # df3.to_csv("feat_bootstrap_xgboost" + timestr + ".csv", index=False)
    # df4 = pd.DataFrame(svm_results).transpose()
    # df4.to_csv("feat_bootstrap_svm" + timestr + ".csv", index=False)
    # df5 = pd.DataFrame(knn_results).transpose()
    # df5.to_csv("feat_bootstrap_knn" + timestr + ".csv", index=False)
    # # #
