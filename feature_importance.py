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

#feature_importance Random forest code

def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

def orconjunction(*conditions):
    return functools.reduce(np.logical_or, conditions)

if __name__ == '__main__':
    #load data
    dataset = pd.read_csv(config.DATAFILE, dtype = {'SURGICAL':str, 'CARDIAC':str, 'CLD': str, 'IVHSHUNT':str, 'NEC':str})

    # #filter dates - dependent on time from sepsis diagnosis
    c_1 = dataset.delta_days < 1
    c_2 = dataset.delta_days > -2
    dataset = dataset[conjunction(c_1, c_2)]

    # #add group 3 to 1 - from sepsis definitions
    dataset = dataset.replace({'SEPSIS_GROUP': {2: 0}})
    dataset = dataset.replace({'SEPSIS_GROUP': {3: 1}})
    c_1 = dataset.SEPSIS_GROUP == 0
    c_2 = dataset.SEPSIS_GROUP == 1
    dataset = dataset[orconjunction(c_1, c_2)]

    # selected features to use - adjust based on dataset

    # selectedColumns_mostdata = ['BA_N_num', 'BA_N_per', 'HCT', 'HGB', 'MacroR', 'MCH', 'MCHC', 'MCV', 'MicroR', 'NRBC_num',
    #                             'NRBC_per','PLT', 'PLT_I','RBC', 'RDW_CV', 'RDW_SD', 'TNC', 'TNC_N', 'WBC', 'WBC_N',
    #                             'SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SUBJECT_ID', 'SEPSIS_GROUP']
    #
    # selectedColumns_extra_var_wgest = ['BA_D_num', 'BA_D_per', 'BASO_num', 'BASO_per', 'EO_per', 'EO_num', 'HFLC_num', 'IG_num', 'IG_per', 'LY_WX', 'LY_WY', 'LY_WZ', 'LY_X', 'LY_Y', 'LY_Z', 'LYMP_num_amper', 'LYMP_per_amper', 'LYMPH_num', 'LYMPH_per', 'MO_WX', 'MO_WY', 'MO_WZ', 'MO_X', 'MO_Y', 'MO_Z', 'MONO_num', 'MONO_per', 'NE_FSC', 'NE_SFL', 'NE_SSC', 'NE_WX', 'NE_WY', 'NE_WZ', 'NEUT_num', 'NEUT_num_amper', 'NEUT_per', 'NEUT_per_amper', 'P_LCR', 'PCT', 'PDW', 'TNC_D', 'WBC_D', 'HFLC_per', 'BA_N_num', 'BA_N_per', 'HCT', 'HGB', 'MacroR', 'MCH', 'MCHC', 'MCV', 'MicroR', 'NRBC_num',
    #                             'NRBC_per','PLT', 'PLT_I','RBC', 'RDW_CV', 'RDW_SD', 'TNC', 'TNC_N', 'WBC', 'WBC_N',
    #                             'SURGICAL', 'GESTATIONAL_AGE', 'AGE_DAYS_ONSET', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SUBJECT_ID', 'SEPSIS_GROUP']
    #
    # No_MPV_selectedColumns_extra_var_wgest = ['BA_D_num', 'BA_D_per', 'BASO_num', 'BASO_per', 'EO_per', 'EO_num', 'HFLC_num', 'IG_num', 'IG_per', 'LY_WX', 'LY_WY', 'LY_WZ', 'LY_X', 'LY_Y', 'LY_Z', 'LYMP_num_amper', 'LYMP_per_amper', 'LYMPH_num', 'LYMPH_per', 'MO_WX', 'MO_WY', 'MO_WZ', 'MO_X', 'MO_Y', 'MO_Z', 'MONO_num', 'MONO_per', 'NE_FSC', 'NE_SFL', 'NE_SSC', 'NE_WX', 'NE_WY', 'NE_WZ', 'NEUT_num', 'NEUT_per', 'P_LCR', 'PCT', 'PDW', 'TNC_D', 'WBC_D', 'HFLC_per', 'BA_N_num', 'BA_N_per', 'HCT', 'HGB', 'MacroR', 'MCH', 'MCHC', 'MCV', 'MicroR', 'NRBC_num',
    #                             'NRBC_per','PLT', 'PLT_I','RBC', 'RDW_CV', 'RDW_SD', 'TNC', 'TNC_N', 'WBC',
    #                             'SURGICAL', 'GESTATIONAL_AGE', 'AGE_DAYS_ONSET', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SUBJECT_ID', 'SEPSIS_GROUP']
    #
    # selectedColumns_clinical_var = ['GESTATIONAL_AGE', 'AGE_DAYS_ONSET', 'SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SUBJECT_ID', 'SEPSIS_GROUP']

    #dataset = dataset[No_MPV_selectedColumns_extra_var_wgest]
    #dataset = dataset.drop(['delta_days'], axis=1)
    #length = len(dataset.columns) - 1

    #remove nulls
    print(len(dataset))
    dataset = dataset.dropna()
    #dataset = dataset[selectedColumns_clinical_var]
    print(len(dataset))
    dataset = dataset.replace(to_replace='TRUE', value=1)
    dataset = dataset.replace(to_replace='FALSE', value=0)
    #dataset = Mean_Impute.impute_mean_na(dataset)
    dataset.index = range(len(dataset))
    patient_splitting_var = ['SUBJECT_ID', 'SEPSIS_GROUP']

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

    # # Vif/dataset formation for non-imputed variables
    Non_collinear_var = calculate_vif(dataset.iloc[:, :-2])
    final_data_set = Non_collinear_var.join(dataset[patient_splitting_var])
    unique_patients = dataset[patient_splitting_var].drop_duplicates(subset="SUBJECT_ID")

    #dataset.to_csv('result.csv')
    X = unique_patients.iloc[:, 0]
    y = unique_patients.iloc[:, 1]
    print(y.mean())
    print(len(unique_patients))

    #split data into sets
    # non cross validation
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TESTSIZE, random_state=config.SEED)
    #define sens and NPV metrics

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
    skf = StratifiedKFold(n_splits=10, random_state=config.SEED, shuffle=True)
    # get the folds and loop over each fold

    def imp_gridSearch(estimator, parameters, cv_scoring, X_train, y_train):
        skfgs = StratifiedKFold(n_splits=5, random_state=config.SEED)
        gs = GridSearchCV(estimator=estimator, param_grid=parameters, cv=skfgs, scoring=cv_scoring,
                          verbose=config.VERBOSE, return_train_score=True)
        gs.fit(X_train, y_train)
        return gs.best_estimator_, gs.best_params_, gs.cv_results_

    def imp_randomForest(X_train, y_train, parameters, cv_scoring, oversample):
        rf = RandomForestClassifier(random_state=config.SEED)
        classifier = Pipeline([
            (config.OVERSAMPLEMETHOD, oversample),
            ('clf', rf)])
        gs = imp_gridSearch(classifier, parameters, cv_scoring, X_train, y_train)
        return gs

    fold_number = 0

    headers = ['classifier', 'fold', 'pred_values', 'test_labels','y_pred', 'accuracy', 'roc', 'sensitivity', 'specificity', 'ppv', 'npv', 'test_set_size', 'percent_sepsis', 'f1_score', 'parameters']
    random_forest_results = [headers]
    feature_importances = []

    for train_index, test_index in skf.split(X, y):
        fold_number = fold_number + 1

        unique_train_patients = X.iloc[train_index].to_frame()
        unique_test_patients = X.iloc[test_index].to_frame()

        train_patient_data = final_data_set[final_data_set.SUBJECT_ID.isin(unique_train_patients.SUBJECT_ID)].drop(['SUBJECT_ID'], axis=1)
        test_patient_data = final_data_set[final_data_set.SUBJECT_ID.isin(unique_test_patients.SUBJECT_ID)].drop(['SUBJECT_ID'], axis=1)
        length = len(train_patient_data.columns) - 1

        X_train, X_test = pd.DataFrame(train_patient_data.iloc[:, 0:length]), pd.DataFrame(test_patient_data.iloc[:, 0:length])
        y_train, y_test = train_patient_data.iloc[:, length], test_patient_data.iloc[:, length]

        X_train_df = X_train
        # normalize data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Random forest
        #randomForestParameters = [{'clf__n_estimators': [500, 750], 'clf__criterion': ['entropy']}]
        randomForestParameters = [{'clf__n_estimators': [250,500, 750, 1000], 'clf__criterion': ['entropy']}]
        #classifier, bestParams, GSresults = classifiers.randomForest(X_train, y_train, randomForestParameters, config.CV_SCORING, SMOTE())
        classifier, bestParams, GSresults = imp_randomForest(X_train, y_train, randomForestParameters,config.CV_SCORING, SMOTE())
        random_forest_results.append(results.StoreResults(classifier, X_test, y_test, 'RF', bestParams, 'TEST', fold_number, GSresults))

        feature_importances.append(classifier.named_steps["clf"].feature_importances_)

        print("Fold {number} complete.".format(number=fold_number))

    timestr = time.strftime("%Y%m%d-%H%M%S")

    feature_importances = tuple(feature_importances)
    feature_importances = np.vstack(feature_importances)
    feature_importances = pd.DataFrame(feature_importances, columns=list(X_train_df))
    feature_importances = feature_importances.transpose()
    feature_importances.to_csv("RSF_feature_importances.csv")