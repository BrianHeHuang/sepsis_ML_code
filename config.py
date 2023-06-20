from sklearn.metrics import make_scorer, classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score

class Config(object):
    DATAFILE = 'rawSepsis.csv'
    SEED = 3245
    BAGGING_SEED = 42
    SPLIT = 5
    TESTSIZE = .10
    NUMBER_OF_FOLDS=10
    #NUMBER_OF_FOLDS = 5

    #CV_SCORING = 'roc_auc'
    CV_SCORING = 'accuracy'

    VERBOSE = 0
    SVM_KERNEL = 'rbf'
    OVERSAMPLEMETHOD = 'SMOTE'


config = Config()