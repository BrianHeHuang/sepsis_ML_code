from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from config import config
from xgboost import XGBClassifier
from imblearn.pipeline import make_pipeline, Pipeline
import itertools
from sklearn.model_selection import GridSearchCV, StratifiedKFold

#define classifiers

def XG(X_train, y_train, parameters, cv_scoring, oversample):
    xg = XGBClassifier(random_state=config.SEED)
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', xg)])
    gs = gridSearch(classifier, parameters, cv_scoring, X_train, y_train)
    return gs

def randomForest(X_train, y_train, parameters, cv_scoring, oversample):
    rf = RandomForestClassifier(random_state=config.SEED)
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', rf)])
    gs = gridSearch(classifier, parameters, cv_scoring, X_train, y_train)
    return gs

def KNN(X_train, y_train, parameters, cv_scoring, oversample):
    knn = KNeighborsClassifier()
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', knn)])
    gs = gridSearch(classifier, parameters, cv_scoring, X_train, y_train)
    return gs

def logistic(X_train, y_train, parameters, cv_scoring, oversample):
    logit = LogisticRegression(solver='lbfgs', penalty='l2', class_weight='balanced', random_state=config.SEED)
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', logit)])
    gs = gridSearch(classifier, parameters, cv_scoring, X_train, y_train)
    return gs

def SVM(X_train, y_train, parameters, cv_scoring, oversample):
    svm = SVC(gamma='scale', kernel=config.SVM_KERNEL, probability=True, class_weight='balanced', random_state=config.SEED)
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', svm)])
    gs = gridSearch(classifier, parameters, cv_scoring, X_train, y_train)
    return gs

def ADA(X_train, y_train, parameters, cv_scoring, oversample):
    ada = AdaBoostClassifier(random_state=config.SEED)
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', ada)])
    gs = gridSearch(classifier, parameters, cv_scoring, X_train, y_train)
    return gs

def gridSearch(estimator, parameters, cv_scoring, X_train, y_train):
    skfgs = StratifiedKFold(n_splits=config.SPLIT, random_state=config.SEED)
    gs = GridSearchCV(estimator=estimator, param_grid=parameters, cv=skfgs, scoring=cv_scoring, verbose=config.VERBOSE, return_train_score=True)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.cv_results_


def Bagging_logistic(X_train, y_train, parameters, cv_scoring, oversample):
    logit = LogisticRegression(solver='lbfgs', penalty='l2', class_weight='balanced', random_state=config.SEED)
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', logit)])
    gs = gridSearch(BaggingClassifier(base_estimator=classifier), parameters, cv_scoring, X_train, y_train)
    return gs

def Bagging_SVM(X_train, y_train, parameters, cv_scoring, oversample):
    svm = SVC(gamma='scale', kernel=config.SVM_KERNEL, probability=True, class_weight='balanced', random_state=config.SEED)
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', svm)])
    gs = gridSearch(BaggingClassifier(base_estimator=classifier), parameters, cv_scoring, X_train, y_train)
    return gs

def Bagging_randomForest(X_train, y_train, parameters, cv_scoring, oversample):
    rf = RandomForestClassifier(random_state=config.SEED)
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', rf)])
    gs = gridSearch(BaggingClassifier(base_estimator=classifier), parameters, cv_scoring, X_train, y_train)
    return gs

def Bagging_KNN(X_train, y_train, parameters, cv_scoring, oversample):
    knn = KNeighborsClassifier()
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', knn)])
    gs = gridSearch(BaggingClassifier(base_estimator=classifier), parameters, cv_scoring, X_train, y_train)
    return gs

def Bagging_XG(X_train, y_train, parameters, cv_scoring, oversample):
    xg = XGBClassifier(random_state=config.SEED)
    classifier = Pipeline([
        (config.OVERSAMPLEMETHOD, oversample),
        ('clf', xg)])
    gs = gridSearch(BaggingClassifier(base_estimator=classifier), parameters, cv_scoring, X_train, y_train)
    return gs

#generates classifier parameters for votingclassifier
def param_gen(classifiers):
    parameters = []
    for L in range(1, len(classifiers) + 1):
        for subset in itertools.combinations(classifiers, L):
            parameters.append(list(subset))
    return parameters

def results_grid(X_test, y_test, classifiers):
    labels = y_test
    for classifier in classifiers:
        predictions = classifier.predict(X_test)
        labels[str(classifier)] = predictions
        #labels = labels.append(predictions)
    labels['sum'] = labels[1:].sum(axis=1)
    if labels['sum'] >= len(classifiers)/2:
        labels['predictions'] = 1
    else:
        labels['predictions'] = 0
    return labels