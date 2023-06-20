from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
import json
import pandas as pd

def StoreResults(classifier, X_test, y_test, fileName, bestParams, set, foldNumber, GSresults):
    results_list = []
    model = fileName
    Fold = foldNumber
    pred_values = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_labels = y_test
    roc = roc_auc_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    test_set_size = tp + fp + tn + fn
    percent_sepsis = (tp + fn)/(test_set_size)
    f1 = f1_score(y_test, y_pred)
    params = json.dumps(bestParams)
    results_list.append(model)
    results_list.append(Fold)
    results_list.append(str(((pd.Series(pred_values[:,1]))).tolist()))
    #results_list.append(pred_values[:,1].tostring())
    test_labels.index = range(len(test_labels))
    results_list.append(str(test_labels.tolist()))
    results_list.append(str((pd.Series(y_pred)).tolist()))
    results_list.append(accuracy)
    results_list.append(roc)
    results_list.append(sensitivity)
    results_list.append(specificity)
    results_list.append(ppv)
    results_list.append(npv)
    results_list.append(test_set_size)
    results_list.append(percent_sepsis)
    results_list.append(f1)
    results_list.append(params)
    return results_list

def getResults(classifier, X_test, y_test, fileName, bestParams, set, foldNumber, GSresults):
    f = open(fileName, 'a')
    y_pred = classifier.predict(X_test)
    #f.write('\n' + 'Best parameters: ' + '\n')
    f.write('\n' + '07_03_20_Soft_Vote_Best3' + '\n')
    f.write('Fold' + str(foldNumber) + '\n')
    f.write('SET; ' + set + '\n')
    params = json.dumps(bestParams)
    f.write(str(params) + '\n')
    f.write('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')
    roc = roc_auc_score(y_test, y_pred)
    f.write('ROC: ' + str(roc) + '\n')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f.write('Sensitivity: ' + str(sensitivity) + '\n')
    f.write('Specificity: ' + str(specificity) + '\n')
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    test_set_size = tp + fp + tn + fn
    percent_sepsis = (tp + fn)/(test_set_size)
    f.write('Positive Predictive Value: ' +str(ppv) + '\n')
    f.write('Negative Predictive Value: ' + str(npv) + '\n')
    f1 = f1_score(y_test, y_pred)
    f.write('F1 Score: ' + str(f1) + '\n')
    f.write('Test Set Size: ' + str(test_set_size) + '\n')
    f.write('Percent Test Sepsis: ' + str(percent_sepsis) + '\n')
    f.close()
