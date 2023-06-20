from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
import json
import pandas as pd
import numpy as np
import json
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from cycler import cycler

#file_names for all auc-roc for one set of results together in a plot
file_names = ["10_fold_feat_logistic.csv", "10_fold_feat_knn.csv", "10_fold_feat_random_forest.csv", "10_fold_feat_xgboost.csv",  "10_fold_feat_svm.csv"]

#file names for clinical vs full feat comparison
#as with bar plots, need to generate two sets of results, one with full set of variables and
#one with only clinical features using runXclassifiers twice first, put in names accordingly below
xbg = ["10_fold_xgboost.csv", "10_fold_clin_xgboost.csv"]
RF = ["to_fold_random_forest.csv","10_fold_clin_random_forest.csv"]
logistic = ["10_fold_logistic.csv", "10_fold_clin_logistic.csv"]
KNN = ["10_fold_knn.csv", "10_fold_clin_knn.csv"]
svm = ["10_fold_svm.csv", "10_fold_clin_svm.csv"]

# feature_file_names = ["features_results/features_knn20200722-224217.csv", "features_results/features_logistic20200722-191121.csv", "features_results/features_random_forest20200722-191121.csv" , "features_results/features_svm20200722-191121.csv", "features_results/features_xgboost20200722-191121"]
# #file = pd.read_csv(file_name, index_col=0)s
#
# xbg = ["results/USETHIS_XGB.csv", "features_results/features_xgboost20200722-191121.csv"]
# RF = ["results/USETHIS_RF.csv", "features_results/features_random_forest20200722-191121.csv"]
# logistic = ["results/logistic20200717-125321.csv", "features_results/features_logistic20200722-191121.csv"]
# KNN = ["results/knn20200723-044056.csv", "features_results/features_knn20200722-224217.csv"]
# svm = ["results/USETHIS_SVM.csv", "features_results/features_svm20200722-191121.csv"]

models = [xbg, RF, logistic, KNN, svm]

naming_dict = {"XGB": "XGBoost", "RF": "Random Forest", "LOGISTIC": "Logistic", "KNN": "KNN", "SVM": "SVM"}

#plotting all auc-roc curves for clin vs full w/ CIs
for i in models:
    fig, ax = plt.subplots()
    for file_name in i:
        if file_name == i[0]:
            color = "b"
        else:
            color = "r"
        file = pd.read_csv(file_name, index_col=0)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        current_classifier = ""
        for column in file:
            fold = file[column]
            pred_value = json.loads(fold['pred_values'])
            pred_label = json.loads(fold['y_pred'])
            y_label = json.loads(fold['test_labels'])
            current_classifier = fold['classifier']
            fpr, tpr, threshold = metrics.roc_curve(y_label, pred_value)
            roc_auc = metrics.auc(fpr, tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        if file_name == i[0]:
            ax.plot(mean_fpr, mean_tpr, color=color,
                    label=r'Full Model (Mean AUC = %0.2f)' % (mean_auc),
                    lw=2, linestyle ="dotted", alpha=.8)
            # ax.plot(mean_fpr, mean_tpr, color=color,
            #         label=r'Full Model (Mean AUC = %0.2f$\pm$%0.2f)' % (mean_auc, std_auc),
            #         lw=2, linestyle ="dotted", alpha=.8)
        else:
            ax.plot(mean_fpr, mean_tpr, color=color,
                    label=r'CF Only Model (Mean AUC = %0.2f)' % (mean_auc),
                    lw=2, linestyle ="dotted", alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
    ax.plot([0, 1], [0, 1], linestyle='-', lw=2, color='k',
                label='Chance', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="{}".format(naming_dict[current_classifier.upper()]))
    ax.legend(loc="lower right")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)

    plt.savefig('features_graphs/10_fold_{}_comparison.png'.format(current_classifier), dpi=300, bbox_inches='tight')
    plt.show()

#plotting comparisons between clinical and model - comparing clinical feature prediction
#
# plt.rc('lines', linewidth=4)
# plt.rc('axes', prop_cycle=(cycler('color', ['k', 'g', 'r', 'b', 'y', "c", "m", "y"]) + cycler('linestyle', ['-', '--', ':', '-.','-', '--', ':', '-.'])))
# fig, ax = plt.subplots()
# ax.plot([0, 1], [0, 1], lw=2, label='Chance', alpha=.8)
# count = 1
# for file_name in svm:
#     file = pd.read_csv(file_name, index_col=0)
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#     current_classifier = ""
#     for column in file:
#         fold = file[column]
#         pred_value = json.loads(fold['pred_values'])
#         pred_label = json.loads(fold['y_pred'])
#         y_label = json.loads(fold['test_labels'])
#         current_classifier = fold['classifier']
#         fpr, tpr, threshold = metrics.roc_curve(y_label, pred_value)
#         roc_auc = metrics.auc(fpr, tpr)
#         #ax.plot(fpr, tpr, 'b', label="Fold " + column + ' AUC = %0.2f' % roc_auc, alpha=0.3, lw=1)
#         interp_tpr = np.interp(mean_fpr, fpr, tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         aucs.append(roc_auc)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = metrics.auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     if count == 1:
#         ax.plot(mean_fpr, mean_tpr,
#             label=r'{} (Full Model Mean AUC = %0.2f)'.format(current_classifier.capitalize()) % (mean_auc),
#             lw=2, alpha=.8)
#     else:
#         ax.plot(mean_fpr, mean_tpr,
#             label=r'{} (Clinical Variable Only Mean AUC = %0.2f)'.format(current_classifier.capitalize()) % (mean_auc),
#             lw=2, alpha=.8)
#     # std_tpr = np.std(tprs, axis=0)
#     # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#     #                 label=r'$\pm$ 1 std. dev.')
#     count = count - 1
# ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
#            title="Receiver Operating Characteristic ({} Model Comparison)".format(naming_dict[current_classifier.capitalize()]))
# ax.legend(loc="lower right")
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
#
# plt.savefig('features_graphs/test_SVM_Comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

#plotting all of the AOC ROCS for the models together
plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=(cycler('color', ['k', 'r', 'g', 'b', 'y', "c", "m", "y"]) + cycler('linestyle', ['-', ':', ':', ':', ':', ':', ':', ':'])))
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], lw=2, label='Chance', alpha=.8)
for file_name in file_names:
    file = pd.read_csv(file_name, index_col=0)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    current_classifier = ""
    for column in file:
        fold = file[column]
        pred_value = json.loads(fold['pred_values'])
        pred_label = json.loads(fold['y_pred'])
        y_label = json.loads(fold['test_labels'])
        current_classifier = fold['classifier']
        fpr, tpr, threshold = metrics.roc_curve(y_label, pred_value)
        roc_auc = metrics.auc(fpr, tpr)
        #ax.plot(fpr, tpr, 'b', label="Fold " + column + ' AUC = %0.2f' % roc_auc, alpha=0.3, lw=1)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr,
            label=r'{} (Mean AUC = %0.2f)'.format(naming_dict[current_classifier.upper()]) % (mean_auc),
            lw=2, alpha=.8)
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                 label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver Operating Characteristic (Means)")
ax.legend(loc="lower right")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid(True)

#plt.savefig('graphs/AUC_ROC_Summary.png', dpi=300, bbox_inches='tight')
plt.savefig('10_fold_AUC_ROC_Summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("Curves Generated!")
