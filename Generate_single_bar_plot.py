from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
import json
import pandas as pd
import numpy as np
import json
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from cycler import cycler
import statistics as stat
from scipy import stats
import scikit_posthocs as sp
import scipy
from matplotlib.lines import Line2D#old scipy is version 1.4.1

#graphing significance grid for models
#file_names = ["10_fold_results/10_fold_logistic20200812-180235.csv", "10_fold_results/10_fold_knn20200812-180235.csv", "10_fold_results/10_fold_random_forest20200812-180235.csv", "10_fold_results/10_fold_xgboost20200812-180235.csv",  "10_fold_results/10_fold_svm20200812-180235.csv"]

#No MPV files
file_names = ["10_fold_results/No_MPV_10_fold_feat_logistic.csv", "10_fold_results/No_MPV_10_fold_feat_knn.csv", "10_fold_results/No_MPV_10_fold_feat_random_forest.csv", "10_fold_results/No_MPV_10_fold_feat_xgboost.csv",  "10_fold_results/No_MPV_10_fold_feat_svm.csv"]
graph_names = ['Logistic', "KNN", "RF", "XGB", "SVM"]

def generate_data(outcome):
    for file in file_names:
        features = pd.read_csv(file).transpose()
        features.columns = features.iloc[0]
        features = features.drop(features.index[0])
        if outcome == "roc":
            df2 = pd.read_csv(file, index_col=0)
            rocs = []
            for column in df2:
                fold = df2[column]
                pred_value = json.loads(fold['pred_values'])
                pred_label = json.loads(fold['y_pred'])
                y_label = json.loads(fold['test_labels'])
                current_classifier = fold['classifier']
                fpr, tpr, threshold = metrics.roc_curve(y_label, pred_value)
                roc_auc = metrics.auc(fpr, tpr)
                rocs.append(roc_auc)
            data.append(rocs)
        else:
            data.append(features[outcome])

# # graphing the significance grid for RSF feature importances
file = "results/RSF_feature_importances.csv"
features = pd.read_csv(file).transpose()
features.columns = features.iloc[0]
features = features.drop(features.index[0])

# feature_names = ['NE_SFL', 'NE_FSC', 'NE_WY', 'GESTATIONAL_AGE', 'MO_Y']
# graph_names = ['NE_SFL', 'NE_FSC', 'NE_WY', 'GEST_AGE', 'MO_Y']

# feature_names = ['NE_SFL', 'NE_FSC', 'NE_WY', 'GESTATIONAL_AGE', 'MO_Y']
# graph_names = ['NE_SFL', 'NE_FSC', 'NE_WY', 'GEST_AGE', 'MO_Y']

# means = [stat.mean(features[i]) for i in feature_names]
# std = [stat.stdev(features[i]) for i in feature_names]
#
# data = [features[i] for i in feature_names]

# t_stat = conduct_t_test(xbg[0], xbg[1], "accuracy")
# t_stat = conduct_t_test(logistic[0], logistic[1], "accuracy")
# t_stat = conduct_t_test(RF[0], RF[1], "accuracy")
# t_stat = conduct_t_test(KNN[0], KNN[1], "accuracy")
# t_stat = conduct_t_test(svm[0], svm[1], "accuracy")
def generate_bar_plot(outcome):
    np.random.seed(12154)
    labels = graph_names
    x = np.arange(len(labels))
    w = 0.9    # bar width
    colors = [(0, 0, 1, 1), (1, 0, 0, 1)]
    #colors = [(0, 0, 1, 1), (0, 0, 1, 1)]
    #colors = [(1, 0, 0, 1), (1, 0, 0, 1)]
    point_colors = [(0, 0, 1, 1), (1, 0, 0, 1)]  # corresponding colors

    fig, ax = plt.subplots()
    rects =ax.bar(x,
           height=means,
           yerr=std,    # error bars
           capsize=6, # error bar cap width in points
           width=w,
           label="Full Model",
           #color=(0,0,1,1),  # face color transparent
           #edgecolor=colors,
           #ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
           )

    def add_value_labels(ax, spacing=15):
        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'
            if y_value < 0:
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.4f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(0, space),  # Vertically shift label by `space`
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha='center',  # Horizontally center label
                va=va)  # Vertically align label differently for
            # positive and negative values.
    # Call the function above. All the magic happens there.
    add_value_labels(ax)

    # bars = ax.patches
    # for rect, label in zip(bars, means):
    #     height = rect.get_height()
    #     ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

    plt.ylabel('Feature Importance')
    plt.yticks(np.arange(0, 0.11, 0.01))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.legend(loc="lower right")
    #plt.grid(which = 'both', axis = 'y')
    # if outcome == "roc":
    #     plt.title('Feature Only Model and Full Model AUC-ROC')
    # elif outcome == "ppv":
    #     plt.title('Feature Only Model and Full Model Positive Predictive Value')
    # elif outcome == "npv":
    #     plt.title('Feature Only Model and Full Model Negative Predictive Value')
    # else:
    #     plt.title('Feature Only Model and Full Model {}'.format(outcome.capitalize()))
    plt.title('Top Features for Random Forest Model')
    #plt.title('Mean OLIG2 Percentage (7790)')
    plt.savefig('results_graphs/{}_graph.png'.format(outcome), dpi=300, bbox_inches='tight')
    plt.show()

generate_bar_plot("rf_feat")
print("Barplots Complete")

