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
import scipy
from matplotlib.lines import Line2D

#for comparing results between clinical feat models and full variable models
#need to generate two sets of results, one with full set of variables and
#one with only clinical features using runXclassifiers twice first, put in names accordingly below

xbg = ["10_fold_xgboost.csv", "10_fold_clin_xgboost.csv"]
RF = ["to_fold_random_forest.csv","10_fold_clin_random_forest.csv"]
logistic = ["10_fold_logistic.csv", "10_fold_clin_logistic.csv"]
KNN = ["10_fold_knn.csv", "10_fold_clin_knn.csv"]
svm = ["10_fold_svm.csv", "10_fold_clin_svm.csv"]

models = [xbg, RF, logistic, KNN, svm]

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05
        while data < p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
            if text=="***":
                break
        if len(text) == 0:
            text = 'n. s.'
    lx, ly = num1-0.225, height[0]
    rx, ry = num2+0.225, height[1]
    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]
    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)
    y = max(ly, ry) + dh
    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)
    plt.plot(barx, bary, c='black')
    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
    plt.text(*mid, text, **kwargs)

def add_scores_dict(dict, file_name):
    file = pd.read_csv(file_name, index_col=0)
    accuracies = []
    aucs = []
    sensitivities = []
    specificities = []
    ppvs = []
    npvs = []
    for column in file:
        fold = file[column]
        accuracies.append(float(fold['accuracy']))
        aucs.append(float(fold['roc']))
        sensitivities.append(float(fold['sensitivity']))
        specificities.append(float(fold['specificity']))
        ppvs.append(float(fold['ppv']))
        npvs.append(float(fold['npv']))
    dict["accuracy"] = (stat.mean(accuracies), stat.stdev(accuracies))
    dict["roc"] = (stat.mean(aucs), stat.stdev(aucs))
    dict["sensitivity"] = (stat.mean(sensitivities), stat.stdev(sensitivities))
    dict["specificity"] = (stat.mean(specificities), stat.stdev(specificities))
    dict["ppv"] = (stat.mean(ppvs), stat.stdev(ppvs))
    dict["npv"] = (stat.mean(npvs), stat.stdev(npvs))

def conduct_t_test(file1, file2, outcome):
    #check for t_test assumptions:
    print("Results for " +str(file1))
    file1 = pd.read_csv(file1, index_col=0)
    # accuracies1 = []
    # aucs1 = []
    # sensitivities1 = []
    # specificities1 = []
    outcome1 = []
    for column in file1:
        fold = file1[column]
        outcome1.append(float(fold[str(outcome)]))
        # accuracies1.append(float(fold['accuracy']))
        # aucs1.append(float(fold['roc']))
        # sensitivities1.append(float(fold['sensitivity']))
        # specificities1.append(float(fold['specificity']))
    print("Checking for normality:")
    print(str(outcome) + " 1: " + str(stats.shapiro(outcome1)))
    # print("Accuracy 1: " + str(stats.shapiro(accuracies1)))
    # print("AUC 1: " + str(stats.shapiro(aucs1)))
    # print("Sensitivity 1: " + str(stats.shapiro(sensitivities1)))
    # print("Specificity 1: " + str(stats.shapiro(specificities1)))

    file2 = pd.read_csv(file2, index_col=0)
    outcome2 = []
    for column in file2:
        fold = file2[column]
        outcome2.append(float(fold[str(outcome)]))
        # accuracies1.append(float(fold['accuracy']))
        # aucs1.append(float(fold['roc']))
        # sensitivities1.append(float(fold['sensitivity']))
        # specificities1.append(float(fold['specificity']))
    print("Checking for normality:")
    print(str(outcome) + " 2: " + str(stats.shapiro(outcome2)))
    # accuracies2 = []
    # aucs2 = []
    # sensitivities2 = []
    # specificities2 = []
    # for column in file2:
    #     fold = file2[column]
    #     accuracies2.append(float(fold['accuracy']))
    #     aucs2.append(float(fold['roc']))
    #     sensitivities2.append(float(fold['sensitivity']))
    #     specificities2.append(float(fold['specificity']))
    # print("Accuracy 2: " + str(stats.shapiro(accuracies2)))
    # print("AUC 2: " + str(stats.shapiro(aucs2)))
    # print("Sensitivity 2: " + str(stats.shapiro(sensitivities2)))
    # print("Specificity 2: " + str(stats.shapiro(specificities2)))

    print("Checking for homogeneity of variance")
    print("Levene " + str(outcome) + ": " + str(stats.levene(outcome1, outcome2)))
    # print("Levene_accuracy: " + str(stats.levene(accuracies1, accuracies2)))
    # print("Levene_auc: " + str(stats.levene(aucs1, aucs2)))
    # print("Levene_sensitivity: " + str(stats.levene(sensitivities1, sensitivities2)))
    # print("Levene_specificity: " + str(stats.levene(specificities1, specificities2)))

    if stats.shapiro(outcome1)[1] <= 0.05 or stats.shapiro(outcome2)[1] <= 0.05:
        print("Non normal distribution. Performing Wilcoxon rank sum test.")
        if stats.levene(outcome1, outcome2)[1] <= 0.05:
            print(outcome + " has non-normal distribution and unequal variance.")
            print("\n")
        wilcoxon = stats.wilcoxon(outcome1, outcome2, mode="exact")
        print("Wilcoxon results are "+str(wilcoxon))
        print("\n")
        return wilcoxon[1]

    if stats.levene(outcome1, outcome2)[1] <= 0.05:
        print("Non equal variances. Performing Welch T-test.")
        t_stat = stats.ttest_ind(outcome1, outcome2, equal_var= False)
        print("Welch T_stat is " + str(t_stat))
        print("\n")
        return t_stat[1]

    t_stat = stats.ttest_ind(outcome1, outcome2)
    print("T_stat is " + str(t_stat))
    print("\n")
    return t_stat[1]

xbg_dict = {}
xbg_feature_dict = {}
add_scores_dict(xbg_dict, xbg[0])
add_scores_dict(xbg_feature_dict, xbg[1])

RF_dict = {}
RF_feature_dict = {}
add_scores_dict(RF_dict, RF[0])
add_scores_dict(RF_feature_dict, RF[1])

logistic_dict = {}
logistic_feature_dict = {}
add_scores_dict(logistic_dict, logistic[0])
add_scores_dict(logistic_feature_dict, logistic[1])

KNN_dict = {}
KNN_feature_dict = {}
add_scores_dict(KNN_dict, KNN[0])
add_scores_dict(KNN_feature_dict, KNN[1])

svm_dict = {}
svm_feature_dict = {}
add_scores_dict(svm_dict, svm[0])
add_scores_dict(svm_feature_dict, svm[1])

dicts = [logistic_dict, KNN_dict, RF_dict, xbg_dict, svm_dict]
feature_dicts = [logistic_feature_dict, KNN_feature_dict, RF_feature_dict, xbg_feature_dict, svm_feature_dict]

# t_stat = conduct_t_test(xbg[0], xbg[1], "accuracy")
# t_stat = conduct_t_test(logistic[0], logistic[1], "accuracy")
# t_stat = conduct_t_test(RF[0], RF[1], "accuracy")
# t_stat = conduct_t_test(KNN[0], KNN[1], "accuracy")
# t_stat = conduct_t_test(svm[0], svm[1], "accuracy")

def generate_bar_plot(outcome):
    np.random.seed(12154)
    labels = ["Logistic", "KNN", "Random Forest", "XGBoost", "SVM"]
    x = np.arange(len(labels))
    w = 0.45    # bar width
    colors = [(0, 0, 1, 1), (1, 0, 0, 1)]
    #colors = [(0, 0, 1, 1), (0, 0, 1, 1)]
    #colors = [(1, 0, 0, 1), (1, 0, 0, 1)]
    point_colors = [(0, 0, 1, 1), (1, 0, 0, 1)]  # corresponding colors

    fig, ax = plt.subplots()
    rects =ax.bar(x + w/2,
           height=[y[outcome][0] for y in dicts],
           yerr=[(y[outcome][1]) for y in dicts],    # error bars
           capsize=6, # error bar cap width in points
           width=w,
           label = "Full Model",   # bar width
           #color=(0,0,1,1),  # face color transparent
           #edgecolor=colors,
           #ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
           )
    rects2 =ax.bar(x - w/2,
           height=[y[outcome][0] for y in feature_dicts],
           yerr=[(y[outcome][1]) for y in feature_dicts],    # error bars
           capsize=6, # error bar cap width in points
           width=w,    # bar width
           label = "Clinical Feature Only Model",
           #color=(1,0,0,1),  # face color transparent
           #edgecolor=colors,
           #ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
           )
    t_stats = [conduct_t_test(logistic[0], logistic[1], outcome), conduct_t_test(KNN[0], KNN[1], outcome), conduct_t_test(RF[0], RF[1], outcome), conduct_t_test(xbg[0], xbg[1], outcome),  conduct_t_test(svm[0], svm[1], outcome)]
    #barplot_annotate_brackets(-.5, 1.5, t_stats[1], [dicts[0][outcome][0], feature_dicts[0][outcome][0]], heights)
    for i in range(len(dicts)):
        barplot_annotate_brackets(i, i, t_stats[i], x, [float(dicts[i][outcome][0])+2*float(dicts[i][outcome][1]), float(feature_dicts[i][outcome][0])+2*float(feature_dicts[i][outcome][1])], dh = 0.03)

    barplot_annotate_brackets(1, 1, t_stats[1], x, [dicts[1][outcome][0], feature_dicts[1][outcome][0]], dh=0.12)
    barplot_annotate_brackets(2, 2, t_stats[2], x, [dicts[2][outcome][0], feature_dicts[2][outcome][0]], dh=0.12)
    # barplot_annotate_brackets(1, 2, .001, bars, heights)
    # barplot_annotate_brackets(0, 2, 'p < 0.0075', bars, heights, dh=.2)

    plt.ylabel('Percent')
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.legend(loc="lower right")
    ax.legend(bbox_to_anchor=(0.53, -0.1), loc="center",
                    mode="expand", borderaxespad=0, ncol=2, facecolor = "darkgray", edgecolor = "white")

    plt.grid(which = 'both', axis = 'y')
    if outcome == "roc":
        plt.title('Feature Only Model and Full Model AUC-ROC')
    elif outcome == "ppv":
        plt.title('Feature Only Model and Full Model Positive Predictive Value')
    elif outcome == "npv":
        plt.title('Feature Only Model and Full Model Negative Predictive Value')
    else:
        plt.title('Feature Only Model and Full Model {}'.format(outcome.capitalize()))

    # if outcome == "roc":
    #     plt.title('AUC-ROC')
    # elif outcome == "ppv":
    #     plt.title('Positive Predictive Value')
    # elif outcome == "npv":
    #     plt.title('Negative Predictive Value')
    # else:
    #     plt.title('{}'.format(outcome.capitalize()))
    #
    plt.savefig('results_graphs/10_fold_{}.png'.format(outcome), dpi=300, bbox_inches='tight')

    plt.show()

generate_bar_plot("accuracy")
generate_bar_plot("roc")
generate_bar_plot("sensitivity")
generate_bar_plot("specificity")
generate_bar_plot("ppv")
generate_bar_plot("npv")

print("Barplots Complete")

