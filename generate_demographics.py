import pandas as pd
from config import config
import functools
import numpy as np
from tableone import TableOne, load_dataset
from scipy import stats
import statistics as stat

def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

def orconjunction(*conditions):
    return functools.reduce(np.logical_or, conditions)

dataset = pd.read_csv(config.DATAFILE, dtype={'SURGICAL': str, 'CARDIAC': str, 'CLD': str, 'IVHSHUNT': str, 'NEC': str})

# #filter dates
c_1 = dataset.delta_days < 1
c_2 = dataset.delta_days > -2
dataset = dataset[conjunction(c_1, c_2)]

# #add group 3 to 1
dataset = dataset.replace({'SEPSIS_GROUP': {2: 0}})
dataset = dataset.replace({'SEPSIS_GROUP': {3: 1}})
c_1 = dataset.SEPSIS_GROUP == 0
c_2 = dataset.SEPSIS_GROUP == 1
dataset = dataset[orconjunction(c_1, c_2)]

# selected features to use
selectedColumns = ['NE_SFL', 'NE_FSC', 'MO_WX', 'MO_Y', 'LY_X', 'MPV', 'LYMP_per_amper', 'MicroR', 'LYMPH_per',
                   'MCV', 'NE_WY', 'LYMP_num_amper', 'LYMPH_num', 'MO_X', 'LY_WY', 'LY_WZ', 'HCT', 'MO_WY', 'LY_Y',
                   'NEUT_per', 'NEUT_per_amper', 'HGB', 'MCH', 'MacroR', 'PCT', 'EO_num', 'PLT_I', 'PLT', 'RBC',
                   'EO_per', 'LY_Z', 'IG_per', 'RDW_CV', 'BA_N_per', 'MO_WZ', 'BASO_per', 'HFLC_per', 'IG_num',
                   'LY_WX', 'BASO_num', 'NEUT_num', 'NEUT_num_amper', 'MONO_num', 'BA_N_num', 'MONO_per',
                   'HFLC_num', 'SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SEPSIS_GROUP']


all_variables = ['BA_D_num', 'BA_D_per', 'BASO_num', 'BASO_per', 'EO_per', 'EO_num', 'HFLC_num',
                                   'IG_num', 'IG_per', 'LY_WX', 'LY_WY', 'LY_WZ', 'LY_X', 'LY_Y', 'LY_Z',
                                   'LYMP_num_amper', 'LYMP_per_amper', 'LYMPH_num', 'LYMPH_per', 'MO_WX', 'MO_WY',
                                   'MO_WZ', 'MO_X', 'MO_Y', 'MO_Z', 'MONO_num', 'MONO_per', 'NE_FSC', 'NE_SFL',
                                   'NE_SSC', 'NE_WX', 'NE_WY', 'NE_WZ', 'NEUT_num', 'NEUT_num_amper', 'NEUT_per',
                                   'NEUT_per_amper', 'P_LCR', 'PCT', 'PDW', 'TNC_D', 'WBC_D', 'HFLC_per', 'BA_N_num',
                                   'BA_N_per', 'HCT', 'HGB', 'MacroR', 'MCH', 'MCHC', 'MCV', 'MicroR', 'NRBC_num',
                                   'NRBC_per', 'PLT', 'PLT_I', 'RBC', 'RDW_CV', 'RDW_SD', 'TNC', 'TNC_N', 'WBC',
                                   'WBC_N',
                                   'SURGICAL', 'GESTATIONAL_AGE', 'AGE_DAYS_ONSET', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC',
                                   'SUBJECT_ID', 'SEPSIS_GROUP']

selectedColumns_clinical_var = ['SEX', 'RACE','GESTATIONAL_AGE', 'AGE_DAYS_ONSET', 'SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SEPSIS_GROUP']
binary_var = ['SEX', 'RACE','SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SEPSIS_GROUP']

# dataset = dataset[selectedColumns_clinical_var]
dataset = dataset
# dataset = dataset.drop(['delta_days'], axis=1)
# length = len(dataset.columns) - 1


# remove nulls
print(len(dataset))
#dataset = dataset.dropna()
print(len(dataset))
print(len(dataset[dataset["SEPSIS_GROUP"]==1]))
dataset = dataset.replace(to_replace='TRUE', value=1)
dataset = dataset.replace(to_replace='FALSE', value=0)
dataset = dataset.replace(to_replace='M', value=1)
dataset = dataset.replace(to_replace='F', value=0)
dataset = dataset.replace(to_replace='UN', value=0)
dataset = dataset.replace(to_replace='BLACK', value=1)
dataset = dataset.replace(to_replace='AMERICAN INDIAN OR ALASKA NATIVE', value=0)
dataset = dataset.replace(to_replace='ASIAN', value=0)
dataset = dataset.replace(to_replace='TWO_OR_MORE', value=0)
dataset = dataset.replace(to_replace='UNKNOWN', value=0)
dataset = dataset.replace(to_replace='WHITE', value=0)
# dataset = Mean_Impute.impute_mean_na(dataset)
dataset.index = range(len(dataset))

# non_normal=[]
# for i in all_variables:
#     outcome1 = dataset[dataset['SEPSIS_GROUP']==1][i]
#     outcome2 = dataset[dataset['SEPSIS_GROUP']==0][i]
#     a = stats.shapiro(outcome1)[0]
#     b = stats.shapiro(outcome1)[1]
#     if a<=0.05 or b<=0.05:
#         non_normal.append(i)

clin_var_table = pd.DataFrame(index = selectedColumns_clinical_var, columns=["Number Missing", "Overall",  "Sepsis Negative", "Sepsis Positive", "P-Value", "Test Used"])
# clin_var_table.at["Total Patients", "Overall"] = len(dataset)
# clin_var_table.at["Total Patients", "Sepsis Negative"] = len(dataset[dataset["SEPSIS_GROUP" == 0]])
# clin_var_table.at["Total Patients", "Sepsis Positive"] = len(dataset[dataset["SEPSIS_GROUP" == 1]])
for i in selectedColumns_clinical_var:
    missing = dataset[i].isna().sum()
    clin_var_table.at[i, "Number Missing"] = missing
    overall = dataset[i].dropna()
    mean = round(stat.mean(overall), 2)
    std = round(stat.stdev(overall), 2)
    string = str(mean) + " (" + str(std) + ")"
    clin_var_table.at[i, "Overall"] = string
    #clin_var_table.at[i, "Overall"] = round((stat.mean(overall)), 2)
    clin_var_table.at[i, "Overall_SD"] = stat.stdev(overall)
    outcome1 = dataset[dataset['SEPSIS_GROUP']==0][i].dropna()
    mean = round(stat.mean(outcome1), 2)
    std = round(stat.stdev(outcome1), 2)
    string = str(mean) + " (" + str(std) + ")"
    clin_var_table.at[i, "Sepsis Negative"] = string
    clin_var_table.at[i, "Sepsis Negative SD"] = stat.stdev(outcome1)
    outcome2 = dataset[dataset['SEPSIS_GROUP']==1][i].dropna()
    mean = round(stat.mean(outcome2), 2)
    std = round(stat.stdev(outcome2), 2)
    string = str(mean) + " (" + str(std) + ")"
    clin_var_table.at[i, "Sepsis Positive"] = string
    #clin_var_table.at[i, "Sepsis Positive"] = stat.mean(outcome2)
    clin_var_table.at[i, "Sepsis Positive SD"] = stat.stdev(outcome2)
    if i in binary_var:
        i_dataset = dataset[~dataset[i].isnull()]
        contingency = pd.crosstab(i_dataset[i], i_dataset["SEPSIS_GROUP"])
        c, p, dof, expected = stats.chi2_contingency(contingency)
        clin_var_table.at[i, "P-Value"] = p
        clin_var_table.at[i, "Test Used"] = "Chi-Squared"
    else:
        t_test = stats.ttest_ind(outcome1, outcome2, nan_policy='omit')
        clin_var_table.at[i, "P-Value"] = t_test[1]
        clin_var_table.at[i, "Test Used"] = "Two-sample T-Test"

clin_var_table["Bonferroni P-Value"] = 0.05 / (len(selectedColumns_heme_variables) + len(selectedColumns_clinical_var))
clin_var_table.to_csv("no_MPV_clinical_variable_table.csv")

clin_var_table = pd.DataFrame(index = selectedColumns_heme_variables, columns=["Number Missing", "Overall", "Sepsis Negative", "Sepsis Positive", "P-Value", "Test Used"])
# clin_var_table.at["Total Patients", "Overall"] = len(dataset)
# clin_var_table.at["Total Patients", "Sepsis Negative"] = len(dataset[dataset["SEPSIS_GROUP" == 0]])
# clin_var_table.at["Total Patients", "Sepsis Positive"] = len(dataset[dataset["SEPSIS_GROUP" == 1]])
for i in selectedColumns_heme_variables:
    missing = dataset[i].isna().sum()
    clin_var_table.at[i, "Number Missing"] = missing
    overall = dataset[i].dropna()
    mean = round(stat.mean(overall), 2)
    std = round(stat.stdev(overall), 2)
    string = str(mean) + " (" + str(std) + ")"
    clin_var_table.at[i, "Overall"] = string
    #clin_var_table.at[i, "Overall"] = round((stat.mean(overall)), 2)
    clin_var_table.at[i, "Overall_SD"] = stat.stdev(overall)
    outcome1 = dataset[dataset['SEPSIS_GROUP']==0][i].dropna()
    mean = round(stat.mean(outcome1), 2)
    std = round(stat.stdev(outcome1), 2)
    string = str(mean) + " (" + str(std) + ")"
    clin_var_table.at[i, "Sepsis Negative"] = string
    clin_var_table.at[i, "Sepsis Negative SD"] = stat.stdev(outcome1)
    outcome2 = dataset[dataset['SEPSIS_GROUP']==1][i].dropna()
    mean = round(stat.mean(outcome2), 2)
    std = round(stat.stdev(outcome2), 2)
    string = str(mean) + " (" + str(std) + ")"
    clin_var_table.at[i, "Sepsis Positive"] = string
    #clin_var_table.at[i, "Sepsis Positive"] = stat.mean(outcome2)
    clin_var_table.at[i, "Sepsis Positive SD"] = stat.stdev(outcome2)
    t_test = stats.ttest_ind(outcome1, outcome2, nan_policy='omit')
    if t_test[1] < 0.0001:
        clin_var_table.at[i, "P-Value"] = "<0.0001"
    else:
        clin_var_table.at[i, "P-Value"] = t_test[1]
    clin_var_table.at[i, "Test Used"] = "Two-sample T-Test"

clin_var_table["Bonferroni P-Value"] = 0.05 / (len(selectedColumns_heme_variables) + len(selectedColumns_clinical_var))
#significant_clin_var_table = clin_var_table[clin_var_table["P-Value"] <= clin_var_table["Bonferroni P-Value"]]
clin_var_table.to_csv("heme_variable_table.csv")
#significant_clin_var_table.to_csv("no_MPV_sig_heme_variable_table.csv")