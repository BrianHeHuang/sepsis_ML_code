import pandas as pd
from config import config
import functools
import numpy as np
from tableone import TableOne, load_dataset
from scipy import stats
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

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
# #selectedColumns = ['NE_SFL', 'NE_FSC', 'MO_WX', 'MO_Y', 'LY_X', 'MPV', 'LYMP_per_amper', 'MicroR', 'LYMPH_per',
#                    'MCV', 'NE_WY', 'LYMP_num_amper', 'LYMPH_num', 'MO_X', 'LY_WY', 'LY_WZ', 'HCT', 'MO_WY', 'LY_Y',
#                    'NEUT_per', 'NEUT_per_amper', 'HGB', 'MCH', 'MacroR', 'PCT', 'EO_num', 'PLT_I', 'PLT', 'RBC',
#                    'EO_per', 'LY_Z', 'IG_per', 'RDW_CV', 'BA_N_per', 'MO_WZ', 'BASO_per', 'HFLC_per', 'IG_num',
#                    'LY_WX', 'BASO_num', 'NEUT_num', 'NEUT_num_amper', 'MONO_num', 'BA_N_num', 'MONO_per',
#                    'HFLC_num', 'SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SEPSIS_GROUP']

selectedColumns_mostdata = ['BA_N_num', 'BA_N_per', 'HCT', 'HGB', 'MacroR', 'MCH', 'MCHC', 'MCV', 'MicroR', 'NRBC_num',
                            'NRBC_per', 'PLT', 'PLT_I', 'RBC', 'RDW_CV', 'RDW_SD', 'TNC', 'TNC_N', 'WBC', 'WBC_N',
                            'SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SUBJECT_ID', 'SEPSIS_GROUP']

# selectedColumns_extra_var = ['BA_D_num', 'BA_D_per', 'BASO_num', 'BASO_per', 'EO_per', 'EO_num', 'HFLC_num', 'IG_num', 'IG_per', 'LY_WX', 'LY_WY', 'LY_WZ', 'LY_X', 'LY_Y', 'LY_Z', 'LYMP_num_amper', 'LYMP_per_amper', 'LYMPH_num', 'LYMPH_per', 'MO_WX', 'MO_WY', 'MO_WZ', 'MO_X', 'MO_Y', 'MO_Z', 'MONO_num', 'MONO_per', 'NE_FSC', 'NE_SFL', 'NE_SSC', 'NE_WX', 'NE_WY', 'NE_WZ', 'NEUT_num', 'NEUT_num_amper', 'NEUT_per', 'NEUT_per_amper', 'P_LCR', 'PCT', 'PDW', 'TNC_D', 'WBC_D', 'HFLC_per', 'BA_N_num', 'BA_N_per', 'HCT', 'HGB', 'MacroR', 'MCH', 'MCHC', 'MCV', 'MicroR', 'NRBC_num',
#                             'NRBC_per','PLT', 'PLT_I','RBC', 'RDW_CV', 'RDW_SD', 'TNC', 'TNC_N', 'WBC', 'WBC_N',
#                             'GESTATIONAL_AGE', 'AGE_DAYS_ONSET','SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SUBJECT_ID', 'SEPSIS_GROUP']
# selectedColumns_extra_var = ['BA_D_num', 'BA_D_per', 'BASO_num', 'BASO_per', 'EO_per', 'EO_num', 'HFLC_num', 'IG_num', 'IG_per', 'LY_WX', 'LY_WY', 'LY_WZ', 'LY_X', 'LY_Y', 'LY_Z', 'LYMP_num_amper', 'LYMP_per_amper', 'LYMPH_num', 'LYMPH_per', 'MO_WX', 'MO_WY', 'MO_WZ', 'MO_X', 'MO_Y', 'MO_Z', 'MONO_num', 'MONO_per', 'NE_FSC', 'NE_SFL', 'NE_SSC', 'NE_WX', 'NE_WY', 'NE_WZ', 'NEUT_num', 'NEUT_num_amper', 'NEUT_per', 'NEUT_per_amper', 'P_LCR', 'PCT', 'PDW', 'TNC_D', 'WBC_D', 'HFLC_per', 'BA_N_num', 'BA_N_per', 'HCT', 'HGB', 'MacroR', 'MCH', 'MCHC', 'MCV', 'MicroR', 'NRBC_num',
#                             'NRBC_per','PLT', 'PLT_I','RBC', 'RDW_CV', 'RDW_SD', 'TNC', 'TNC_N', 'WBC', 'WBC_N',
#                             'SURGICAL', 'CARDIAC', 'CLD', 'IVHSHUNT', 'NEC', 'SUBJECT_ID', 'SEPSIS_GROUP']

selectedColumns_heme_variables = ['BA_D_num', 'BA_D_per', 'BASO_num', 'BASO_per', 'EO_per', 'EO_num', 'HFLC_num',
                                   'IG_num', 'IG_per', 'LY_WX', 'LY_WY', 'LY_WZ', 'LY_X', 'LY_Y', 'LY_Z',
                                   'LYMP_num_amper', 'LYMP_per_amper', 'LYMPH_num', 'LYMPH_per', 'MO_WX', 'MO_WY',
                                   'MO_WZ', 'MO_X', 'MO_Y', 'MO_Z', 'MONO_num', 'MONO_per', 'NE_FSC', 'NE_SFL',
                                   'NE_SSC', 'NE_WX', 'NE_WY', 'NE_WZ', 'NEUT_num', 'NEUT_num_amper', 'NEUT_per',
                                   'NEUT_per_amper', 'P_LCR', 'PCT', 'PDW', 'TNC_D', 'WBC_D', 'HFLC_per', 'BA_N_num',
                                   'BA_N_per', 'HCT', 'HGB', 'MacroR', 'MCH', 'MCHC', 'MCV', 'MicroR', 'NRBC_num',
                                   'NRBC_per', 'PLT', 'PLT_I', 'RBC', 'RDW_CV', 'RDW_SD', 'TNC', 'TNC_N', 'WBC', 'WBC_N', 'SEPSIS_GROUP']

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

feature_importance_var = ["NE_SFL", "NE_FSC", "NE_WY", "GESTATIONAL_AGE", "MO_Y", "P_LCR"]

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
print("Black "+str(len(dataset[dataset["RACE"]=='BLACK'])))
print("American Indian/Alaskan Native" + str(len(dataset[dataset["RACE"]=='AMERICAN INDIAN OR ALASKA NATIVE'])))
print("Asian" + str(len(dataset[dataset["RACE"]=='ASIAN'])))
print("Two or more " +str(len(dataset[dataset["RACE"]=='TWO_OR_MORE'])))
print("White " +str(len(dataset[dataset["RACE"]=='WHITE'])))
print("Unknown " +str(len(dataset[dataset["RACE"]=='UNKNOWN'])))
print("Episodes " +str(len(set(dataset["EPISODE_ID"]))))


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

feature_importance_var = ["NE_SFL", "NE_FSC", "NE_WY", "GESTATIONAL_AGE", "MO_Y"]

naming_dict = {"NE_SFL": "Neutrophil Fluorescence Intensity", "NE_FSC": "Neutrophil Cell Size", "NE_WY": "Neutrophil Dispersion Width", "GESTATIONAL_AGE": "Gestational Age", "MO_Y": "Monocyte Fluorescence Intensity"}

for i in feature_importance_var:
    missing = dataset[i].isna().sum()
    overall = dataset[i].dropna()
    mean = round(stat.mean(overall), 2)
    std = round(stat.stdev(overall), 2)
    string = str(mean) + " (" + str(std) + ")"
    outcome1 = dataset[dataset['SEPSIS_GROUP']==0][i].dropna()
    mean = round(stat.mean(outcome1), 2)
    std = round(stat.stdev(outcome1), 2)
    string = str(mean) + " (" + str(std) + ")"
    outcome2 = dataset[dataset['SEPSIS_GROUP']==1][i].dropna()
    mean = round(stat.mean(outcome2), 2)
    std = round(stat.stdev(outcome2), 2)
    string = str(mean) + " (" + str(std) + ")"
    plt.figure(figsize=(8, 6))
    # plt.hist(outcome1, bins=25, alpha=0.5, label="Sepsis Negative")
    # plt.hist(outcome2, bins=25, alpha=0.5, label="Sepsis Positive")
    plt.hist(outcome1, bins=50, alpha=0.5, label="Sepsis Negative", weights=np.ones(len(outcome1)) / len(outcome1))
    plt.hist(outcome2, bins=50, alpha=0.5, label="Sepsis Positive", weights=np.ones(len(outcome2)) / len(outcome2))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.xlabel("Value", size=14)
    plt.ylabel("Percentage", size=14)
    plt.title(naming_dict[i])
    plt.legend(loc='upper right')
    plt.savefig(i+" histogram_50_bins.png")
