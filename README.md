# sepsis_ML_code
Base code used for sepsis ML analysis

Python repository for sepsis machine learning model analysis (logistic, KNN, RF, XGB, SVM)

To run analysis:
1. adjust file name in config.py and run
2. run classifiers.py to generate functions for models
3. adjust runXClassifiers.py to match desired variables, and output files
(runClassifiers.py can be used for a non cross validated, non bagged version of the models).
-to compare to clinical feature only model, adjust variables to just clinical features and result file names accordingly
4. feature_importance.py can be used to generate feature importance rankings from the random forest model

Remaining files are used for data visualization
1. Generate_Bar_Plots.py creates bar plots to compare results between two sets of models (eg clinical feature and full features)
2. Generate_curves does the same for AUC-ROC curve, and can also plot AUC-ROC curves for multiple models together
3. Generate_single_bar_plot is used to plot feature importance
4. Generate_demographics generates a "Table 1", and generate_histogram generates a demographics histogram, although these are fully based on the original dataset for this project
