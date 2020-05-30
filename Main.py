#%%
import warnings
import pandas as pd
from sklearn import preprocessing
from Data import DataManager
from Model import ModelManager
from Analyser import Analyser
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns', 51)
output1='output_Q1/'
size=(150,100)
warnings.filterwarnings('ignore')
applyOHE=False
# %%
# 1. Read from Dataset
dataManager=DataManager(applyOHE)
dfFullData=dataManager.readData()
#%%
# 2. Analysing Data from Dataset
dfFullData.describe()
#%%
dfFullData.info()
#%%
# Base on displot 01_INITIAL_DistPlot.png, boxplot 02_OUTLIER_BoxPlot.png, data information above and the original datset:
# 1.some feature in dataset does not have normal distibution thus has to be skewed,
# 2.The true label, 'diagnosis' has binary values and the ratio of Yes to No is disproportionate thus stratification has to be done.
# 3.The range of numercal values in some features is wide thus has to be scaled down.
# 4.The true label, 'diagnosis' has to be converted to numbers via label encoding since it has only 2 values
# 5.There are some features with outliers, thus outliers has to be removed
# 6.Features 'ID' and 'Unnamed' has to be dropped as they are not useful
# 7. There are no empty cells
# In addition, the follwing steps also has to be checked:

# 7.Check for duplicates
# 8.Check for high correlation between features

analyser=Analyser(output1)
# Display Histogram.To check general data distibution on numrical data after unskew. File output is at \output_Q1\01_INITIAL_DistPlot.png
analyser.histogramOrBoxPlotAnalysis(dfFullData,strCols=True,hist=True,boxSize=size,fileName='01_INITIAL')
# %%
# Display Boxplot.To check on outliers on numrical data which has been scaled. File output is at \output_Q1\02_OUTLIER_BoxPlot.png
analyser.histogramOrBoxPlotAnalysis(dfFullData,strCols=True,hist=False,boxSize=size,fileName='02_OUTLIER')
# %%
# 3. Data Pre Process
# 3.1 Remove unnecessary features
# Drop cells which is not useful in classification
dropColslist=['id','Unnamed: 32']
df_drop_idUnamed=dataManager.dropUnnecessaryColumns(dfFullData,dropColslist)
df_drop_idUnamed.info()
# %%
# 3.2 Check duplicated records
# Check for duplicates. There are no duplicates
duplicateRowsDF=df_drop_idUnamed[df_drop_idUnamed.duplicated()]
duplicateRowsDF

#%%
# 3.3 Check for skewed data and try to normalize records
# Check for skewed data in numerical data and process skewed data to normalize it
dfskew=dataManager.checkSkew(df_drop_idUnamed)
dfskew.to_excel(output1+"DF_Skew.xlsx")
dfskew
# Display Histogram.To check general data distibution on numrical data after unskew. File output is at \output_Q1\04_SKEW_DistPlot.png
analyser.histogramOrBoxPlotAnalysis(dfskew,strCols=False,hist=True,boxSize=size,fileName='04_SKEW')

#%%
# 3.4 Apply encoding on dataset.
# OHE or LE applied on non numeric data
df_encode=dataManager.applyEncodingToNonNumericData(dfskew)
#Display One hot encoding table
df_encode
# Display Histogram.To check general data distibution on numrical data after OHE. File output is at \output_Q1\05_Encode_DistPlot.png
analyser.histogramOrBoxPlotAnalysis(df_encode,strCols=True,hist=True,boxSize=size,fileName='05_Encode')

#%%
# 3.5 Perform scaling on encoded data.
# Perform scaling on encoded data
df_Scale =dataManager.scaleData(df_encode)
# Display one hot encoded data which has been scaled
df_Scale.to_excel(output1+"DF_Scale.xlsx")
df_Scale
# Display Histogram.To check general data distibution on all data after  scaling.File output is at \output_Q1\06_SCALE_DistPlot.png
analyser.histogramOrBoxPlotAnalysis(df_Scale,strCols=True,hist=True,boxSize=size,fileName='06_SCALE')

#%%
# 3.6 Remove outlier data.
# Remove outliers on dataframes
# Base on boxplot, there are outliers in data frame
df_noOutlier =dataManager.removeOutlier(df_Scale)
df_noOutlier.to_excel(output1+"DF_NoOutlier.xlsx")
#%%
# Display Histogram.To check general data distibution on all data 
# after  outlier removed.File output is at \output_Q1\
analyser.histogramOrBoxPlotAnalysis(df_noOutlier,strCols=True,hist=True,boxSize=size,fileName='07_NO_OUTLIER')
#%%
# 3.7 Analyse Correlation between features and remove highly correlated featuers.
#Correlation between features with athreshold pf 90%
df_corr_ohe=dataManager.showCorr(df_noOutlier,0.90)
df_corr_ohe
#%%
# Base on correlation table, some features  has 
# high correlation.Will have to drop some of them before running model, 

dropColslist=['radius_mean','perimeter_mean','radius_worst','area_mean','radius_se','area_worst','perimeter_se','concavity_mean','texture_mean','concave points_mean']
df_Final=dataManager.dropUnnecessaryColumns(df_noOutlier,dropColslist)
df_Final.to_excel(output1+"DF_Final.xlsx")
df_Final
#%%
modMan=ModelManager(df_Final,dataManager.getTargetLable(),dataManager.seedValue)
#%%
#Find the best hyper paramters to get best results
epoch=[23,20]
batSize=[20,15] 
optimizers=['rmsprop','adam']
outAct=['softmax','sigmoid']
hiddenUnit=[256,128]

dictParam={'epochs':epoch,'batch_size':batSize,'anOptimizer':optimizers,'outActivation':outAct,'hidUnit':hiddenUnit}
df_full,df_result,bestParam,bestScore,model=modMan.findOptimizeParamCV(dictParam,fold=3)
#%%
#Print full results to output_Q1/DF_Full_Result.xlsx
df_full.to_excel(output1+"DF_Full_Result.xlsx")

# %%
# Show the best parameter to be used after grid search
bestParam
df_param=pd.DataFrame([bestParam])
df_param
# %%
#Print partial results to output_Q1/DF_Partial_Result.xlsx
df_result.to_excel(output1+"DF_Partial_Result.xlsx")
df_result.head()
# %%
# Show the best score after grid search
print('Best accuracy after grid search on training data: '+str(bestScore))
# %%
# Evaluating the best model found in grid search using Test data
X_test,y_test,X_train,y_train=modMan.getTestAndTrainData()
res=model.score(X_test,y_test)
print('Accuracy of grid search model on test data: '+str(res))

#%%
# Train new model with best parameters using full data set
df,nw,hist=modMan.trainModel(bestParam,X_train,y_train)
# %%
#Show mertrics after training with best parameters
df
#%%
param= nw.evaluate(X_test, y_test,batch_size=bestParam.get('batch_size'))
#%%
print('Eval test loss:', param[0])
print('Eval test accuracy:', param[1]*100)
print('Eval test precision:', param[2]*100)
print('Eval test recall:', param[3]*100)
print('Eval test false negative:', param[4])
print('Eval test false positive:', param[5])
print('Eval test true negative:', param[6])
print('Eval test true positive:', param[7])

#%%
analyser.plot_loss(hist,'Loss')  
#%%
analyser.plot_accuracy(hist,'Accuracy') 
#%%
import pickle

data = [X_test,y_test,X_train,y_train]
with open(output1+'Q1_Data.pickle', 'wb+') as out_file:
    pickle.dump(data, out_file)

nw.save(output1+"Q1_ANN_model.h5") 

