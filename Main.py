#%%
import warnings
import pandas as pd
from sklearn import preprocessing
from Data import DataManager
from Analyser import Analyser
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns', 51)
output='output/'
size=(150,100)
warnings.filterwarnings('ignore')
# %%
# 1. Read from Dataset
dataManager=DataManager()
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

analyser=Analyser()
# Display Histogram.To check general data distibution on numrical data after unskew. File output is at \output\01_INITIAL_DistPlot.png
analyser.histogramOrBoxPlotAnalysis(dfFullData,strCols=True,hist=True,boxSize=size,fileName='01_INITIAL')
# %%
# Display Boxplot.To check on outliers on numrical data which has been scaled. File output is at \output\02_OUTLIER_BoxPlot.png
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
dfskew.to_excel(output+"DF_Skew.xlsx")
dfskew
# Display Histogram.To check general data distibution on numrical data after unskew. File output is at \output\04_SKEW_DistPlot.png
analyser.histogramOrBoxPlotAnalysis(dfskew,strCols=False,hist=True,boxSize=size,fileName='04_SKEW')

#%%
# 3.4 Apply encoding on dataset.
# OHE applied on categorical data which has more the 2 values
# LE applied on categorical data which has the 2 values
df_le=dataManager.applyEncodingToNonNumericData(dfskew)
#Display One hot encoding table
df_le
# Display Histogram.To check general data distibution on numrical data after OHE. File output is at \output\05_OHE_DistPlot.png
analyser.histogramOrBoxPlotAnalysis(df_le,strCols=True,hist=True,boxSize=size,fileName='05_LE')
#%%
# 3.5 Perform scaling on encoded data.
# Perform scaling on encoded data
df_Scale_le =dataManager.scaleData(df_le)
# Display one hot encoded data which has been scaled
df_Scale_le.to_excel(output+"DF_Scale_LE.xlsx")
df_Scale_le
# Display Histogram.To check general data distibution on all data after  scaling.File output is at \output\06_SCALE_DistPlot.png
analyser.histogramOrBoxPlotAnalysis(df_Scale_le,strCols=True,hist=True,boxSize=size,fileName='06_SCALE')

#%%
# 3.6 Remove outlier data.
# Remove outliers on dataframes
# Base on boxplot, there are outliers in data frame
df_noOutlier_ohe =dataManager.removeOutlier(df_Scale_le)
df_noOutlier_ohe.to_excel(output+"DF_NoOutlier_LE.xlsx")
#%%
# Display Histogram.To check general data distibution on all data 
# after  outlier removed.File output is at \output\
analyser.histogramOrBoxPlotAnalysis(df_noOutlier_ohe,strCols=True,hist=True,boxSize=size,fileName='07_NO_OUTLIER')
#%%
# 3.7 Analyse Correlation between features and remove highly correlated featuers.
#Correlation between features with athreshold pf 90%
df_corr_ohe=dataManager.showCorr(df_noOutlier_ohe,0.90)
df_corr_ohe
#%%
# Base on correlation table, some features  has 
# high correlation.Will have to drop some of them before running model, 

dropColslist=['radius_mean','perimeter_mean','radius_worst','area_mean','radius_se','area_worst','perimeter_se','concavity_mean','texture_mean','concave points_mean']
df_Final_ohe=dataManager.dropUnnecessaryColumns(df_noOutlier_ohe,dropColslist)
df_Final_ohe.to_excel(output+"DF_Final_LE.xlsx")
df_Final_ohe
#%%