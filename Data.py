import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
class DataManager():
    '''
    This  class handles all data preocessing steps
    On class creation, have to specify whether to apply one hot encoding on dataset
    '''
    seedValue=123
    __separator=','
    __csvFileAllData='dataset_breastCancer.csv'
    __tgtCol = 'diagnosis'
    __tgtCol_M = 'diagnosis_M'
    __tgtCol_B = 'diagnosis_B'
    __dataFolder ='data'

    def __init__(self, applyOHE):

        self.applyOHE=applyOHE
         
    def getTargetLable(self):
        '''
        Return a list of 2 target lables when ohe hot encoding is applied
        else return a list of a target lable
        '''
        if self.applyOHE:
            return [DataManager.__tgtCol_M,DataManager.__tgtCol_B]
        else:
            return [DataManager.__tgtCol]

    def readData(self):
        '''
        Returns full  dataset as a Panda Frame
        '''
        for root, dirs, files in os.walk(DataManager.__dataFolder):
            for afile in files:
                if afile == DataManager.__csvFileAllData:
                    fullDataPath=os.path.join(root,afile)
                    dfFullData = pd.read_csv(fullDataPath, sep=DataManager.__separator)
                    dfFullData.describe()       
        return dfFullData
    
    
    def dropEmptyData(self,inDataFrame,replace=None):
        '''
        Return a dataframe after  empty cell is dropped
        'unknown' is missing value  in a record. 
        Replace with empty cell and remove them from DataFrame
        '''
        dataFrame=inDataFrame.copy()
        print('Dropping datapoints with missing entries...')
        oldR=dataFrame.shape[0]
        if(replace==None):
            dataFrame = dataFrame.dropna()
        else:
            dataFrame = dataFrame.replace(replace.lower(),np.NaN)
            dataFrame = dataFrame.dropna()
        print('Dropped datapoints = {}'.format(oldR-dataFrame.shape[0]))
        dataFrame.reset_index(drop = True, inplace = True)
        return dataFrame

    def dropDuplicateData(self,inDataFrame):
        '''
        Return a dataframe of data duplicates dropped
        '''   
        #Find duplicates,keep latest and remove the duplicates
        dataFrame=inDataFrame.copy()
        print('Dropping duplicated datapoints..')
        oldR=dataFrame.shape[0]
        dataFrame.drop_duplicates(keep='last',inplace=True)
        print('Dropped datapoints = {}'.format(oldR-dataFrame.shape[0]))
        dataFrame.reset_index(drop = True, inplace = True)
        return dataFrame

    def checkSkew(self,inDataFrame, skewThreshold=0.0):
        '''
        Return a dataframe of data after skewing is fix
        If a feature has skew value greater than threshold, it will be skewed
        '''
        dataFrame=inDataFrame.copy()
        mask = inDataFrame.dtypes != np.object
        nonStrCols=inDataFrame.columns[mask]
        skewCols=dataFrame[nonStrCols].skew().sort_values(ascending=False)
        skewCols = np.abs(skewCols)
        print(skewCols)
        # Define acceptable skew from 0 to 'skewVal'. 
        # If a feature exceeds them that column has to be skewed
        skewCols = skewCols.loc[skewCols >= skewThreshold] 
        for col in skewCols.index.tolist():
            dataFrame[col] = np.log1p(dataFrame[col])
        dataFrame.reset_index(drop = True, inplace = True)
        return dataFrame

    def applyEncodingToNonNumericData(self,inDataFrame):
        '''
        Convert non numeric data to numeric. Encoding done on non numeric label
        Returnlable encoded Data Frame, df_le if one hot encoding is not applied, else Return one hot encoed data frame
        
        df_le  : with only label encoding applied to all non numeric colums
        df_ohe : with only one hot encoding applied to all non numeric colums
        
        '''
        df_le=inDataFrame.copy()
        df_ohe=inDataFrame.copy()
        le=preprocessing.LabelEncoder()
        mask = inDataFrame.dtypes == np.object
        #All string columns
        strCols = inDataFrame.columns[mask]
        #All string columns which has only 2 values
        for aCol in strCols:
            # using one hot encoding
            if self.applyOHE:
                dummy=pd.get_dummies(df_ohe[aCol],prefix=aCol)
                df_ohe=pd.concat([df_ohe,dummy],axis=1)
                df_ohe=df_ohe.drop(aCol,axis=1)
            else:
                #  feature has 2 only values. So just use label encoder
                df_le[aCol]=le.fit_transform(df_le[aCol])
        
        if self.applyOHE:
            df_ohe.reset_index(drop = True, inplace = True)
            return df_ohe
        else:
            df_le.reset_index(drop = True, inplace = True)
            return df_le

    
    def scaleData(self, inDataFrame):
        '''
        Return a dataframe of scaled data.
        Scale data using minmax scaler
        '''
        dataFrame=inDataFrame.copy()
        #scale data
        print('Scaling data using MinMax scaler...') 
        scaler = preprocessing.MinMaxScaler()
        scaledf = scaler.fit_transform(dataFrame)
        dataFrame = pd.DataFrame(scaledf,columns=dataFrame.columns)
        #After scaling convert back Target label to int
        if self.applyOHE:
            dataFrame[DataManager.__tgtCol_M] = dataFrame[DataManager.__tgtCol_M].astype(np.int64)
            dataFrame[DataManager.__tgtCol_B] = dataFrame[DataManager.__tgtCol_B].astype(np.int64)          
        else:
            dataFrame[DataManager.__tgtCol] = dataFrame[DataManager.__tgtCol].astype(np.int64)

        dataFrame.reset_index(drop = True, inplace = True)
        return dataFrame
    
    
    def removeOutlier(self,inDataFrame):
        '''
        remove outlier which is not within 3 std deviation
        '''
        dataFrame=inDataFrame.copy()
        z_scores = stats.zscore(dataFrame)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        dataFrame = dataFrame[filtered_entries]
        dataFrame.reset_index(drop = True, inplace = True)
        return dataFrame

    def showCorr(self,inDataFrame,threshold=0.8):
        '''
        Show correlation between feature variables
        Threshold for high correlation is  > 0.8
        '''
        thrsVal=str(threshold)
        dataFrame=inDataFrame.copy()
        dataFrame.reset_index(drop = True, inplace = True)
        # Calculate the correlation values
        feature_cols = dataFrame.columns[:-1]
        corr_values = dataFrame[feature_cols].corr()

        # Simplify by emptying all the data below the diagonal
        tril_index = np.tril_indices_from(corr_values)

        # Make the unused values NaNs
        for coord in zip(*tril_index):
            corr_values.iloc[coord[0], coord[1]] = np.NaN
    
        # Stack the data and convert to a data frame
        corr_values = (corr_values.stack()
                      .to_frame()
                      .reset_index()
                      .rename(columns={'level_0':'feature1','level_1':'feature2',0:'correlation'}))

        # Get the absolute values for sorting
        corr_values['abs_correlation'] = corr_values.correlation.abs()
        # The most highly correlated values
        return corr_values.sort_values('correlation', ascending=False).query('abs_correlation>'+thrsVal)

    def dropUnnecessaryColumns(self,inDataFrame,colsList):
        print('Dropping datapoints...')
        dataFrame=inDataFrame.copy()
        oldR=dataFrame.shape[0]
        dataFrame.drop(colsList, axis = 1,inplace=True)
        dataFrame.reset_index(drop = True, inplace = True) 
        print('Dropped datapoints = {}'.format(oldR-dataFrame.shape[0]))
        dataFrame.reset_index(drop = True, inplace = True)
        return dataFrame