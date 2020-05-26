import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn import preprocessing

class Analyser():
    '''
    This class handles data plotting 
    '''
    def __init__(self):
        pass

    def histogramOrBoxPlotAnalysis(self,inDataFrame,strCols=True, boxSize=(80,80), hist=True,fileName=''):
        '''
        Analyse categorical and numerical data using histogram.
        A histogram or box plot to analyse each feature column in dataframe
        Non numerical column will be converted to numerical 

        strCols: 
        If True, will include string features in histogram/boxplot diagram converted to number using label encoding
        If False, only numerical columns shown.
        Default value is True

        boxSize:
        Size of boxplot/histogram diagram. Default is 80,80

        hist:
        If true, plot histogram, else boxplot. Default is true
        '''
        dataFrame=inDataFrame.copy()
        plt.style.use('seaborn-white')
        le=preprocessing.LabelEncoder()
        cols=5
        
        numMask=dataFrame.dtypes!=np.object
        if strCols:
            dataCols=dataFrame.columns
        else:
            dataCols=dataFrame.columns[numMask]
        rows=math.ceil(len(dataCols)/cols)
        sns.set(font_scale=3.5)  # crazy big
        fig, axes = plt.subplots(rows, cols, figsize=boxSize,sharex=False)
       
        x,y,limit=0,0,0
        for aCol in dataCols:
            if dataFrame[aCol].dtypes == np.object:
                dataFrame[aCol]=le.fit_transform(dataFrame[aCol])
            if hist:
                sns.distplot(dataFrame[aCol] , color="red", ax=axes[x, y])
                imgName=fileName+'_DistPlot'+'.png'
            else:
                sns.boxplot(y=dataFrame[aCol] , color="blue", ax=axes[x, y])
                imgName=fileName+'_BoxPlot'+'.png'
            y+=1
            limit+=1
            if y%cols ==0:
                x+=1
                y=0
            if limit>=len(dataCols):
                break
        
        plt.savefig('output/'+imgName)
        plt.close()
