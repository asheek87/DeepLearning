
from sklearn import preprocessing,model_selection
from sklearn.model_selection import StratifiedKFold, GridSearchCV,train_test_split,KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models,layers,metrics
import numpy.random as nr
import pandas as pd
import numpy as np
nr.seed(123)

class ModelManager():
    '''
    This class manages all Neural network related processes.
    '''
    def __init__(self, dataFrame,targetColsList,seed):
        self.splitVal=0.2
        self.__df=dataFrame
        self.__seed=seed
        self.__targetColsList=targetColsList
        self.__targetLableLength=len(targetColsList)
        self.__kFold=None
        self.__splitData()
       
    
    def __splitData(self):
        '''
        Split Data into Train and test
        Test size is 20%
        '''
        self.__feature_cols = [x for x in self.__df.columns if x not in self.__targetColsList]
        self.__X_data = np.array(self.__df[self.__feature_cols])
        self.__y_data = np.array(self.__df[self.__targetColsList])
        print('Overall Feature shape: {}'.format(self.__X_data.shape))
        print('Overall Target shape: {}'.format(self.__y_data.shape))
        #split Data for Training and Test
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X_data, self.__y_data,
                                                    stratify=self.__y_data,random_state=self.__seed, 
                                                    test_size=self.splitVal)

        print('Train Feature shape: {}'.format(self.__X_train.shape))
        print('Train Target shape: {}'.format(self.__y_train.shape))
        print('Test Feature shape: {}'.format(self.__X_test.shape))
        print('Test Target shape: {}'.format(self.__y_test.shape))
    # Create function returning a compiled network
    def create_network(self,anOptimizer='adam',outActivation='softmax',hidActivation='relu',hidUnit=64):
        '''
        Creating a neural network
        '''
        # Start neural network
        network = models.Sequential()
        # Add 3 fully connected layer with a ReLU activation function and 1 droput layer to prevent overfitting
        network.add(layers.Dense(units=hidUnit, activation=hidActivation, input_shape=(len(self.__feature_cols),)))
        network.add(layers.Dense(units=hidUnit, activation=hidActivation))
        network.add(layers.Dropout(0.1))
        network.add(layers.Dense(units=hidUnit, activation=hidActivation))

        # Add fully connected layer with a sigmoid activation function
        # Use a single node in output layer when our lable is a single column with binary values
        if self.__targetLableLength==1:
            network.add(layers.Dense(units=1, activation=outActivation))
        else:
            network.add(layers.Dense(units=self.__targetLableLength, activation=outActivation))

        # Compile neural network
        network.compile(loss='binary_crossentropy', # Use binary as target label has  0 or 1 values
                    optimizer=anOptimizer,
                    metrics=['accuracy',metrics.Precision(name = 'Precision'),metrics.Recall(name = 'Recall'),
                    metrics.FalseNegatives(name = 'FN'),metrics.FalsePositives(name = 'FP'),
                    metrics.TrueNegatives(name = 'TN'),metrics.TruePositives(name = 'TP')] 
                    ) # Accuracy performance metric
    
        # Return compiled network
        return network
    

    def getTestAndTrainData(self):
        '''
        Return testing and training data. Each is a list object
        X_test,y_test,X_train,y_train
        '''
        return self.__X_test,self.__y_test,self.__X_train,self.__y_train

    def getKFold(self):
        '''
        Return kFold object which has been initialised in findOptimizeParamCV() or retun None
        '''
        return self.__kFold
        

    def findOptimizeParamCV(self,dictParam,fold=5):
        '''
        Return 1) dataframe for all results 2) the optmized parameter in a dictionary
        3) The best accuracy score 
        4) the model object with optimized paramter in a  dictionary

        dictModelParam: a dictionary of neural network parameters that is being tested.
        '''

        resultsCols=['params','mean_test_score','std_test_score','rank_test_score']
        
        #define the cv
        if self.__targetLableLength==1:
            kf = StratifiedKFold(n_splits=fold,shuffle= True,random_state=self.__seed)
        else:
            kf = KFold(n_splits=fold,shuffle= True, random_state=self.__seed)
       
        self.__kFold=kf
        newtworkModel = KerasClassifier(build_fn=self.create_network, epochs=20, batch_size=25, verbose=0)
        clf = GridSearchCV(estimator = newtworkModel, param_grid=dictParam,cv=kf)
        result=clf.fit(self.__X_train,self.__y_train)


        dfFull=pd.DataFrame(result.cv_results_)
        dfResult=dfFull[resultsCols]
        dfResult=dfResult.sort_values(by=['rank_test_score'], ascending=True)
        dfFull=dfFull.sort_values(by=['rank_test_score'], ascending=True)

        return dfFull,dfResult,clf.best_params_,clf.best_score_,clf.best_estimator_
    
    def trainModel(self, bestParam, xData,yData):

        kf=self.getKFold()
        network=self.create_network(anOptimizer=bestParam.get('anOptimizer'),outActivation=bestParam.get('outActivation'),
                                hidUnit=bestParam.get('hidUnit'))
        dfMain = pd.DataFrame()
        adict={}
        count=0
        underscore="_"
        for train, test in kf.split(xData, yData):
            history=network.fit(xData[train], yData[train], epochs=bestParam.get('epochs'), 
                                batch_size=bestParam.get('batch_size'),shuffle=True,verbose=0)
            scores = network.evaluate(xData[test], yData[test], verbose=0)
            for i in range(0,len(network.metrics_names)):
                adict.update({network.metrics_names[i]:scores[i]})

            df = pd.DataFrame([adict])
            df.insert(0,'fold',count+1,True)
            count+=1
            adict.clear()
            dfMain=dfMain.append(df,ignore_index = True)
        
        return dfMain,network,history 




        

    


