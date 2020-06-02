from keras.engine import Model
from keras.applications import vgg16
from keras.layers import Dropout, Flatten, Dense
from keras import models,layers,metrics
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV,train_test_split,KFold
import pandas as pd

class CNNModel:
    __outNode=5 # 5 types of pokemon to predict
    def __init__(self,sideDIM,XTrain,yTrain,XTest,yTest):
        self.__sideDIM=sideDIM
        self.__X_train =XTrain 
        self.__y_train=yTrain 
        self.__X_test =XTest
        self.__y_test=yTest
        self.__seed=123
    

    def __createVGG16BaseModel(self):
        '''
        Using a pretained model VGG16
        Return base model layers

        importing VGG16 from keras with pre-trained weights that is trained on imagenet
        include_top set to False to exclude the top classification layer 
        weights is set to use the weights from pre-training on Imagenet
        '''
        base_model = vgg16.VGG16(weights='imagenet', 
                       include_top=False, 
                       input_shape=(self.__sideDIM,self.__sideDIM,3)) #3 = RGB Channel
        
        for layer in base_model.layers:
            layer.trainable = False
        
        return base_model.layers

    def __createTopModel(self,anOptimizer='adam',hiddenActivation='relu',outActivation='softmax',hidUnit=64):
        baseMode=self.__createVGG16BaseModel()
        model = models.Sequential(baseMode)
        model.add(Flatten())
        model.add(Dense(hidUnit, activation=hiddenActivation))
        model.add(Dense(hidUnit, activation=hiddenActivation))
        model.add(layers.Dropout(0.1))
        model.add(Dense(int(hidUnit/2), activation=hiddenActivation))
        model.add(Dense(CNNModel.__outNode, activation=outActivation)) #output layer

        model.compile(loss='binary_crossentropy',
              optimizer=anOptimizer, #'rmsprop'
            #   metrics=['accuracy']
              metrics=['accuracy',metrics.Precision(name = 'Precision'),metrics.Recall(name = 'Recall'),
              metrics.FalseNegatives(name = 'FN'),metrics.FalsePositives(name = 'FP'),
              metrics.TrueNegatives(name = 'TN'),metrics.TruePositives(name = 'TP')] 
              )
        
        return model


    def findOptimizeParamCV(self,dictParam,fold=5):
        '''
        Return 1) dataframe for all results 2) the optmized parameter in a dictionary
        3) The best accuracy score 
        4) the model object with optimized paramter in a  dictionary

        dictModelParam: a dictionary of neural network parameters that is being tested.
        '''

        resultsCols=['params','mean_test_score','std_test_score','rank_test_score']
        
        kf = KFold(n_splits=fold,shuffle= True, random_state=self.__seed)
       
        self.__kFold=kf
        newtworkModel = KerasClassifier(build_fn=self.__createTopModel, epochs=20, batch_size=25, verbose=0)
        clf = GridSearchCV(estimator = newtworkModel, param_grid=dictParam,cv=kf)
        result=clf.fit(self.__X_train,self.__y_train)
        
        dfFull=pd.DataFrame(result.cv_results_)
        dfResult=dfFull[resultsCols]
        dfResult=dfResult.sort_values(by=['rank_test_score'], ascending=True)
        dfFull=dfFull.sort_values(by=['rank_test_score'], ascending=True)

        return dfFull,dfResult,clf.best_params_,clf.best_score_,clf.best_estimator_

    def getKFold(self):
        '''
        Return kFold object which has been initialised in findOptimizeParamCV() or retun None
        '''
        return self.__kFold

    def trainModel(self, bestParam, xData,yData):

        kf=self.getKFold()
        network=self.__createTopModel(anOptimizer=bestParam.get('anOptimizer'),outActivation=bestParam.get('outActivation'),
                                hidUnit=bestParam.get('hidUnit'))
        dfMain = pd.DataFrame()
        adict={}
        count=0

        for train, test in kf.split(xData, yData):
            history=network.fit(xData[train], yData[train], epochs=bestParam.get('epochs'), 
                                batch_size=bestParam.get('batch_size'),shuffle=True,verbose=1)
            scores = network.evaluate(xData[test], yData[test], verbose=0)
            for i in range(0,len(network.metrics_names)):
                adict.update({network.metrics_names[i]:scores[i]})

            df = pd.DataFrame([adict])
            df.insert(0,'fold',count+1,True)
            count+=1
            adict.clear()
            dfMain=dfMain.append(df,ignore_index = True)
        
        return dfMain,network,history 

    
