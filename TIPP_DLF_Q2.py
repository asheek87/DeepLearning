#%%
from ImgManager import ImgManager
from CNNModel import CNNModel
from Analyser import Analyser
import warnings
import pandas as pd
import time
import numpy as np
output2='output_Q2/'
warnings.filterwarnings('ignore')
#%%
imgMan=ImgManager(testSize=0.2)
imgMan.readImages()

# %%
imgMan.procesImages()

# %%
# imgMan.displayProcessImages()
#%%
X_Train,y_train,X_Test,y_test,X_Val,y_Val=imgMan.get_Train_Test_Val_Data()
print(X_Train.shape)
print(y_train.shape)
print(X_Test.shape)
print(y_test.shape)
print(X_Val.shape)
print(y_Val.shape)


# %%
cnnMod=CNNModel(imgMan.getSideDimension(),X_Train,y_train,X_Test,y_test)
#%%
#Find the best hyper paramters to get best results
epoch=[5,10]
batSize=[25,20] 
optimizers=['rmsprop','adam']
outAct=['softmax','sigmoid']
hiddenUnit=[256,128]

dictParam={'epochs':epoch,'batch_size':batSize,'anOptimizer':optimizers,'outActivation':outAct,'hidUnit':hiddenUnit}
start=time.time()
df_full,df_result,bestParam,bestScore,model=cnnMod.findOptimizeParamCV(dictParam,fold=2)
end=time.time()
# %%
print('Time taken for grid search is '+str(start-end)+" seconds")
# %%
#Print full results to output_Q2/DF_Full_Result.xlsx
df_full.to_excel(output2+"DF_Full_Result.xlsx")
df_full
# %%
# %%
# Show the best parameter to be used after grid search
bestParam
df_param=pd.DataFrame([bestParam])
df_param
# %%
#Print partial results to output_Q2/DF_Partial_Result.xlsx
df_result.to_excel(output2+"DF_Partial_Result.xlsx")
df_result.head()
# %%
# Show the best score after grid search
print('Best accuracy after grid search on training data: '+str(bestScore))
# %%
# Evaluating the best model found in grid search using Test data
res=model.score(X_Test,y_test)
print('Accuracy of grid search model on test data: '+str(res))

#%%
# Train new model with best parameters using full data set
start=time.time()
df,nw,hist=cnnMod.trainModel(bestParam,X_Train,y_train)
end=time.time()
#%%
print('Time taken for training model is '+str(end-start)+" seconds")
# %%
#Show mertrics after training with best parameters
df
#%%
param= nw.evaluate(X_Test, y_test,batch_size=bestParam.get('batch_size'))
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
analyser=Analyser(output2)
analyser.plot_loss(hist,'Loss')  
#%%
analyser.plot_accuracy(hist,'Accuracy') 
#%%
# Validate model using validation img
# %matplotlib inline
import matplotlib.pyplot as plt

strLableList=imgMan.getStringlablesValList()
index=0
wrong=0
right=0
for anImage in X_Val:
    actual = strLableList[index]
    anImageExpand = np.expand_dims(anImage, axis=0)
    prob = nw.predict(anImageExpand)
    predictIndx = np.argmax(prob)
    predictStr=imgMan.getStrKeyFromVal(predictIndx)
    if actual !=predictStr:
        wrong=+1
    if actual ==predictStr:
        rignt=+1
        # plt.imshow(anImage)
        # plt.show()
        # print('Actual img is: '+ actual)
        # print('Probability:'+ str(predictIndx) )
        # print('Predicted img is '+predictStr +'\n')
    index+=1
print('Wrong '+str(wrong))
print('Right '+str(right))


#%%
import pickle

data = [X_Test,y_test,X_Train,y_train,X_Val,y_Val]
with open(output2+'Q2_Data.pickle', 'wb+') as out_file:
    pickle.dump(data, out_file)

nw.save(output2+"Q2_cNN_model.h5") 