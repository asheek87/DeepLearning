import os
import cv2
import numpy as np
from Image import Image
from keras.utils import to_categorical


class ImgManager():
    __folder='pokemon'
    
    __testSet=10
    __Test='Test'
    __Train='Train'
    __Validate='Validate'
    __valPrefix='val_'

    def __init__(self,testSize=0.1):
        '''
        test size: between 0.1 to 0.9. Up to 1 decimal place Default is 0.1
        '''
        self.__myImgList=[]
        self.__myProcessImgTrainList=[]
        self.__myProcessImgIntLableTrainList=[]
        self.__myProcessImgStrLableTrainList=[]
        self.__myProcessImgTestList=[]
        self.__myProcessImgIntLableTestList=[]
        self.__myProcessImgStrLableTestList=[]
        self.__myProcessImgValList=[]
        self.__myProcessImgIntLableValList=[]
        self.__myProcessImgStrLableValList=[]
        self.__dictLable={}
        self.test=testSize *ImgManager.__testSet
        self.__selectedDim=0
    
    def __imgStrtoIntLable(self,lableList):
        '''
        map string lable to int label
        Return a dictionary of string label to int lable
        '''
        print('Mapping string lable to int label images...')
        for i in range(0,len(lableList)):
            self.__dictLable.update({lableList[i]:i})
            print(lableList[i]+' :'+str(i))
        print('\n')

    def readImages(self):
        '''
        Read all images.Each image will be filtered to be a test or train data
        Return None
        '''
        for root, dirs, files in os.walk(ImgManager.__folder):
            if len(dirs)!=0:
                self.__labelFolders=dirs
                self.__imgStrtoIntLable(self.__labelFolders)
        smallest_W, smallest_H  = 1000 , 1000
        for aFolder in self.__labelFolders:
            count=1
            print('Reading images from folder '+aFolder+' ...')
            for root, dirs, files in os.walk(ImgManager.__folder+'\\'+aFolder):
                for aFile in files:                    
                    img=cv2.imread(ImgManager.__folder+'\\'+aFolder+'\\'+aFile)
                    #Define an image for validation
                    if aFile.startswith(ImgManager.__valPrefix):
                        myImg=Image(img,aFolder,self.__dictLable.get(aFolder),ImgManager.__Validate)
                    else:
                        #Define whether an img is for test or train
                        if count <= (ImgManager.__testSet -self.test):
                            myImg=Image(img,aFolder,self.__dictLable.get(aFolder),ImgManager.__Train)
                        else:
                            myImg=Image(img,aFolder,self.__dictLable.get(aFolder),ImgManager.__Test)
                        if count==ImgManager.__testSet:
                            count=1
                        else:
                            count+=1  
                    self.__myImgList.append(myImg)

                    #Find smallest width and height
                    w,h,channel=img.shape
                    if w<smallest_W:
                        smallest_W=w
                    if h<smallest_H:
                        smallest_H=h
        #saving the smallest side                
        if smallest_W<=smallest_H:
            self.__selectedDim =smallest_W
        else:
            self.__selectedDim =smallest_H
        print('Reading images from folder DONE \n')
        
    def getStrKeyFromVal(self,val):
        for key, value in self.__dictLable.items(): 
            if val == value: 
                aKey= key
                break
            else: 
                aKey=None 
        return aKey
        
    def getSideDimension(self):
        return self.__selectedDim
    
    def procesImages(self):
        '''
        Will take in original images and for each image will generate new images
        after undergoing transformation. 
        Return None
        '''
        print('Processing all images... ')

        for myImage in self.__myImgList:
            transformList=myImage.transform(self.__selectedDim)
            for item in transformList:
                if(myImage.getGrpType()==ImgManager.__Train):
                    self.__myProcessImgTrainList.append(item)
                    self.__myProcessImgIntLableTrainList.append(myImage.getIntLable())
                    self.__myProcessImgStrLableTrainList.append(myImage.getStrLable())
                elif(myImage.getGrpType()==ImgManager.__Test):
                    self.__myProcessImgTestList.append(item)
                    self.__myProcessImgIntLableTestList.append(myImage.getIntLable())
                    self.__myProcessImgStrLableTestList.append(myImage.getStrLable())
                else:
                    self.__myProcessImgValList.append(item)
                    self.__myProcessImgIntLableValList.append(myImage.getIntLable())
                    self.__myProcessImgStrLableValList.append(myImage.getStrLable())


        print('Test Images'+str(len(self.__myProcessImgTestList)))
        print('Train Images '+str(len(self.__myProcessImgTrainList)))
        print('Validation Images '+str(len(self.__myProcessImgValList)))
        print('Processing all images DONE ')
    
    def displayProcessImages(self, noOfImages=5):
        index=0
        for img in self.__myProcessImgTrainList:
            if(index<noOfImages):
                cv2.imshow("img_"+str(index),img)
            index+=1
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #TRAIN
    def __getProcessedImagesTrainList(self):
        # convert to array and normalize the RGB values
        return np.array(self.__myProcessImgTrainList,dtype="float32")/255

    def __getIntlablesTrainList(self):
        return to_categorical(np.array(self.__myProcessImgIntLableTrainList))
    
    def getStringlablesTrainList(self):
        return np.array(self.__myProcessImgStrLableTrainList)
    
    #TEST
    def __getProcessedImagesTestList(self):
        # convert to array and normalize the RGB values
        return np.array(self.__myProcessImgTestList,dtype="float32")/255
    
    def __getIntlablesTestList(self):
        #one-hot encoding on the labels
        return to_categorical(np.array(self.__myProcessImgIntLableTestList))

    def getStringlablesTestList(self):
        return np.array(self.__myProcessImgStrLableTestList)
    
    #Validation
    def __getProcessedImagesValList(self):
        # convert to array and normalize the RGB values
        return np.array(self.__myProcessImgValList,dtype="float32")/255
    
    def __getIntlablesValList(self):
        #one-hot encoding on the labels
        return to_categorical(np.array(self.__myProcessImgIntLableValList))

    def getStringlablesValList(self):
        return np.array(self.__myProcessImgStrLableValList)
    
    def get_Train_Test_Val_Data(self):
        return self.__getProcessedImagesTrainList(),self.__getIntlablesTrainList(),self.__getProcessedImagesTestList(),self.__getIntlablesTestList(),self.__getProcessedImagesValList(),self.__getIntlablesValList()






