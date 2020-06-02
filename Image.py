import cv2
import numpy as np
class Image():
    
    def __init__(self,imgFile,imgLblStr,imgLblInt,imgGrp):
        self.__origImg=imgFile
        self.__strLable=imgLblStr
        self.__intLable=imgLblInt
        self.imgGrp=imgGrp # Train or test image
    
    def transform(self,aSide):
        '''
        Image will be resize and changed color
        Return a list of an image
        '''
        #resize original image to square, change to RGB channel and normalize to array
        self.__imageResize = cv2.resize(self.__origImg, (aSide, aSide))
        self.__imageResize = cv2.cvtColor(self.__imageResize,cv2.COLOR_BGR2RGB)
        self.__imageFlipHoriz = cv2.flip(self.__imageResize,1)
        imglist=[self.__imageResize,self.__imageFlipHoriz]

        return imglist

    
    def getIntLable(self):
        '''
        return int lable of image
        '''
        return self.__intLable

    def getStrLable(self):
        '''
        return string lable of image
        '''
        return self.__strLable
    def getGrpType(self):
        return self.imgGrp
