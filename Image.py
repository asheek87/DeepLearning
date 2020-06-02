import cv2
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
        #resize original image to square
        self.__imageResize = cv2.resize(self.__origImg, (aSide, aSide))
        self.__imageResize = cv2.cvtColor(self.__imageResize,cv2.COLOR_BGR2RGB)
        # #color change
        # self.__imgColorChange = cv2.cvtColor(self.__imageResize,cv2.COLOR_BGR2RGB)
        # #blur image
        # ksize = (10, 10)
        # self.__imgBlur = cv2.blur(self.__imageResize, ksize)

        # # flipcode = 0: flip vertically
        # # flipcode > 0: flip horizontally
        # # flipcode < 0: flip vertically and horizontally
        # # flip resize image vertically and horizontally
        # self.__imgFlip_vert= cv2.flip(self.__imageResize,0)
        # self.__imgFlip_Horiz= cv2.flip(self.__imageResize,1)
        # self.__imgFlip_vertHoriz= cv2.flip(self.__imageResize,-1)
        # #rotate image
        # self.__imgRotate90CC=cv2.rotate(self.__imageResize,cv2.ROTATE_90_COUNTERCLOCKWISE)
        # self.__imgRotate90C=cv2.rotate(self.__imageResize,cv2.ROTATE_90_CLOCKWISE)
        # self.__imgRotate180=cv2.rotate(self.__imgBlur,cv2.ROTATE_180)
        # self.__imgRotate180CChnage = cv2.cvtColor(self.__imgColorChange,cv2.COLOR_BGR2RGB)
 
        # imglist=[self.__imageResize,self.__imgColorChange,self.__imgBlur,
        # self.__imgFlip_vert,self.__imgFlip_Horiz,self.__imgFlip_vertHoriz,
        # self.__imgRotate90CC,self.__imgRotate90C,self.__imgRotate180,self.__imgRotate180CChnage
        # ]
        imglist=[self.__imageResize]

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
