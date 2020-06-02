import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn import preprocessing

class Analyser():
    '''
    This class handles data plotting 
    '''
    def __init__(self,output1):
        self.__output1=output1


    def plot_loss(self,hist,fileName=''):
        plt.style.use('seaborn-white')
        # train_loss = hist.history['loss']
        test_loss = hist.history['loss']
        epochX = list(range(1, len(test_loss) + 1))
        plt.plot(epochX, test_loss, color = 'red', label = 'test loss')
        # plt.plot(epochX, train_loss, label = 'traning loss')

        plt.xlabel('Epoch', fontsize=15)
        plt.tick_params(axis='x', labelsize=14)
        plt.ylabel('Loss',fontsize=15)
        plt.tick_params(axis='y', labelsize=14)
        plt.title('Loss vs. Epoch',fontsize=15)
        plt.legend(fontsize=10)
        plt.savefig(self.__output1+fileName)


    
    def plot_accuracy(self,hist,fileName=''):
        plt.style.use('seaborn-white')
        # train_acc = hist.history['accuracy']
        test_acc = hist.history['accuracy']
        epochX = list(range(1, len(test_acc) + 1))
        plt.plot(epochX, test_acc, color = 'red', label = 'test accuracy')
        # plt.plot(epochX, train_acc, label = 'training accuracy')  
        plt.xlabel('Epoch',fontsize=15)
        plt.tick_params(axis='x', labelsize=14)
        plt.ylabel('Accuracy',fontsize=15)
        plt.tick_params(axis='y', labelsize=14)
        plt.title('Accuracy vs. Epoch',fontsize=15)  
        plt.legend(loc='lower right',fontsize=10)
        plt.savefig(self.__output1+fileName)