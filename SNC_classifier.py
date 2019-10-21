# Assignment 2 skeleton code
# This code shows you how to use the 'argparse' library to read in parameters

import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from dispkernel import dispKernel

# Command Line Arguments
parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
parser.add_argument('trainingfile', help='name stub for training data and label output in csv format',default="train")
parser.add_argument('validationfile', help='name stub for validation data and label output in csv format',default="valid")
parser.add_argument('numtrain', help='number of training samples',type= int,default=200)
parser.add_argument('numvalid', help='number of validation samples',type= int,default=20)
parser.add_argument('-seed', help='random seed', type= int,default=1)
parser.add_argument('-learningrate', help='learning rate', type= float,default=0.1)
parser.add_argument('-actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'],default='linear')
parser.add_argument('-numepoch', help='number of epochs', type= int,default=50)

args = parser.parse_args()

traindataname = args.trainingfile + "data.csv"
trainlabelname = args.trainingfile + "label.csv"

print("training data file name: ", traindataname)
print("training label file name: ", trainlabelname)

validdataname = args.validationfile + "data.csv"
validlabelname = args.validationfile + "label.csv"

print("validation data file name: ", validdataname)
print("validation label file name: ", validlabelname)

print("number of training samples = ", args.numtrain)
print("number of validation samples = ", args.numvalid)

print("learning rate = ", args.learningrate)
print("number of epoch = ", args.numepoch)

print("activation function is ",args.actfunction)

#############################################################################
traindata = np.loadtxt(open("traindata.csv","rb"), delimiter=",")
trainlabel = np.loadtxt(open("trainlabel.csv","rb"), delimiter=",")
validdata = np.loadtxt(open("validdata.csv","rb"), delimiter=",")
validlabel = np.loadtxt(open("validlabel.csv","rb"), delimiter=",")

class single_neuron(object):
    def __init__(self, traindata1, trainlabel1, validdata1, validlabel1):
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.weights = [ random.uniform(0,1) for x in range(0,9)]
        self.bias = random.uniform(0,1)
        self.train_num = args.numtrain
        self.epoch_num = args.numepoch
        self.random_seed = args.seed
        self.traindata = traindata1
        self.trainlabel = trainlabel1
        self.validdata = validdata1
        self.validlabel = validlabel1
        self.validnum = args.numvalid
    def Z_calc(self,I_j):
        output = 0
        for i in range(0,9):
            output = output + self.weights[i]*I_j[i]
        return (output + self.bias)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def act_funct(self,Z):
        if args.actfunction == 'linear':
            return (Z,1)
        if args.actfunction == 'relu':
            if Z == 0:
                return (0,0)
            else:
                return (max(0,Z), (max(0,Z)/Z))
        if args.actfunction == 'sigmoid':
            y = self.sigmoid(Z)
            dy = self.sigmoid(Z)*(1-self.sigmoid(Z))
            return(y,dy)

    def main1(self):
        loss_plot = np.zeros(self.epoch_num)
        loss_plot_valid =  np.zeros(self.epoch_num)
        accuracy = np.zeros(self.epoch_num)
        accuracy_valid = np.zeros(self.epoch_num)
        for i in range(0,self.epoch_num):
            num_correct = 0.0
            num_correct_valid = 0.0
            derror_dw = np.zeros(9)
            derror_db = 0.0
            loss = 0.0
            loss_valid = 0.0

            for j in range(0,self.train_num):
                I = self.traindata[j]
                Y, dY = self.act_funct(self.Z_calc(I))
                L = self.trainlabel[j]
                derror_dw =  derror_dw + ((2*(Y-L)*I) * dY)
                derror_db = derror_db + ((2*(Y-L)) * dY)
                loss = loss + (Y-L)**2
                if (Y >0.5) and L ==1:
                    num_correct = num_correct + 1
                if (Y <= 0.5) and L == 0:
                    num_correct = num_correct + 1
             #adjusting weights
            for q in range(0,self.validnum):
                I_valid = self.validdata[q]
                Y_valid, dY_valid = self.act_funct(self.Z_calc(I_valid))
                L_valid = self.validlabel[q]
                loss_valid = loss_valid + ((Y_valid- L_valid) ** 2)
                if (Y_valid > 0.5) and L_valid == 1:
                    num_correct_valid = num_correct_valid + 1
                if (Y_valid <= 0.5) and L_valid == 0:
                    num_correct_valid = num_correct_valid + 1
            self.weights = self.weights - ((derror_dw/self.train_num)) * args.learningrate
            self.bias = self.bias - ((derror_db/self.train_num)) * args.learningrate
            accuracy[i] = num_correct/float(self.train_num)
            loss_plot[i] = loss/self.train_num
            accuracy_valid[i] = num_correct_valid /float(self.validnum)
            loss_plot_valid[i] = loss_valid / self.validnum
        x= np.arange(1,self.epoch_num+1,1)
        print("The following is the training accuracy over 200 epochs:")
        print(accuracy)
        print("The following is the validation accuracy over 200 epochs:")
        print(accuracy_valid)
        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.plot(x, loss_plot)
        plt.plot(x, loss_plot_valid)
        plt.legend(['Training Data','Validation Data'])
        plt.subplot(1, 2, 2)
        plt.ylim((0,1.1))
        plt.title('Accuracy')
        plt.plot(x, accuracy)
        plt.plot(x, accuracy_valid)
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend(['Training Data','Validation Data'])
        plt.show()
        dispKernel(self.weights,3,60)

NN = single_neuron(traindata, trainlabel, validdata, validlabel)

NN.main1()

