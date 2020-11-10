import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn.model_selection import train_test_split


class SGDSolver():

    def __init__(self, path):
        #load input dataset specified in path and split data into train and validation.
        
        data = pd.read_csv(path)
        data = data.drop(['Serial No.'], axis=1)
        
        y = data['Chance of Admit '].values
        x = data.drop('Chance of Admit ', axis=1)
        #split total data into training and validation data
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05)
        
        #make into np arrays instead of DataFrame Scripts or whatever
        y_train = np.array([y_train])
        self.y_test = np.array([y_test])
        x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.x = x_train
        self.y = y_train.T
        self.bestLoss = float("inf")
        self.bestW = None
        self.bestB = None

    def training(self, alpha, lam, nepoch, epsilon):
        """Training process of linear regression will happen here. User will provide
        learning rate alpha, regularization term lam, specific number of training epoches,
        and a variable epsilon to specify pre-mature end condition,
        ex. if error < epsilon, training stops. """
        
        curAlpha = alpha[0]
        curLambda = lam[0]
        while (curAlpha <= alpha[1]):        #mult min alpha by some number until max alpha
            while (curLambda <= lam[1]):        #mult min lam by some number until max lam
                
                w = np.random.randint(-5, high=5, size=7) #initial random w's
                b = np.random.randint(-10, high=10) #initial random b
                indices = [i for i in range(0, len(self.x))]
                random.shuffle(indices)
                counter = 1
                for epoch in range(nepoch): # nepoch is bascially iterations
                    
                    b_grad = b      # giving b_grad the right shape
                    w_grad = w      # giving w_grad the right shape
                    
                    if epoch >= counter*len(self.x):
                        random.shuffle(indices)
                        counter += 1
                    # get random sample index
                    index = indices[epoch%len(self.x)]

                    # compute gradients
                    b_grad = -2.0 * (self.y[index].T - b - np.dot(w.T,self.x[index, :].T))
                    w_grad = -2.0 * (self.y[index].T - b - np.dot(w.T,self.x[index, :].T)) * self.x[index, :].T + curLambda * w

                    # update parameters
                    b = b - curAlpha * b_grad
                    w = w - curAlpha * w_grad

                    # Calculating MSE
                    loss = (np.sum(np.square(self.y.T - b - np.dot(w.T,self.x.T)))).mean()

                    #ceiling to prevent overflow
                    if(loss > 10000):
                        break
                    
                    # Adding the L2 Regularization cost 
                    L2_cost = curLambda / 2 * np.sum(w**2)
                    loss += L2_cost

                    # Comparing loss to epsilon
                    if loss < epsilon:
                        break
                
                if loss < self.bestLoss:
                    self.bestLoss, self.bestW, self.bestB = loss, w, b

                curLambda *= 1.0001
            curAlpha *= 1.0001

                    
    def testing(self, testX):
        #Uses trained weight and bias to compute the predicted y values
        #returns a n*1 y vector
        y = np.ones(testX.shape[0])
        for i in range(testX.shape[0]):
            y[i] = self.bestB + np.dot(self.bestW.T,testX[i,:])
        y = np.array([y]).T
        return y

        
#main testing
model = SGDSolver('tests/train.csv')
# Compute the time to do grid search on training
start = time.time()
model.training([10**-10, 10], [1, 1e10], 16000, 10**-8)
end = time.time()
