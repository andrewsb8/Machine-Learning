"""
Another version of the general logistic regression algorithm using gradient
descent for parameter convergence.  This script utilizes stratified cross
validation in place of splitting the data set in question into a training
and testing set.

Author: Brian Andrews
"""

import sys
import math
import xlrd
from xlrd import open_workbook
import numpy as np
#import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#***************************FUNCTIONS*******************************************
#function that copies the excel sheet to a table
def store(colnum, rownum, case, data, table = [[]]):
    numFromString = 0
    check = False
    for i in range(case): #loops traversing xlsx files
        for j in range(data):
            if j == data-1:
                table[i,j] = table[i,11]-table[i,10] #difference in pachy min and pachy apex
            elif j == data-2: #add spherical equivalence MRx error
                table[i,j] = table[i,7] + (table[i,8]/2) #cols 7 and 8 have spherical and cylindrical mrx data
            elif mainSheet.cell_value(rownum-1, colnum+j) == "DateOfBirth": #need to edit date of birth column
                date = datetime(*xlrd.xldate_as_tuple(mainSheet.cell_value(rownum+i, colnum+j), 0))
                dateStr = str(date) #change the date to a string
                table[i,j] = 2017 - strToFloat(dateStr)
            else:    
                #check if there are strings
                check = isinstance(mainSheet.cell_value(rownum+i,colnum+j), str)
                if check == False: #if entry is not a string (therefore, an int), add to table
                    table[i,j] = mainSheet.cell_value(rownum+i, colnum+j)
                if check == True: #if entry is a string
                    numFromString = strToFloat(mainSheet.cell_value(rownum+i, colnum+j)) #strip it and take first number 
                    table[i,j] = numFromString #add new float to table
                    check = False #make check False again so the loop works
    return table

#function returns the average of a list/array
def average(array = []):
    sum1 = 0
    for h in range(len(array)):
        sum1 += array[h]
    av = sum1/len(array)
    return av

#converts string to float and only takes first number as an argument
def strToFloat(dat):
    newNum = 0
    newStr = ""
    for i in range(len(dat)): #takes the number before the first space (want magnitude of strain not coordinates...for now)
        if dat[i] == ' ':
            break
        if (dat[i] == '-' and i != 0):
            break
        newStr += dat[i] #put the number into a new string
    newNum = float(newStr) #make it a float
    return newNum

"""
This next block of functions relate solely to the estimation of the parameters
for a logistic regression model.
"""
#this function gives the result of the sigmoid function
def logistic_func(z):
   if z < 0:
       pr_y = float(1 - 1/(1 + math.exp(z)))
   if z >=0:
       pr_y = float(1.0 / float((1.0 + math.exp(-1.0*z))))
   return pr_y

#calculate the exponent to be used in the sigmoid above
def calc_hypothesis(theta, x):
   z = 0
   for i in range(len(theta)):
      z += x[i]*theta[i]
   return logistic_func(z)

#error or cost function 
def Cost_Function(X,Y,theta,m):
   sumOfErrors = 0
   for i in range(m):
      xi = X[i]
      hi = calc_hypothesis(theta,xi)
      if Y[i] == 1:
         error = Y[i] * math.log(hi)
      elif Y[i] == 0:
         error = (1-Y[i]) * math.log(1-hi)
      sumOfErrors += error
   const = -1/m
   J = const * sumOfErrors
   return J

#error or cost function with respect to the partial derivative.
#this cost dictates the change the model parameters
def partial_derivate_cost(X,Y,theta,j,m,alpha):
   errors_sum = 0
   for i in range(m):
      xi = X[i]
      xij = xi[j]
      hi = calc_hypothesis(theta,X[i])
      error = (hi - Y[i])*xij # partial derivative w.r.t xij
      errors_sum += error
   m = len(Y)
   constant = float(alpha)/float(m)
   J = constant * errors_sum
   return J

#gradient descent is the algorithm method for finding the global minima
def gradient_descent(X,Y,theta,m,alpha,lambdaf):
   theta_new = []
   for pos_i in range(len(theta)):
      CFDerivative = partial_derivate_cost(X,Y,theta,pos_i,m,alpha)
      updated_theta = theta[pos_i] - CFDerivative - 2*lambdaf*theta[pos_i]
      theta_new.append(updated_theta)
   return theta_new

#method to start model creation
def logistic_regression(X,Y,alpha,theta,num_iters,lambdaf):
   m = len(Y)
   maxr2 = -100000
   minCost = 200
   best_theta = []
   best_theta2 = []
   iter_found = 0
   for i in range(num_iters):
      new_theta = gradient_descent(X,Y,theta,m,alpha,lambdaf)
      theta = new_theta
      if i > 10:
          #gotem = Cost_Function(X,Y,theta,m)
          r2 = adjr2(theta,X,Y)
          #print(r2,maxr2, gotem, minCost)
          #print(best_theta)
          #print(best_theta2)
          if r2 > maxr2:
              maxr2 = r2
              best_theta = theta
              iter_found = i
          """if gotem < minCost:
              minCost = gotem
              best_theta2 = theta"""
   print(iter_found)
   return best_theta

#definition of the sigmoid function generalized so that arbitrary length
#theta arrays can be used for multivariate models
def plot_sigmoid(x,theta):
    z = 0
    check = isinstance(x,float)
    for j in range(len(theta)):
        if (check == True and j == len(theta)-1):
            z += theta[j]
        if (check == True and j < len(theta) - 1):
            z += theta[j]*x
        if check == False:
            z += theta[j]*x[j]
    if z >= 0:
        return 1/(1 + math.exp(-z))
    if z < 0:
        return 1 - 1/(1 + math.exp(z))

#calculation of adjusted R^2 to judge quality of model and choose best
#model when multiple are considered
def adjr2(theta, x, y):
    r2sum = 0
    avy = average(y)
    sumy = 0
    for i in range(len(y)):
        r2sum += (y[i] - plot_sigmoid(x[i],theta))*(y[i] - plot_sigmoid(x[i],theta))
        sumy += (y[i] - avy)*(y[i] - avy)
    r2sum = r2sum/(len(y)-len(theta)+1)
    sumy = sumy/len(y)
    return 1 - r2sum/sumy

def errorsumsq(theta,x,y):
    sum1 = 0
    for i in range(len(y)):
        sum1 += (y[i] - calc_hypothesis(theta, x[i]))*(y[i] - calc_hypothesis(theta, x[i]))
    return sum1

def mse(theta,x,y):
    sum2 = 0
    for i in range(len(y)):
        sum2 += (y[i] - calc_hypothesis(theta, x[i]))*(y[i] - calc_hypothesis(theta, x[i]))
    sum2 = math.sqrt(sum2)/len(y)
    return sum2

#function to calculate sensitivity/sensitivity defined as the function marking ectasia > .5 and control < .5
def sense(theta,x,y, nE, nC):
    se_sp = [0,0] #sensitivity, specificity
    for i in range(len(y)):
        y1 = plot_sigmoid(theta, x[i])
        print(y1, y[i])
        if (y[i] == 1 and y1 > .5):
            se_sp[0] += 1
        if (y[i] == 0 and y1 < .5):
            se_sp[1] += 1
    se_sp[0] = se_sp[0]/nE
    se_sp[1] = se_sp[1]/nC
    return se_sp


#***************************MAIN PROGRAM****************************************
#Number of cases to be examined
numberOfCases = 70
numControl = 262
#Number of data points collected
TotalnumberOfDataPoints = 21 #total data including eye measurements
numberOfDataPoints = 6 #number of data points regarding strain

#column and row of excel sheet where necessary data is kept
#ignore headings and names/id/etc.
coln = 5
rown = 1

#k-folds for cross validation
k = 4
validation_length = (numControl + numberOfCases)/4
train_length = (numControl + numberOfCases) - validation_length
validation_control_length = math.floor(numControl/4)
validation_ect_length = math.floor(numberOfCases/4)

#******************************************************************************
#access control data
book = open_workbook('C:\\Users\\BrianAndrews\\OneDrive\\OptoQuest\\Ecstasia Study\\Final Data Sets and Analysis\\Control12_strain_fix.xlsx')
mainSheet = book.sheet_by_index(0)

#table to store control data
control = np.zeros((numControl, TotalnumberOfDataPoints), float)

#begin reading and storing the data from the excel sheet
control = store(coln, rown, numControl, TotalnumberOfDataPoints, control)

#close book and delete pointers
book.release_resources()
del book

#*******************************************************************************
#access ectasia data
book = open_workbook('C:\\Users\\BrianAndrews\\OneDrive\\OptoQuest\\Ecstasia Study\\Final Data Sets and Analysis\\Ectasia13_strain_fix.xlsx')
mainSheet = book.sheet_by_index(0)

#table to store ectasia data
ectasia = np.zeros((numberOfCases, TotalnumberOfDataPoints), float)

#begin reading and storing the data from the excel sheet
ectasia = store(coln,rown,numberOfCases,TotalnumberOfDataPoints, ectasia)

#close book and delete pointers
book.release_resources()
del book

#***************************************************************************

indices = [0,1,3,4,6,7,8,10,12,13,14,15,16,17,18,20] #full model
#indices = [0,7,8,10,11,12,15,17,18,20] #pretreatment
#indices = [0,1,3,4,6,7,8,10,11,12,15,17,18,20] #full model minus a few


y = []
for b in range(numControl):
    y.append(0)
for c in range(numberOfCases):
    y.append(1)

x = [[] for j in range(numControl + numberOfCases)] #new array each round of log regression
divisor = []
for h in range(len(indices)):
    avE = average(ectasia[:,indices[h]]) #calculate averages
    avC = average(control[:,indices[h]])
    val1 = int(math.log(abs(avE))/math.log(10)) #determine the power of ten for both
    val2 = int(math.log(abs(avC))/math.log(10))
    """
    choose the larger one.  helps prevent math range error for log function
    """
    if val2 >= val1:   
        divisor.append(val2)
    if val1 > val2:
        divisor.append(val1)
#print(divisor)
for m in range(numControl):
    for n in range(len(indices)): #arrange data accordingly, dividing by the necessary power of ten
        #print(divisor[n], control[m, bigJindex[indices[n]]]/(10**divisor[n]))
        x[m].append(control[m, indices[n]]/(10**divisor[n]))
    #x[m].append((-.003974*control[m,10])+4.89) #linear regression meanSP derived from pachymin
    #x[m].append((-.003542*control[m,10])+5.151) #linear regression maxSP derived from pachymin
    x[m].append(1)
for o in range(numberOfCases): #same for ectasia group
    for p in range(len(indices)):
        x[o + numControl].append(ectasia[o, indices[p]]/(10**divisor[p]))
    #x[o + numControl].append((-.003974*control[m,10])+4.89) #linear regression meanSP derived from pachymin
    #x[o + numControl].append((-.003542*control[m,10])+5.151) #linear regression maxSP derived from pachymin
    x[o + numControl].append(1)
#print(x)

#*******************************************************************************
#logistic regression model construction with k-fold cross validation
for j in range(k):
    xtrain = [] #arrays to split the x array into train and validation sets
    xvalid = []
    ytrain = []
    yvalid = []
    #print("len(x)",len(x))
    #print(j*validation_control_length,(j*validation_control_length) + validation_control_length)
    #print(j*validation_ect_length,(j*validation_ect_length)+ validation_ect_length)
    for k in range(numControl):
        if (k >= j*validation_control_length and k < (j*validation_control_length) + validation_control_length):
            xvalid.append(x[k])
            yvalid.append(y[k])
        else:
            xtrain.append(x[k])
            ytrain.append(y[k])
    for m in range(numberOfCases):
        if (m >= j*validation_ect_length and m < (j*validation_ect_length) + validation_ect_length):
            xvalid.append(x[m+numControl])
            yvalid.append(y[m+numControl])
        else:
            xtrain.append(x[m+numControl])
            ytrain.append(y[m+numControl])

    #print(len(xvalid),len(yvalid))
    #print(len(xtrain),len(ytrain))
        
    
    initial_theta = [] # Initial guess
    for a in range(len(x[0])):
        initial_theta.append(0)
    alpha = .1 # learning rate
    lambdaf = 0 #ridge learning rate
    iterations = 70000
    optimal_theta = logistic_regression(xtrain,ytrain,alpha,initial_theta,iterations,lambdaf)

    print(optimal_theta)
    print(adjr2(optimal_theta, xtrain,ytrain))
    print(adjr2(optimal_theta, xvalid,yvalid))
    print(mse(optimal_theta, xvalid,yvalid))
    print(sense(optimal_theta,xvalid,yvalid, validation_ect_length, validation_control_length))
