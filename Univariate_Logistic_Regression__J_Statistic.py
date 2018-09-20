"""
Objective: Establish a threshold value for each variable according to Youden's
Index (J statistic) and choose the variable and the cutoff of the optimal
predictor which is quantified by max(J).

Variables considered will be quantized according to a k binning process.

Author: Brian Andrews
"""

import sys
import math
import xlrd
from xlrd import open_workbook
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#***************************FUNCTIONS*******************************************
#function that copies the excel sheet to a table
def store(colnum, rownum, case, data, table = [[]]):
    numFromString = 0
    check = False
    for i in range(case): #loops traversing xlsx files
        for j in range(data):
            if j == data-1: #add spherical equivalence MRx error
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

#function to find cutoff for ROC curve using Youden's index
def findmax(spec = [], sens = []):
    maxi = -2
    index = -1
    for y in range(len(spec)):
        trial = spec[y] + sens[y] - 1
        if trial > maxi:
            index = y
            maxi = trial
    return index, maxi

#function to define the bins
def makebins(numbins, ect = [], cont = []):
    if max(ect) >= max(cont):
        maxi = max(ect)
    if max(ect) < max(cont):
        maxi = max(cont)
    if min(ect) <= min(cont):
        mini = min(ect)
    if min(ect) > min(cont):
        mini = min(cont)

    binsize = math.sqrt((maxi-mini)*(maxi-mini))/numbins

    array = []
    loc = mini
    array.append(loc)
    for i in range(numbins):
        loc += binsize
        array.append(loc)
        
    return array, binsize

"""
This next block of functions relate solely to the estimation of the parameters
for a logistic regression model.
"""
#this function gives the result of the sigmoid function
def logistic_func(z):
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
def gradient_descent(X,Y,theta,m,alpha):
   theta_new = []
   for pos_i in range(len(theta)):
      CFDerivative = partial_derivate_cost(X,Y,theta,pos_i,m,alpha)
      updated_theta = theta[pos_i] - CFDerivative
      theta_new.append(updated_theta)
   return theta_new

#method to start model creation
def logistic_regression(X,Y,alpha,theta,num_iters):
   m = len(Y)
   for x in range(num_iters):
      new_theta = gradient_descent(X,Y,theta,m,alpha)
      theta = new_theta
   return theta

#definition of the sigmoid function generalized so that arbitrary length
#theta arrays can be used for multivariate models
def plot_sigmoid(x,theta):
    z = 0
    check = isinstance(x,float)
    for j in range(len(theta)):
        if j == len(theta)-1:
            z += theta[j]
        if (check == True and j < len(theta) - 1):
            z += theta[j]*x
        if (check == False and j < len(theta) - 1):
            z += theta[j]*x[j]
    return 1/(1 + math.exp(-z))

#calculation of adjusted R^2 to judge quality of model and choose best
#model when multiple are considered
def adjr2(theta, x, y):
    r2sum = 0
    avy = average(y)
    for i in range(len(y)):
        r2sum += (y[i] - plot_sigmoid(x[i],theta))*(y[i] - plot_sigmoid(x[i],theta))
    r2sum = r2sum/(len(y)-len(theta))
    return 1 - r2sum/avy

#***************************MAIN PROGRAM****************************************
#Number of cases to be examined
numberOfCases = 70
numControl = 258
#Number of data points collected
TotalnumberOfDataPoints = 18 #total data including eye measurements
numberOfDataPoints = 6 #number of data points regarding strain

#column and row of excel sheet where necessary data is kept
#ignore headings and names/id/etc.
coln = 4
rown = 1

#******************************************************************************
#access control data
book = open_workbook('C:\\Users\\andrewb\\OneDrive\\OptoQuest\\Ecstasia Study\\Final Data Sets and Analysis\\Control10.xlsx')
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
book = open_workbook('C:\\Users\\andrewb\\OneDrive\\OptoQuest\\Ecstasia Study\\Final Data Sets and Analysis\\Ectasia11.xlsx')
mainSheet = book.sheet_by_index(0)

#table to store ectasia data
ectasia = np.zeros((numberOfCases, TotalnumberOfDataPoints), float)

#begin reading and storing the data from the excel sheet
ectasia = store(coln,rown,numberOfCases,TotalnumberOfDataPoints, ectasia)

#close book and delete pointers
book.release_resources()
del book

#***************************************************************************
#variable to store the largest j statistic variable to choose predictor
bigJ = -2
#variable for j statistic to compare to and possibly replace bigJ
maybe = -3

#number of bins
k = 64 #technically it will be this value + 1, this k is used more to set bin width

#loop through variables
for y in range(TotalnumberOfDataPoints):
    #arrays to store sensitivity values, specificity values, and threshold or bin values
    emean = [] #sensitivity
    com = [] #specificity
    hs = [] #store values of high strain thresholds

    """
    Function to make the bins with two outputs:
    array of constants marking bin minimums - hs
    constant width of each bin - binwidth
    """
    hs, binwidth = makebins(k,ectasia[:,y],control[:,y])

    """
    Calculate the average of each variable in each group.
    Their respective magnitudes govern the behavior of thresholds
    """
    avC = average(control[:,y])
    avE = average(ectasia[:,y])

    """
    define what is considered the threshold for high strain
    loop through thresholds or bins
    """
    for i in range(len(hs)):
        #counting variables
        hscountE = 0 #high strain count ectasia
        countE = 0 #number of ectasia cases
        hscountC = 0 #high strain count control
        countC = 0 #control count

        #now iterate through the updating process
        if avC <= avE:
            for u in range(numberOfCases):
                if ectasia[u,y] >= hs[i]:
                    hscountE += 1
                    countE += 1
                if ectasia[u,y] < hs[i]:
                    countE += 1
            for v in range(numControl):                
                if control[v,y] >= hs[i]:
                    hscountC += 1
                    countC += 1
                if control[v,y] < hs[i]:
                    countC += 1
        if avC > avE:
            for u in range(numberOfCases):
                if ectasia[u,y] <= hs[i]:
                    hscountE += 1
                    countE += 1
                if ectasia[u,y] > hs[i]:
                    countE += 1
            for v in range(numControl):                
                if control[v,y] <= hs[i]:
                    hscountC += 1
                    countC += 1
                if control[v,y] > hs[i]:
                    countC += 1

        emean.append(hscountE/countE)
        com.append(1-(hscountC/countC))

    """
    Use Youden's index to find optimal cutoff for each variable
    Function findmax provides two outputs:
    b - index in arrays emean and com that produce highest J for respective predictor
    maybe - the largest J value for this specific predictor
    maybe is then compared with the current values in array of bigJ
    bigJ contains the top four J observed from all predictors
    indices (can be used for mappings in db?) for those variables are stored in bigJIndex
    """
    b, maybe = findmax(emean,com)

    if maybe > bigJ:
        bigJ = maybe
        bigJindex = y
    print(y,maybe,emean[b], com[b])

print(bigJindex, bigJ) #print best four predictors and itheir indices

"""
First the data will be organized such that the functions defined above will
be able to use them.  So the same excel files as before will be accessed.

I am reaccessing the excel file rather than just manipulating the old data
to be concise and easier to look back on.
"""
#data structures for model creation and plot creation
x = [[] for i in range(numControl + numberOfCases)]
xplot = [] 
y = []

#******************************************************************************
#access control data
book = open_workbook('C:\\Users\\andrewb\\OneDrive\\OptoQuest\\Ecstasia Study\\Final Data Sets and Analysis\\Control10.xlsx')
mainSheet = book.sheet_by_index(0)

for j in range(numControl):
    check = isinstance(mainSheet.cell_value(rown + j, coln + bigJindex), float) #check format of data
    if check == True:
        x[j].append(mainSheet.cell_value(rown + j, coln + bigJindex)) #predictor value
        x[j].append(1) #a value of 1 is given for the intercept
        y.append(0) #Control group is given a 0 or a "negative diagnosis"
        xplot.append(mainSheet.cell_value(rown + j, coln + bigJindex))
    if check == False:
        x[j].append(strToFloat(mainSheet.cell_value(rown + j, coln + bigJindex)))
        x[j].append(1)
        y.append(0)
        xplot.append(strToFloat(mainSheet.cell_value(rown + j, coln + bigJindex)))


#close book and delete pointers
book.release_resources()
del book

#*******************************************************************************
#access ectasia data
book = open_workbook('C:\\Users\\andrewb\\OneDrive\\OptoQuest\\Ecstasia Study\\Final Data Sets and Analysis\\Ectasia11.xlsx')
mainSheet = book.sheet_by_index(0)

for j in range(numberOfCases):
    check = isinstance(mainSheet.cell_value(rown + j, coln + bigJindex), float)
    if check == True:
        x[j+numControl].append(mainSheet.cell_value(rown + j, coln + bigJindex))
        x[j+numControl].append(1)
        y.append(1) #ectasia group is given a 1 or a "positive diagnosis"
        xplot.append(mainSheet.cell_value(rown + j, coln + bigJindex))
    if check == False:
        x[j+numControl].append(strToFloat(mainSheet.cell_value(rown + j, coln + bigJindex)))
        x[j+numControl].append(1)
        y.append(1)
        xplot.append(strToFloat(mainSheet.cell_value(rown + j, coln + bigJindex)))


#close book and delete pointers
book.release_resources()
del book
"""
print(x)
print(xplot)
print(y)
"""

#*******************************************************************************
#logistic regression model construction
initial_theta = [0,0] # Initial guess
alpha = 1 # learning rate
iterations = 7000 # Number of iterations
optimal_theta = logistic_regression(x,y,alpha,initial_theta,iterations)

print(optimal_theta)
print(adjr2(optimal_theta, xplot,y))

z = []
for i in range(len(xplot)):
    z.append(plot_sigmoid(xplot[i], optimal_theta))

plt.title("Logistic Regression Model: Max Strain Pretreatment Predicting Ectasia")
plt.plot(xplot, y, 'bo')
plt.plot(xplot, z, 'r^')
plt.show()



































