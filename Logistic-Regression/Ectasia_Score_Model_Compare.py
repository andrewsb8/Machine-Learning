#*******************************************************************************************************************************************************************
#This script was used in a project while working for OptoQuest in the Cleveland
#Clinic to research postoperative risk of ectasia forming in LASIK patients.

#This script takes multiple combinations of data points (defined explicitly by the user), trains multivariate logistic regression models for those
# combinations of data points, and compares the models by various metrics. These metrics are the area under the receiving operating characteristic
#curve, the J statistic (provides the best cutoff for sigmoid output to decide between positives and negatives to maximize sensitivity and specificity),
#and Decision curves which shows what conditions are better for which models.

#Author: Brian Andrews
#Last Date Modified: 8/2/2018
#********************************************************************************************************************************************************************

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
            if j == data-1:
                table[i,j] = table[i,11]-table[i,10] #difference in pachy min and pachy apex
            elif j == data-2: #add spherical equivalence MRx error
                table[i,j] = table[i,7] + (table[i,8]/2) #cols 7 and 8 have spherical and cylindrical mrx data
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
def sense(theta,x,y, nE, nC, cutoff):
    se_sp = [0,0] #sensitivity, specificity
    for i in range(len(y)):
        y1 = plot_sigmoid(theta, x[i])
        #print(y1, y[i])
        if (y[i] == 1 and y1 > cutoff):
            se_sp[0] += 1
        if (y[i] == 0 and y1 < cutoff):
            se_sp[1] += 1
    return se_sp

#function to calculate Area under the Operating Characteristic Curve
#look up Right handed Reimann sum to get an idea of how this works visually
def AUC(sense, spec):
    #print(sense, spec)
    num = len(spec)
    sen = len(sense)
    areasum = 0
    for j in range(1,num,1): #right handed Reimann sum integrator
        areasum += sense[j-1]*(spec[j-1]-spec[j])
    return areasum


#***************************MAIN PROGRAM****************************************
#Number of cases to be examined
numberOfCases = 71
numControl = 270
#Number of data points collected
TotalnumberOfDataPoints = 49 #total data including eye measurements
numberOfDataPoints = 6 #number of data points regarding strain

#column and row of excel sheet where necessary data is kept
#ignore headings and names/id/etc.
coln = 4
rown = 1

#******************************************************************************
#access control data
book = open_workbook('C:\\Users\\BrianAndrews\\OneDrive\\OptoQuest\\Ecstasia Study\\Final Data Sets and Analysis\\big_model_control.xlsx')
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
book = open_workbook('C:\\Users\\BrianAndrews\\OneDrive\\OptoQuest\\Ecstasia Study\\Final Data Sets and Analysis\\big_model_ectasia.xlsx')
mainSheet = book.sheet_by_index(0)

#table to store ectasia data
ectasia = np.zeros((numberOfCases, TotalnumberOfDataPoints), float)

#begin reading and storing the data from the excel sheet
ectasia = store(coln,rown,numberOfCases,TotalnumberOfDataPoints, ectasia)

#close book and delete pointers
book.release_resources()
del book

#***************************************************************************
"""
indices key or legend or whatever:
0 - Age
1 - Mean Strain Outcome
2 - Mean Strain Change
3 - Mean Strain Pretreatment
4 - Max Strain Outcome
5 - Max Strain Change
6 - Max Strain Pretreatment
7 - MRx Spherical
8 - MRx Cylindrical
9 - MRx Axis
10 - Pachymin
11 - Pachyapex
12 - ksteep
13 - PSTA
14 - PTA
15 - CLMI
16 - RSB
17 - 7th Anterior Zernike
18 - 12th anterior Zernike
19-46: Rest of the zernikes (see excel sheets)
47 - Spherical Equivalence
48 - Pachyapex -(minus)- Pachymin

Below are different models all labeled.
Full model: consists of every model
Pretreatment: All variables we don't calculate plus strain calculated via linear regression.
No Strain Pretreatment: Above but leave out calculated strain

The idea is to loop through all of these models and effectively compare them to also judge quality.
"""

indices = [[0,1,3,4,6,7,8,10,12,13,14,15,16,17,18,20], #full model
           [0,7,8,10,11,12,15,17,18,20], #pretreatment
           [0,7,8,10,11,12,15,17,18,20], #no strain pretreatment
           [0,10,11,12,15,17,18,20], #no mrx with strain
           [0,10,11,12,15,17,18,20]] #no mrx without strain


#arrays to store values necessary for later
optimal_thetas = []
all_x = []
all_stat = []

y = []
for b in range(numControl):
    y.append(0)
for c in range(numberOfCases):
    y.append(1)

for k in range(len(indices)):
    print(indices[k])
    x = [[] for j in range(numControl + numberOfCases)] #new array each round of log regression
    divisor = []
    for h in range(len(indices[k])):
        avE = average(ectasia[:,indices[k][h]]) #calculate averages
        avC = average(control[:,indices[k][h]])
        val1 = int(math.log(abs(avE))/math.log(10)) #determine the power of ten for both
        val2 = int(math.log(abs(avC))/math.log(10))
        """
        choose the larger one.  helps prevent math range error for log function
        """
        if val2 >= val1:
            divisor.append(val2)
        if val1 > val2:
            divisor.append(val1)
    """
    build list of x data points
    """
    for m in range(numControl):
        for n in range(len(indices[k])): #arrange data accordingly, dividing by the necessary power of ten
            x[m].append(control[m, indices[k][n]]/(10**divisor[n]))
        """
        These added values are the latent strain calculations using thickness.  Derived from a separate
        linear regression script I will get you.

        I could throw in the linear regression here, but it would be quicker to just run a separate
        method, store the parameters, and pull them here.
        """
        if (k == 1 or k == 3): #corresponds to pretreatment model with latent strain
            x[m].append((-.003974*control[m,10])+4.89) #linear regression meanSP derived from pachymin
            x[m].append((-.003542*control[m,10])+5.151) #linear regression maxSP derived from pachymin
        x[m].append(1)
    for o in range(numberOfCases): #same for ectasia group
        for p in range(len(indices[k])):
            x[o + numControl].append(ectasia[o, indices[k][p]]/(10**divisor[p]))
        if (k == 1 or k == 3):
            x[o + numControl].append((-.003974*control[m,10])+4.89) #linear regression meanSP derived from pachymin
            x[o + numControl].append((-.003542*control[m,10])+5.151) #linear regression maxSP derived from pachymin
        x[o + numControl].append(1)
    all_x.append(x)

    #*******************************************************************************
    #logistic regression model construction
    initial_theta = [] # Initial guess
    for a in range(len(x[0])): #number of coefficients equal to the number of variables in the x array
        initial_theta.append(0)
    print(len(x[0]), len(indices[k]), len(initial_theta))
    alpha = .1 # learning rate
    lambdaf = 0 #ridge learning rate (if 0, not utilizing ridge method)
    iterations = 90000 #yeah it's long, but it works *shrug*
    optimal_theta = logistic_regression(x,y,alpha,initial_theta,iterations,lambdaf)
    optimal_thetas.append(optimal_theta)

    print(optimal_theta) #these are the parameters you would pull when calculating risk score
    #print(adjr2(optimal_theta, x,y)) #calculate adjusted R^2 for quality assurance
    #print(mse(optimal_theta, x,y)) #also calculate standard error for quality assurance

    all_stat.append([adjr2(optimal_theta,x,y),mse(optimal_theta,x,y)])
    print(all_stat)

    """
    NOTE:

    If adjr2 is very very low or, god forbid, negative, go above and mess with the variables alpha and iterations.
    This could be automated to some degree, but would make this script run potential forever if restraint is not
    used.
    """

"""
Now the models are created for each necessary category.  We want to also calculate other metrics from which to compare the models.

Calculate AUC of each model
Calculate optimal cutoff of ROC using J statistic
Also calculate Decision curve and produce graphs comparing models.
"""
all_senses = []
all_specs = []
all_oneminusspecs = []
all_netbenefit = []
all_auc = []
all_cases = numControl + numberOfCases
for z in range(len(optimal_thetas)):
    if z == 0:
        allnegative = []
        treateveryone = []
    res = 30 #resolution
    cutoff_percent = 0
    cutoffs = []
    senses = []
    specs = []
    oneminusspecs = [] #for plotting ROC
    netbenefit = []
    bigJ = -2
    cutoff_bigJ = -2
    cutoff_index = -1
    for m in range(res+1):
        if z == 0:
            treateveryone.append((numberOfCases/all_cases)-((numControl/all_cases)*(cutoff_percent/(1-cutoff_percent)))) #assuming everyone will develop ectasia
            allnegative.append(0) #assuming no one develops ectasia (this does not seem to be the correct interpretation here and in the line above...)
        spec = sense(optimal_thetas[z],all_x[z],y,numberOfCases,numControl,cutoff_percent) #returns counts of true positive and true negatives given a cutoff threshold that is the percent output of the regression model
        #print(spec, cutoff_percent)
        senses.append(spec[0]/numberOfCases) #arrays for construction of ROC curve
        specs.append(spec[1]/numControl)
        oneminusspecs.append(1-(spec[1]/numControl))
        maybe = senses[m] - oneminusspecs[m] #J statistic for this cutoff percentage
        if maybe > bigJ: #find optimal cutoff provided by J statistic
            bigJ = maybe
            cutoff_bigJ = cutoff_percent
            cutoff_index = m
        netbenefit.append((spec[0]/all_cases) - (((numControl-spec[1])/all_cases)*(cutoff_percent/(1-cutoff_percent)))) #calculate decision curve quantities for model
        cutoffs.append(cutoff_percent) #create array of cutoffs for plotting
        cutoff_percent += float(1/res)
    all_senses.append(senses)
    all_specs.append(specs)
    all_oneminusspecs.append(oneminusspecs)
    all_netbenefit.append(netbenefit)

    auc = AUC(all_senses[z],all_oneminusspecs[z])
    all_auc.append([cutoffs[cutoff_index],all_senses[z][cutoff_index], all_specs[z][cutoff_index], auc])
    print(all_auc)

    """
    the block below basically just plots the ROC curve, you can ignore this as you won't need this.
    I keep it for testing and visualization.
    """
    line = []
    for k in range(res+1):
        line.append(float(k/res))
    plt.title("ROC Curve")
    plt.plot(all_oneminusspecs[z],all_senses[z], 'b-')
    plt.plot(line,line,'g-')
    plt.show()

"""
At this point, the following arrays should be saved:
optimal_thetas
all_stat
all_auc

Those three arrays store the parameters and statistical quantities used to judge their quality.
We want to try and incorporate these in order to show how the variables our software produces
improves the model.

The plots below I would also like to incorporate, but I'll have to talk to Dr. Dupps about it
to see how we could utilize it in a way that is quick to pick up as the graph alone does not
make it's utility obvious at face value.  So, ignore this for now.
"""

plt.title("Model Decision Curve")
plt.plot(cutoffs, all_netbenefit[0], 'b-', label = "Full Model Net Benefit")
plt.plot(cutoffs, all_netbenefit[2], 'p-', label = "Presim (No Strain) Model Net Benefit")
plt.plot(cutoffs, treateveryone, 'r-', label = "No One Develops Ectasia")
plt.plot(cutoffs, allnegative, 'g-', label = "Everyone Has Ectasia")
plt.xlim(0,.9)
plt.ylim(-.1,.25)
plt.legend()
plt.show()

plt.title("Model Decision Curve")
plt.plot(cutoffs, all_netbenefit[1], 'b-', label = "Presim (W Strain) Model Net Benefit")
plt.plot(cutoffs, all_netbenefit[2], 'p-', label = "Presim (No Strain) Model Net Benefit")
plt.plot(cutoffs, treateveryone, 'r-', label = "No One Develops Ectasia")
plt.plot(cutoffs, allnegative, 'g-', label = "Everyone Has Ectasia")
plt.xlim(0,.9)
plt.ylim(-.1,.25)
plt.legend()
plt.show()
