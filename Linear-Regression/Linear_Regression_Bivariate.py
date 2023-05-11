"""
Medasync Cost Prediction Algorithm v2 Production Version:

This script creates multivariate linear regression models for a list of
Diagnosis Codes.  The model calculates parameters used in a linear model to
predicts cost per day based on retrospective data of similar patients
(with similar diagnoses) with the same length of stay and number of
comorbidities.

Author: Brian Andrews
Date: 2018
"""

import sys
import math
import pymysql
import numpy as np
import datetime

#database information is redacted to protect previous client's information
connection = pymysql.connect(host = '',
                             user = '',
                             password = '',
                             db = '')

cursor = connection.cursor()

cursor.execute("delete from model_parameters")
connection.commit()

#*******************************************************************************
#section for necessary functions

#function to calculate the average value of a list
def average(x = []):
    if len(x) == 0:
        return 0
    sumx = 0
    for i in range(len(x)):
        sumx += x[i]

    return sumx/len(x)


#function to calculate coeff of determination (R^2)
def rsq(avy, param = [], x0 = [], x1 = [], y = []):
    num = 0
    dem = 0
    for u in range(len(x0)):
        num += (y[u] - (param[0]*x0[u] + param[1]*x1[u] + param[2]))*(y[u] - (param[0]*x0[u] + param[1]*x1[u] + param[2]))
        dem += (y[u] - avy)*(y[u] - avy)
    if (num == 0 or dem == 0):
        return 0
    return 1 - (num/dem)

#function that returns the standard error of a model
def var(days, numco, params, values):
    sumsq = 0
    for h in range(len(values)): #calculate average sum of squares
        sumsq += (values[h]-((params[0]*numco[h]) + (params[1]*days[h]) + params[2]))*(values[h]-((params[0]*numco[h]) + (params[1]*days[h]) + params[2]))
    var = math.sqrt(sumsq)/len(values)
    return var

#function to calculate the inverse of a matrix
def inverse(mat = [[]]):
    d = np.linalg.det(mat)

    if (math.isnan(d) == True or d == 0):
        return 0

    #calculate the adjugate or transpose cofactor matrix of mat
    minor = np.zeros((len(mat)-1,len(mat)-1),float)
    newmat = np.zeros((len(mat),len(mat)), float)


    for i in range(len(mat)):
        for j in range(len(mat)):
            m = 0
            n = 0
            minor = np.zeros((len(mat)-1,len(mat)-1),float)
            for k in range(len(mat)):
                for l in range(len(mat)):
                    if (k != i and l != j):
                        minor[m,n] = mat[k,l]
                        n += 1
                if n == 2:
                    m += 1
                    n = 0
            if (i+j) % 2 == 0:
                #positive
                newmat[j,i] = np.linalg.det(minor)/d
            if (i+j) % 2 == 1:
                #negative
                newmat[j,i] = -np.linalg.det(minor)/d

    return newmat



#calculate regression parameters
def regress(x0 = [], x1 = [], y = []): #x is # of comorbidities and y is cost per day
    #vector for sum quantities
    vector = []
    #vector for the parameters
    params = []
    #calculate necessary sums and put them in the matrix and vectors appropriately
    count = 0 #count length of array ignoring zero values
    x0sum = 0
    x1sum = 0
    x0x1 = 0
    x02x1 = 0
    x0x12 = 0
    x02x12 = 0
    x02sum = 0
    x12sum = 0
    yx0 = 0
    yx1 = 0
    ysum = 0
    yx0x1 = 0

    for u in range(len(y)): #calculate all the sums
        if y[u] != 0: #no zero cost per day
            x0sum += x0[u]
            x1sum += x1[u]
            x02sum += x0[u]*x0[u]
            x12sum += x1[u]*x1[u]
            x0x1 += x0[u]*x1[u]
            x02x1 += x0[u]*x0[u]*x1[u]
            x0x12 += x0[u]*x1[u]*x1[u]
            x02x12 += x0[u]*x0[u]*x1[u]*x1[u]
            yx0 += y[u]*x0[u]
            yx1 += y[u]*x1[u]
            yx0x1 += y[u]*x0[u]*x1[u]
            ysum += y[u]
            count += 1

    #introduce matrix and vector containing necessary sums (see book for derivation)
    matrix = np.array([[x02sum, x0x1,x0sum], [x0x1,x12sum,x1sum],[x0sum,x1sum,count]])

    vector.append(yx0)
    vector.append(yx1)
    vector.append(ysum)

    new = inverse(matrix)

    if isinstance(new, int) == True:
        return [0,0,0]

    params = np.dot(new, vector)

    return params

#******************************************************************************
#MAIN PROGRAM
alldata = [[] for i in range(12)]
cursor.execute("select * from Patient_Costs")
for row in cursor.fetchall():
    for j in range(len(alldata)):
        alldata[j].append(row[j])

print(alldata[0][0])

#making a list of all of the diagnosis codes
pdiagcodes = []
i = 0
for h in range(len(alldata[0])):
    for g in range(len(pdiagcodes)):
        if alldata[1][h] == pdiagcodes[g]:
            i = 1
    if i == 0:
        pdiagcodes.append(alldata[1][h])
    if i == 1:
        i = 0

categoryid = []
subcategoryid = []
cursor.execute("select * from CategoryId")
for row in cursor.fetchall():
    print(row)
    categoryid.append(row[0])
    subcategoryid.append(row[1])

totalpatientcount = 0
#*******************************************************************************
#start the main loop
for a in range(len(pdiagcodes)):
    #Outline of Table:
    #PatientIdMappingId | DiagCode | Length of Stay | Number of Comorbidities | Array of costs
    bigtable = [[] for i in range(12)] #same as alldata table

    for j in range(len(alldata[0])):
        gotem = 0
        if alldata[1][j] == pdiagcodes[a]:
            for t in range(len(bigtable[0])):
                if alldata[0][j] == bigtable[0][t]:
                    gotem = 1
            if gotem == 0:
                for n in range(len(bigtable)):
                    bigtable[n].append(alldata[n][j])
            if gotem == 1:
                gotem = 0

    #print(bigtable)

    #make arrays of nonzero cost per day for plotting purposes
    #calculate r^2 of total model instead of each category
    x2 = []
    x21 = []
    y2 = []
    numpatients = 0
    for t in range(len(bigtable[0])):
        costdaysum = 0
        for q in range(len(bigtable)-4):
            costdaysum += bigtable[q + 4][t]
        if costdaysum != 0:
            x2.append(bigtable[3][t])
            x21.append(bigtable[2][t])
            y2.append(costdaysum)
            numpatients += 1
            totalpatientcount += 1

    xfound = [] #array that will keep track of individual entries of x2
    xbool = 0
    for h in range(len(x2)): #create an array of the x values to find duplicates and determine if there isn't enough data for the model
        for x in range(len(xfound)):
            if x2[h] == xfound[x]:
                xbool =1
        if xbool == 0:
            xfound.append(x2[h])
        if xbool == 1:
            xbool = 0

    #calculate the parameters
    if (len(x2) > 1 and len(y2) > 1 and len(xfound) > 1):
        result = regress(x2,x21,y2)
        if (result[0] == 0 and result[1] == 0 and result[2] == 0):
            rsquared = 0
        else:
            rsquared = rsq(average(y2), result, x2,x21, y2)
    if ((len(x2) <= 1 or len(y2) <= 1) or (len(x2) > 1 and len(y2) > 1 and len(xfound) <= 1)): #need more than one data point to do regression analysis
        result = [0,0,0]
        rsquared = 0

    for q in range(len(categoryid)):
        #make arrays of nonzero cost per day for plotting purposes
        x2 = []
        x21 = []
        y2 = []
        for t in range(len(bigtable[0])):
            if bigtable[q+4][t] != 0:
                x2.append(bigtable[3][t])
                x21.append(bigtable[2][t])
                y2.append(bigtable[q+4][t])

        xfound = [] #array that will keep track of individual entries of x2
        xbool = 0
        for h in range(len(x2)): #create an array of the x values to find duplicates and determine if there isn't enough data for the model
            for x in range(len(xfound)):
                if x2[h] == xfound[x]:
                    xbool =1
            if xbool == 0:
                xfound.append(x2[h])
            if xbool == 1:
                xbool = 0


        #calculate the parameters
        if (len(x2) > 1 and len(y2) > 1 and len(xfound) > 1):
            result = regress(x2,x21,y2)
            se = var(x2, x21, result, y2) #calculate the standard error of the model to provide a delta for uncertainty
        #these conditions make model creation impossible because there is not enough data to solve the system of equations
        if ((len(x2) <= 1 or len(y2) <= 1) or (len(x2) > 1 and len(y2) > 1 and len(xfound) <= 1)): #need more than two data points to do this regression analysis
            result = [0,0,0]
            se = 0

        print(pdiagcodes[a],categoryid[q],subcategoryid[q], result, rsquared, numpatients, se)
        cursor.execute("insert into model_parameters values ('{0}','{1}','{2}','{3}','{4}','{5}','{6}','{7}', '{8}')".format(pdiagcodes[a],categoryid[q],subcategoryid[q],result[0],result[1],result[2],rsquared,numpatients,se))
        connection.commit()

    del bigtable

now = datetime.datetime.now()

output = open("model_update_log.txt",'r')
lines = output.readlines()
readlines = []
for line in lines:
    readlines.append(line.strip('\n'))
output.close()

writestuff = open("model_update_log.txt", 'w+')
writestuff.write("*****************************************************************\n")
writestuff.write("Date and Time: %s\n" % (now))
writestuff.write("Program ran: parameter_estimation.py\n")
writestuff.write("Number of patients considered: %d\n" % (totalpatientcount))
writestuff.write("\n\n")
for n in range(len(readlines)):
    writestuff.write(readlines[n])
    writestuff.write("\n")

writestuff.close()
