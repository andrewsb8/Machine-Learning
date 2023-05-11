"""
Medasync Cost Prediction Algorithm v2 Production Version:

The goal of this program is to run univariate regression for each primary
diagnosis code.  This code will be used to initially populate the table
Diagnosis Cost Regression Equations Table.

For each diagnosis code the following occurs:
- Information broken down by Charge code
- Collect, length of stay, transaction totals, and number of comorbidities
- Calculate cost per day
- Perform linear univariate regression on cost per day versus # of comorbidities
- Record parameters resultant of regression and repeat for next primary
    diagnosis codes

This is the test version of this software which includes various pieces of
extraneous information included throughout the process which are meant to
assist in the debugging process.

The true release of this software will be stripped to include only necessary
information to complete the analysis.

********************************************************************************
- NOTE: CHECK TO SEE WHICH DB LOGIN IS BEING USED BEFORE EACH USE.  THIS IS FOR
THE PROD DB.  COMMITTING THIS CODE WILL CHANGE THE TABLE IN THE PROD DB.
********************************************************************************

Author: Brian Andrews
"""

import sys
import math
import xlrd
from xlrd import open_workbook
import numpy as np
import matplotlib.pyplot as plt
import pyodbc
import xlwt

#******************************************************************************
#Access to database
connection = pyodbc.connect(
    r'DRIVER={SQL Server};'
    r'SERVER='
    r'DATABASE=testCostAnalysis;'
    r'UID=;'
    r'PWD=;'
    )

#command used to store queries.  Can't have two connections open at once...
cursor = connection.cursor()

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
def rsq(avy, param = [], x = [], y = []):
    num = 0
    dem = 0
    for u in range(len(x)):
        num += (y[u] - (param[0]*x[u] + param[1]))*(y[u] - (param[0]*x[u] + param[1]))
        dem += (y[u] - avy)*(y[u] - avy)
    if (num == 0 or dem == 0):
        return 0
    return 1 - (num/dem)

#calculate regression parameters
def regress(x = [], y = []): #x is # of comorbidities and y is cost per day
    params = []
    count = 0 #count length of array ignoring zero values
    x2sum = 0
    xysum = 0
    xsum = 0
    ysum = 0
    #calculate sum of x^2 and x*y
    for g in range(len(x)):
        if y[g] != 0: #can't have zero cost per day, that would mean either no length of stay and/or recorded transactions
            x2sum += x[g]*x[g]
            xysum += x[g]*y[g]
            xsum += x[g]
            ysum += y[g]
            count += 1
    #print(count, x2sum, xsum)

    #normal regression
    m = (1 / ((count * x2sum) - (xsum * xsum))) * ((count * xysum) - (xsum * ysum))
    b = (1 / ((count * x2sum) - (xsum * xsum))) * ((x2sum * ysum) - (xsum * xysum))

    #when a zero intercept is required
    #m = xysum/x2sum
    #b = 0

    params.append(m)
    params.append(b)

    return params


#*******************************************************************************
#section for collecting data from data base and putting them into tables

#get the patient and facility ids for primary diagnosis codes
cursor.execute("select PatientID, FacilityID, DiagCode from ReportDiagnosis")
repdiag = [[] for h in range(3)]
for b in cursor.fetchall():
    repdiag[0].append(b[0])
    repdiag[1].append(b[1])
    repdiag[2].append(b[2])

#enter the fourth rows via the following query
cursor.execute("select PatientID, FacilityID, StayDays from ReportPatients")
stays = [[] for i in range(3)]
for i in cursor.fetchall():
    stays[0].append(i[0])
    stays[1].append(i[1])
    stays[2].append(i[2])

#query for comorbidities
cursor.execute("select PatientID, FacilityID, Classification, OnsetDate, DiagnosisCode from Diagnosis")
diag = [[] for j in range(5)]
for k in cursor.fetchall():
    diag[0].append(k[0])
    diag[1].append(k[1])
    diag[2].append(k[2])
    diag[3].append(k[3])
    diag[4].append(k[4])

#new query for transactions
cursor.execute("select PatientId, FacilityId, ChargeCode, TransactionType, cast(Total AS FLOAT) AS Tot from ReportTransactions where TransactionType = 'A'")
trans = [[] for l in range(5)]
for m in cursor.fetchall():
    trans[0].append(m[0])
    trans[1].append(m[1])
    trans[2].append(m[2])
    trans[3].append(m[3])
    trans[4].append(m[4])


charge = [[] for i in range(5)]
cursor.execute("select ChargeCode, FacilityID, CategoryName, CategoryId, SubCategoryId from ChargeCodeMappings")
for h in cursor.fetchall():
    charge[0].append(h[0])
    charge[1].append(h[1])
    charge[2].append(h[2])
    charge[3].append(h[3])
    charge[4].append(h[4])

#********************************************************************************
#start to organize data for main loop

#making a list of all of the diagnosis codes
pdiagcodes = []
i = 0
for h in range(len(repdiag[0])):
    for g in range(len(pdiagcodes)):
        if repdiag[2][h] == pdiagcodes[g]:
            i = 1
    if i == 0:
        pdiagcodes.append(repdiag[2][h])
    if i == 1:
        i = 0

#keep tracks of all of the charge code categories
chargecategories = []
categoryid = []
subcategoryid = []
i = 0
for g in range(len(charge[0])):
    for n in range(len(chargecategories)):
        if charge[2][g] == chargecategories[n]:
            i = 1
    if i == 0:
        chargecategories.append(charge[2][g])
        categoryid.append(charge[3][g])
        subcategoryid.append(charge[4][g])
    if i == 1:
        i = 0

#start the main loop
for a in range(len(pdiagcodes)):
    bigtable = [[] for u in range(len(chargecategories)+4)]

    #Creating table of data with the following rows:

    #Patient ID | Facility ID | Length of Stay | Number of Comorbidities | Array of cost per days broken down by charge code

    #enter the first two rows
    for y in range(len(repdiag[0])):
        gotem = 0
        if repdiag[2][y] == pdiagcodes[a]:
            for t in range(len(bigtable[0])):
                if (repdiag[0][y] == bigtable[0][t] and repdiag[1][y] == bigtable[1][t]):
                    gotem = 1
            if gotem == 0:
                bigtable[0].append(repdiag[0][y])
                bigtable[1].append(repdiag[1][y])
                for i in range(len(chargecategories)+2):
                    bigtable[i+2].append(0)
            if gotem == 1:
                gotem = 0

    #enter the third rows      
    for k in range(len(stays[0])):
        for u in range(len(bigtable[0])):
            if (bigtable[0][u] == stays[0][k] and bigtable[1][u] == stays[1][k]):
                bigtable[2][u] = stays[2][k]

    #fourth row
    for n in range(len(diag[0])):
        for b in range(len(bigtable[0])):
            if(bigtable[0][b] == diag[0][n] and bigtable[1][b] == diag[1][n] and diag[2][n] != 'Principal' and diag[2][n] != 'Admitting'):
                bigtable[3][b] += 1

    #fill in the rest of the matrix with cost per days for each charge code
    for d in range(len(trans[0])):
        for b in range(len(bigtable[0])):
            if (bigtable[0][b] == trans[0][d] and bigtable[1][b] == trans[1][d]):
                for c in range(len(charge[0])):
                    if (trans[1][d] == charge[1][c] and trans[2][d] == charge[0][c]):
                        for h in range(len(chargecategories)):
                            if charge[2][c] == chargecategories[h]:
                                if bigtable[2][b] != 0:
                                    bigtable[h+4][b] += trans[4][d]/bigtable[2][b]

    
    for q in range(len(chargecategories)):
        numpatients = 0
        #make arrays of nonzero cost per day for plotting purposes
        x2 = []
        y2 = []
        for t in range(len(bigtable[0])):
            if bigtable[q+4][t] != 0:
                x2.append(bigtable[3][t])
                y2.append(bigtable[q+4][t])
                numpatients += 1

        xfound = [] #array that will keep track of individual entries of x2
        xbool = 0
        for h in range(len(x2)): #create an array of the x values
            for x in range(len(xfound)):
                if x2[h] == xfound[x]:
                    xbool =1
            if xbool == 0:
                xfound.append(x2[h])
            if xbool == 1:
                xbool = 0

        #calculate the parameters
        if (len(x2) > 1 and len(y2) > 1 and len(xfound) > 1):
            result = regress(x2, y2)
            rsquared = rsq(average(y2), result, x2, y2)
        if ((len(x2) <= 1 or len(y2) <= 1) or (len(x2) > 1 and len(y2) > 1 and len(xfound) <= 1)): #need more than one data point to do regression analysis
            result = [0,0]
            rsquared = 0
        
        print(pdiagcodes[a], chargecategories[q], result, rsquared)
        cursor.execute("insert into DiagnosisCostRegressionEquations values (?, ?,?,?,?,?,?)",pdiagcodes[a],categoryid[q],subcategoryid[q],result[0],result[1],rsquared,numpatients)
        cursor.commit()
























