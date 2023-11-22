"""
This code takes the results of the regression analysis in the MedaSync rds db
and uses the previously calculated parameters and user input variables to
calculate estimates for each category plus the total.

This function is designed for deployment on AWS Lambda.

Author: Brian Andrews
"""

import sys
import math
import pymysql


def lambda_handler(event,context):

    #user entered variables
    PrimaryDiagCode = event['DiagCode']
    Stay = event['LOS']
    NumOfCo = event['NOC']

    
    #******************************************************************************
    #Access to database
    connection = pymysql.connect(host = '',
                             user = '',
                             password = '',
                             db = '')

    cursor = connection.cursor()
       
    cursor.execute("select CatId, SubCatId, NumCoeff, YearCoeff, Intercept, R2, standarderror_per_day from model_parameters where DiagCode = '" + PrimaryDiagCode + "'")

    result = {}
    catid = []
    subcatid = []
    estimate = []
    se = []
    for b in cursor.fetchall():
        if b[5] > 0:
            catid.append(b[0])
            subcatid.append(b[1])
            val = (b[2]*NumOfCo) + (b[3]*Stay) + b[4]
            val = val*Stay
            if val >= 0:
                estimate.append(val)
                result["success"] = 1
            if val < 0:
                result["success"] = 0
            se.append(b[6]*Stay)
        if b[5] <= 0:
            result["success"] = 0

    result["catid"] = catid
    result["subcatid"] = subcatid
    result["estimates"] = estimate
    result["standerr"] = se

    cursor.close()
    connection.close()

    return result


