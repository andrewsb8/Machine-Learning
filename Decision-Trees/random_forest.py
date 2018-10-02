"""
This script will take data from a csv file regarding temperature and will train a random forest regression model to predict future
temperatures (highs and lows) based on previous day's temperatures as well as monthly averages.

Author: Brian Andrews
"""

import sys
import math
import csv
import random

"""
The following lists are meant to store the data during training.  year_month is a 1D list and entries of this list correspond to rows of the other
multidimensional lists.

year_month : stores year and month
date : stores day of each month in each row
maxi : stores max temp of each day
mini : stores min temp of each day
avg_high_low : stores avg high and low of each month
"""
year_month = []
date = []
maxi = []
mini = []
avg_year_month = []
avg_high_low = []

"""
The following reads the necessary data from the really terribly kept data file Weather_Data.csv.
That file holds data since 2006 but does not contain data from January 2012 to July 2015 :(
"""
"""
with open('Weather_Data.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    prev_line = ''
    found_date = 0
    for r in reader:
        line = r[0]
        if (prev_line == '' and r[0] != 'Date' and r[0] != '#days' and r[0] != '#days31' and r[0] != '#days 30' and r[1] == '' and r[0] != ''):
            year_month.append(line)
            line2 = line
        if (found_date == 1 and line != '#days' and line != '#days31'):
            #print(line)
            if prev_line == '30':
                if line == '31':
                    date.append(line)
                    year_month.append(line2)
                    try:
                        maxi.append(float(r[1]))
                        mini.append(float(r[2]))
                    except:
                        print(r[1], r[2], "hi/lo: Cannot be turned into an int")
                        maxi.append(r[1])
                        mini.append(r[2])
            else:
                date.append(line)
                year_month.append(line2)
                try:
                    maxi.append(float(r[1]))
                    mini.append(float(r[2]))
                except:
                    print(r[1], r[2], "hi/lo: Cannot be turned into an int")
                    maxi.append(r[1])
                    mini.append(r[2])
        if (line == '31' or prev_line == '30'):
            #print(line)
            found_date = 0
        if (line == 'Date' or line == 'Day'):
            found_date = 1
        if line == 'Avg':
            avg_year_month.append(line2)
            avg_year_month.append(line2)
            try:
                avg_high_low.append(float(r[1]))
                avg_high_low.append(float(r[2]))
            except:
                print(r[1], r[2], "Average hi/lo: Cannot be turned into an int")
                avg_high_low.append(r[1])
                avg_high_low.append(r[2])
        prev_line = r[0]
"""    
#print(year_month, date, maxi)
#print(avg_year_month,avg_high_low)

#*****************************************************************************************************************************************************
#Data is collected.  The model construction will begin here.

#REMEMBER THAT FROM JAN 2012 TO AUG 2015 THERE IS NO DATA
#So, after Dec 31st, 2011, the next date that can be considered for learning
#is Aug 3rd, 2015 because there is no two days before the first and the
#second.  Which is SUPER annoying.

#the tree structure will be defined here
class Node:
    def __init__(self, data, lev):
        self.level = lev
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data, left_right):
        if (left_right == 1 and self.left == None):
            self.left = Node(data, self.level + 1)
        if (left_right == 0 and self.right == None):
            self.right = Node(data, self.level + 1)

    def printtree(self):   #prints tree starting from left most child node
        if self.left != None:   #goes left child, parent, right child
            self.left.printtree()
        print(self.data, self.level)
        if self.right != None:
            self.right.printtree()
            
""" testing out the nodes and tree functionality
root = Node([1,2,3], 0)
#print(root.data, root.level)
root.insert([4,5,6],1)
#print(root.left.data, root.left.level)
root.left.insert([7,8,9],1)
#print(root.left.left.data, root.left.left.level)
root.insert([10,11,12],0)
#print(root.right.data,root.right.level)
root.insert([13,14,15],0) #this should not enter the tree at all
root.left.insert([34,35,36],0)
root.printtree()
"""

def average(array):
    total = sum(array)
    lens = len(array)
    return total/lens

def std(av,array): #compute standard deviation of a list of values
    total = 0
    for i in range(len(array)):
        total += (array[i]-av)*(array[i]-av)
    var = total/len(array)
    return math.sqrt(var)


            


























