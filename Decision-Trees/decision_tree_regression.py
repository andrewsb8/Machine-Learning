"""
This script will take data from a csv file regarding temperature and will train a decision tree
regression model to predict future temperatures (highs and lows) based on the respective high or low
temperature of the day before.

Author: Brian Andrews
"""

import sys
import math
import csv
import random
import numpy as np

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
   
#print(year_month, date, maxi)
#print(avg_year_month,avg_high_low)

#*****************************************************************************************************************************************************
#Data is collected.  The model construction will begin here.

#REMEMBER THAT FROM JAN 2012 TO AUG 2015 THERE IS NO DATA
#So, after Dec 31st, 2011, the next date that can be considered for learning
#is Aug 3rd, 2015 because there is no two days before the first and the
#second.  Which is SUPER annoying.

def average(array):
    sum1 = 0
    y = len(array)
    if y == 0:
        return 0
    else:
        for j in range(y):
            try:
                sum1 += array[j]
            except:
                sum1 += 0
        return sum1/y

def std(av,array): #compute standard deviation of a list of values
    total = 0
    y = len(array)
    if y == 0:
        return 0
    else:
        for i in range(y):
            try:
                total += (array[i]-av)*(array[i]-av)
            except:
                total += 0
        var = total/len(array)
        return math.sqrt(var)

def maximum(array):
    max1 = -10000
    for i in range(len(array)):
        try:
            if array[i] > max1:
                max1 = array[i]
        except:
            max1 = max1
    return max1

def minimum(array):
    max1 = 10000
    for i in range(len(array)):
        try:
            if array[i] < max1:
                max1 = array[i]
        except:
            max1 = max1
    return max1


def check_accuracy(self, criterion, val):
    dummy_self = self
    while dummy_self.leaf == None:
        if criterion < dummy_self.left.num:
            dummy_self = dummy_self.left
        elif criterion >= dummy_self.right.num:
            dummy_self = dummy_self.right
        print(dummy_self.level,dummy_self.num, criterion, val)
    if dummy_self.leaf != None:
        x = ((dummy_self.leaf-val)*(dummy_self.leaf-val))
        print(dummy_self.leaf, val,criterion,x)
        return x


#the tree structure will be defined here
class Node:
    def __init__(self, data, lev,par, cool,classify,number,le):
        self.level = lev #numbering levels in the tree
        self.left = None #child nodes do not exist yet
        self.right = None
        self.parent = par #node above this node
        self.data = data #array of data to be split
        self.std_lev = cool #standard deviation of temperature data in node
        self.class_ = classify #stores the condition for the node i.e. less than
        self.num = number #stores the number threshold i.e. 76 degress
        self.leaf = le #link from node to leaf, None if the node is not the final level

    def insert_Node(self, data, left_right, cool,cla,num):
        if (left_right == 1 and self.left == None):
            self.left = Node(data, self.level + 1,self,cool,cla,num,None)
        if (left_right == 0 and self.right == None):
            self.right = Node(data, self.level + 1,self,cool,cla,num,None)

    def insert_Leaf(self, data):
        self.leaf = data #set the leaf value of the bottom node as the value of average of data in that node

    def split(self, depth, root):
        if (self.right == None and self.left == None and self.level != depth-1):
            top = maximum(self.data[2])
            bottom = minimum(self.data[2])
            #print(top,bottom,len(self.data[2]),len(root.data[2]))
            sdr_max = -100000
            std_left_sdr = 0
            std_right_sdr = 0
            threshold = 0
            for b in range(int(bottom),int(top)+1,1):
                left = [[] for i in range(len(self.data)+1)] #data tables to be moved to child node
                right = [[] for j in range(len(self.data)+1)]
                for c in range(len(self.data[2])):
                    if (self.data[0][c] == "15-Aug" and (self.data[1][c] == '1' or self.data[1][c] == '2')):
                        this_variable_doesnt_matter = 1
                    else:
                        if (isinstance(self.data[2][c],float) and isinstance(root.data[2][self.data[3][c]-1],float)): #and isinstance(root.data[2][self.data[3][c]-1],float)):
                            if (self.data[2][c-1] < b and c != 0 and c != 1):
                                left[0].append(self.data[0][c])
                                left[1].append(self.data[1][c])
                                left[2].append(self.data[2][c])
                                left[3].append(self.data[3][c])
                            if (self.data[2][c-1] >= b and c != 0 and c != 1):
                                right[0].append(self.data[0][c])
                                right[1].append(self.data[1][c])
                                right[2].append(self.data[2][c])
                                right[3].append(self.data[3][c])
                std_left = std(average(left[2]),left[2])
                std_right = std(average(right[2]),right[2])
                length_left = len(left[2])
                length_right = len(right[2])
                total = length_left + length_right
                sdr_trial = self.std_lev - ((length_left/total)*std_left) - ((length_right/total)*std_right)
                if sdr_trial > sdr_max: #define the qualities of the optimal set of data for child nodes
                    sdr_max = sdr_trial
                    std_left_sdr = std_left
                    std_right_sdr = std_right
                    threshold = b
                    right_final = right
                    left_final = left
            #1 problem: In recursion, self.data[2][c-1] won't work because the data is no longer sequential.  Must fix before extrapolating to multiple levels.
            #problem 1 in the line above should be fixed by referencing root node data
            #Now insert both the left and the right child nodes to current node
            self.insert_Node(left_final,1,std_left_sdr,"<",threshold)
            self.insert_Node(right_final,0,std_right_sdr,">=",threshold)
            #begin recursion
            self.left.split(depth,root)
            self.right.split(depth,root)
        if self.level == depth-1:
            self.insert_Leaf(average(self.data[2]))
    

    def printtree(self):   #prints tree starting from left most child node
        if self.left != None:   #goes left child, parent, right child
            self.left.printtree()
        print(self.class_, self.num, "Level: ", self.level, "Leaf Value: ", self.leaf)
        if self.right != None:
            self.right.printtree()


#first have a set depth for the tree
set_depth = 6

#start root node for high temp of the day
root_high_temp = Node([year_month,date,maxi,np.arange(0,len(maxi))],0,None,std(average(maxi), maxi),'',0,None)
#create tree for high temp of the day
root_high_temp.split(set_depth,root_high_temp)
root_high_temp.printtree()

#start root node for low temp of the day
root_low_temp = Node([year_month,date,mini,np.arange(0,len(mini))],0,None,std(average(mini), mini),'',0,None)
#create tree for high temp of the day
root_low_temp.split(set_depth,root_low_temp)
#root_low_temp.printtree()

#check of accuracy of the trees (this should be terrible, but we shall see)
counth = 0
countl = 0
sum1 = 0
sum2 = 0
for i in range(len(root_high_temp.data[2])):
    if (isinstance(root_high_temp.data[2][i], float) == True and isinstance(root_high_temp.data[2][i-1], float) == True):
        sum1 += check_accuracy(root_high_temp, root_high_temp.data[2][i-1], root_high_temp.data[2][i])
        counth += 1
    if (isinstance(root_low_temp.data[2][i], float) == True and isinstance(root_low_temp.data[2][i-1], float) == True):
        sum2 += check_accuracy(root_low_temp,root_low_temp.data[2][i-1],root_low_temp.data[2][i])
        countl += 1

print(math.sqrt(sum1)/counth, math.sqrt(sum2)/countl)
print(math.sqrt(sum1/counth), math.sqrt(sum2/countl))   

   
   





























