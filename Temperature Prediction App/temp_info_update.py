"""
This script reads in a decision tree object for predicting high/low temperature based on the
previous day's temperature.  The previous day's high/low temperatures are retrived from a locally
stored json object file and used to predict the current day's high and low temperatures.  The json
object is the updated and overwritten.  The training script for this decision tree (and other
pending models) are not stored on this server.  This script will be run every 24 hours to update
a json object file that stores the related retrived and predicted temperature information.  A flask
app will handle the call for the json object to be sent to a user.

To be done:
    -Find source for current day's weather
    -Place this script on the ec2 instance with appropriately built json object and decision tree
        files
    -Test and automate    

Author: Brian Andrews
Last Edited: 4/4/2019
"""
import sys
import pickle
import math
import numpy as np
import requests
import datetime
import json

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

    def printtree(self):   #prints tree starting from left most child node
            if self.left != None:   #goes left child, parent, right child
                self.left.printtree()
            print(self.class_, self.num, "Level: ", self.level, "Leaf Value: ", self.leaf)
            if self.right != None:
                self.right.printtree()

#predict temp by traversing tree
#self is the tree
#criterion is the prior day temp            
def predict_1day(self, criterion): 
    dummy_self = self
    while dummy_self.leaf == None:  #traverse tree
        if criterion < dummy_self.left.num:
            dummy_self = dummy_self.left
        elif criterion >= dummy_self.right.num:
            dummy_self = dummy_self.right
    if dummy_self.leaf != None: #return leaf value
        return dummy_self.leaf


high_temp = pickle.load(open("root_high_temp_1day","rb")) #load model objects
#high_temp.printtree()
low_temp = pickle.load(open("root_low_temp_1day","rb"))
#low_temp.printtree()

#create a json object to store temperature information
temp_json = {}

#import at least today's predicted temp and yesterday's predicted temp (high & low)
#getting yesterday's numbers will come from a json file!
with open("temp_json_file.json","r") as readit:
    temp_json_r = json.load(readit) #added r to denote that this is being read

print(temp_json_r)

#old information being transmitted from old json object put into new appropriate spots    
temp_json['yesterday_high'] = temp_json_r['today_high']
temp_json['yesterday_low'] = temp_json_r['today_low']
temp_json['yesterday_predict_high'] = temp_json_r['predict_high']
temp_json['yesterday_predict_low'] = temp_json_r['predict_low']
readit.close()
del temp_json_r


#new information from darksky api
session = requests.Session()
#establish url for 2nd and arch latitude, longitude and only request daily information
url = "https://api.darksky.net/forecast/386d20be8eba9bb9dbfc46d3be00a368/39.9520, 75.1431?exclude=hourly,minutely,currently,alerts,flags"
result = session.get(url)
result = json.loads(result.text)
temp_json['today_high'] = result['daily']['data'][0]['temperatureHigh']
temp_json['today_low'] = result['daily']['data'][0]['temperatureLow']


#predict today's temperature with tree
temp_json['predict_high'] = predict_1day(high_temp,temp_json['yesterday_high'])
temp_json['predict_low'] = predict_1day(low_temp,temp_json['yesterday_low'])

#need the date to ensure the code works over time
temp_json['now'] = datetime.datetime.now().ctime() #get the date so I know this log is kept up to date

print(temp_json)

#store new json object
with open("temp_json_file.json","w") as writeit:
    json.dump(temp_json, writeit)
















