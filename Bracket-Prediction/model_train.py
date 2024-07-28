import sys
import math
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

"""
Data is taken from PLL_Data.xlsx sheet Weekly Aggregates. Each game is taken and the stats
lines are subtracted. A decision tree model is then trained based on the differences between
the two teams aggregated stats leading into the week in question. Win or lose will be defined
for the team with the lowest index in the current_game dataframe. Win = 1, Lose = -1.

Author: andrewsb8
Last Date Modified: 9/9/2019
"""

#import the data to train the model
raw_dat = pd.read_excel("C:\\Users\\*****\\OneDrive\\Bracket_Prediction\\PLL\\PLL_Data.xlsx", sheet_name='Weekly Aggregates')

#create new dataframe with same columns as raw_dat
train_dat = pd.DataFrame(columns=raw_dat.columns)
train_dat = train_dat.drop(columns = ['Week','Game ID', 'Team ID'])

#for each game, want to calculate the difference between all of the stats and append it to the train_dat dataframe for model training
min_game_id = raw_dat.loc[:,'Game ID'].min()
max_game_id = raw_dat.loc[:,'Game ID'].max()

#pull each game and subtract the two stat lines to create the training data set
for i in range(min_game_id, max_game_id+1, 1):
    current_game = raw_dat[raw_dat['Game ID'] == i]
    current_game = current_game.drop(columns = ['Week','Game ID', 'Team ID'])
    current_game.loc['Diff'] = current_game.iloc[0] - current_game.iloc[1]
    train_dat = train_dat.append(current_game.loc['Diff'], ignore_index = True)

#print(train_dat)

#separate the data into X and Y
X = train_dat.drop(columns = ['Win'])
Y = train_dat.loc[:,'Win']

#scale the data with a standard scaler
scale = StandardScaler()
scale.fit(X)
x_train = scale.transform(X)

#now create the model to be trained and fit it to the data
#need to use a random_state to ensure same results
Dtree = DecisionTreeClassifier(random_state = 1)
tree_model = Dtree.fit(x_train, Y)

print(tree_model.feature_importances_)

#MODEL IS TRAINED: output serialized version of the model
pickle.dump(tree_model, open("tree_model","wb"))

