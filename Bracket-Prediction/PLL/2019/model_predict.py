import sys
import math
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

"""
This script takes game data from Round X Games in PLL_Data.xlsx. It transforms the data the
same way as in model_train.py. Then the model trained in model_train.py is loaded and used
to predict the outcome of each game in the input data sheet.

Author: andrewsb8
Last Date Modified: 9/9/2019
"""

#import the data to train the model
sheet_with_data = 'Round 3 Games'
raw_dat = pd.read_excel("C:\\Users\\*****\\OneDrive\\Bracket_Prediction\\PLL\\PLL_Data.xlsx", sheet_name=sheet_with_data)

#create new dataframe with same columns as raw_dat
test_dat = pd.DataFrame(columns=raw_dat.columns)
test_dat = test_dat.drop(columns = ['Week','Game ID', 'Team ID', 'Predicted Result'])

#for each game, want to calculate the difference between all of the stats and append it to the train_dat dataframe for model training
min_game_id = raw_dat.loc[:,'Game ID'].min()
max_game_id = raw_dat.loc[:,'Game ID'].max()

#pull each game and subtract the two stat lines to create the training data set
for i in range(min_game_id, max_game_id+1, 1):
    current_game = raw_dat[raw_dat['Game ID'] == i]
    print(current_game)
    current_game = current_game.drop(columns = ['Week','Game ID', 'Team ID', 'Predicted Result'])
    current_game.loc['Diff'] = current_game.iloc[0] - current_game.iloc[1]
    test_dat = test_dat.append(current_game.loc['Diff'], ignore_index = True)

#apply the scaler to the test data
scale = StandardScaler()
scale.fit(test_dat)
test_dat = scale.transform(test_dat)

#load the model
tree_model = pickle.load(open("tree_model","rb"))
#predict outcomes
yeet = tree_model.predict(test_dat)
print(yeet)
