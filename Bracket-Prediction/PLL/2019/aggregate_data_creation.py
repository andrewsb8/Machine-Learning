import sys
import math
import pandas as pd
import openpyxl

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

"""
This script produces a data set for weekly averages, aggregates, and percentages for specific
stats which can be found in the definitions of the dataframes below. A set of data for each team
is printed to an excel sheet for each week. Those stats are each teams leading into the week labeled.
The model(s?) will be trained with these weekly aggregated stats. These sheets are manually placed
into the Weekly Aggregates sheet of PLL_Data.xlsx from test.xlsx.

Author: andrewsb8
Last Date Modified: 9/9/2019
"""

#create a nested list with the following column format:    
#[av goals, av 2 pt shot, 2 pt %, av shots, shot %, av gb, av TO, av CT, FO %, Av PP, PP %, Av Saves, Save %, Point Differential, Streak, Upsets, GAA]

#rows for nested list will be the following:
#Whipsnakes 1
#Chaos 2
#Redwoods 3
#Archers 4
#Atlas 5
#Chrome 6

number_of_teams = 6
number_of_stats = 17

#*************************************************************************************************************************
#making three data frames to keep the weekly results imported data, the stat totals, and the information
#i want to put into the excel sheet for training a model preformatted.

nested = [[0 for i in range(number_of_stats)] for i in range(number_of_teams)]
main_datfram = pd.DataFrame(nested, index = [1,2,3,4,5,6], #index = ['Whipsnakes','Chaos','Redwoods','Archers','Atlas','Chrome'],
                            columns = ['av goals', 'av 2 pt shot', '2 pt %', 'av shots', 'shot %', 'av gb', 'av TO', 'av CT', 'FO %', 'av PP', 'PP %', 'av Saves', 'Save %', 'Point Differential', 'Streak', 'Upsets', 'GAA'])

nested2 = [[0 for i in range(15)] for i in range(number_of_teams)]
side_datfram = pd.DataFrame(nested2, index = [1,2,3,4,5,6], #index = ['Whipsnakes','Chaos','Redwoods','Archers','Atlas','Chrome'],
                            columns = ['total goals', 'total 2 pt shot', 'total 2 pt goals', 'total shots', 'total goals (no 2 pt)', 'total gb', 'total TO', 'total CT', 'total FO','total FO won', 'total PP','total attempted PP', 'total Saves', 'total shots on goal', 'total GA'])

weekly_data = pd.read_excel("C:\\Users\\*****\\OneDrive\\Bracket_Prediction\\PLL\\PLL_Data.xlsx", sheet_name='Weekly Results')

#delete the lists because they are no longer needed
del(nested,nested2)

#initiate an output excel file
writer = pd.ExcelWriter('test.xlsx', engine='openpyxl')

#***********************************************************************************************************************
#Start the calculations

#go through one week at a time
number_of_weeks = 10
for b in range(number_of_weeks):
    #take data for current week, 6 entries per week, so multiply b by 6. This will change if teams are added
    #to the league!
    temp_df = weekly_data.loc[number_of_teams*b:(number_of_teams*b)+5]
    if b == 0:
        df_len = len(temp_df)        

    #Calculate each stat for this week by looping through the temporary dataframe, add it to the total, then use the totals to get to the averages:
    for c in range(number_of_teams*b, (number_of_teams*b)+df_len,1):
        team_id = int(temp_df.loc[c,'Team ID'])
        
        #Goals (including 2 pt goals as 2 points)
        #add the stat for this row's team ID to the correct row of the side_datfram
        side_datfram.loc[team_id, 'total goals'] += temp_df.loc[c,'Goals']
        #now take the average and add it to the main_datfram
        main_datfram.loc[team_id, 'av goals'] = side_datfram.loc[team_id, 'total goals']/(b+1)

        #2 pt shots
        side_datfram.loc[team_id, 'total 2 pt shot'] += temp_df.loc[c,'2 Pt Shots']
        main_datfram.loc[team_id, 'av 2 pt shot'] = side_datfram.loc[team_id, 'total 2 pt shot']/(b+1)

        #2 pt % (total 2 pt goals)
        side_datfram.loc[team_id, 'total 2 pt goals'] += temp_df.loc[c,'2 Pt Shots']*(temp_df.loc[c,'2 Pt %']/100)
        main_datfram.loc[team_id, '2 pt %'] = side_datfram.loc[team_id, 'total 2 pt goals']/side_datfram.loc[team_id, 'total 2 pt shot']

        #Shots
        side_datfram.loc[team_id, 'total shots'] += temp_df.loc[c,'Shots']
        main_datfram.loc[team_id, 'av shots'] = side_datfram.loc[team_id, 'total shots']/(b+1)

        #shot % (Some shooting percentages in the PLL database may be incorrect. Some seem to divide # of goals by shots
        #and others seem to just divide the score by the number of shots.. May be something that needs to be corrected.
        side_datfram.loc[team_id, 'total goals (no 2 pt)'] += temp_df.loc[c,'Shots']*(temp_df.loc[c,'Shots %']/100)
        main_datfram.loc[team_id, 'shot %'] = side_datfram.loc[team_id, 'total goals (no 2 pt)']/side_datfram.loc[team_id, 'total shots']

        #gbs
        side_datfram.loc[team_id, 'total gb'] += temp_df.loc[c,'GB']
        main_datfram.loc[team_id, 'av gb'] = side_datfram.loc[team_id, 'total gb']/(b+1)

        #TOs
        side_datfram.loc[team_id, 'total TO'] += temp_df.loc[c,'TO']
        main_datfram.loc[team_id, 'av TO'] = side_datfram.loc[team_id, 'total TO']/(b+1)

        #CTOs
        side_datfram.loc[team_id, 'total CT'] += temp_df.loc[c,'CT']
        main_datfram.loc[team_id, 'av CT'] = side_datfram.loc[team_id, 'total CT']/(b+1)

        #av PowerPlay
        side_datfram.loc[team_id, 'total PP'] += temp_df.loc[c,'Power Play']
        main_datfram.loc[team_id, 'av PP'] = side_datfram.loc[team_id, 'total PP']/(b+1)

        #PP % : this quantity is reported in successful attempts and a percentage success rate which warrants division of the num
        if temp_df.loc[c,'PP %'] == 0:
            side_datfram.loc[team_id, 'total attempted PP'] += 0
        if temp_df.loc[c,'PP %'] > 0:
            side_datfram.loc[team_id, 'total attempted PP'] += temp_df.loc[c,'Power Play']/(temp_df.loc[c,'PP %']/100)
        if side_datfram.loc[team_id, 'total attempted PP'] > 0:
            main_datfram.loc[team_id, 'PP %'] = side_datfram.loc[team_id, 'total PP']/side_datfram.loc[team_id, 'total attempted PP']

        #Saves
        side_datfram.loc[team_id, 'total Saves'] += temp_df.loc[c,'Saves']
        main_datfram.loc[team_id, 'av Saves'] = side_datfram.loc[team_id, 'total Saves']/(b+1)

        #Save % : Divide saves by the save percentage to get total shots on goal and then divide saves by total shots on goal!
        side_datfram.loc[team_id, 'total shots on goal'] += temp_df.loc[c,'Saves']/(temp_df.loc[c,'Save %']/100)
        main_datfram.loc[team_id, 'Save %'] = side_datfram.loc[team_id, 'total Saves']/side_datfram.loc[team_id, 'total shots on goal']

        #now calculate quantities that depend on the other teams score in the game
        #could do this with a loop OR
        team_info = temp_df.loc[(temp_df['Game ID'] == temp_df.loc[c, 'Game ID']) & (temp_df['Team ID'] != team_id)]
        index = int(team_info.head(1).index.tolist()[0])

        #FO %
        total_FO = (4 + temp_df.loc[c, 'Goals'] + temp_df.loc[index, 'Goals'])
        side_datfram.loc[team_id, 'total FO'] += total_FO
        side_datfram.loc[team_id, 'total FO won'] += total_FO*(temp_df.loc[c,'FO %']/100)
        main_datfram.loc[team_id, 'FO %'] = side_datfram.loc[team_id, 'total FO won']/side_datfram.loc[team_id, 'total FO']
        print(side_datfram.loc[team_id, 'total FO'], side_datfram.loc[team_id, 'total FO won'], main_datfram.loc[team_id, 'FO %'])

        #GAA
        side_datfram.loc[team_id, 'total GA'] += temp_df.loc[index, 'Goals']
        main_datfram.loc[team_id, 'GAA'] = side_datfram.loc[team_id, 'total GA']/(b+1)

        #Point Differential
        main_datfram.loc[team_id, 'Point Differential'] += temp_df.loc[c, 'Goals'] - temp_df.loc[index, 'Goals']

        #Streak
        if temp_df.loc[c, 'Win'] == 1:
            if main_datfram.loc[team_id, 'Streak'] >= 0:
                main_datfram.loc[team_id, 'Streak'] += 1
            if main_datfram.loc[team_id, 'Streak'] < 0:
                main_datfram.loc[team_id, 'Streak'] = 1
        if temp_df.loc[c, 'Win'] == 0:
            if main_datfram.loc[team_id, 'Streak'] > 0:
                main_datfram.loc[team_id, 'Streak'] = -1
            if main_datfram.loc[team_id, 'Streak'] <= 0:
                main_datfram.loc[team_id, 'Streak'] -= 1

    main_datfram.to_excel(writer, sheet_name='yeet'+str(b))


writer.save()      
writer.close()
#print(temp_df.iloc[:,0:17])
#print(side_datfram.iloc[:,0:17])
#print(main_datfram.loc[:,'FO %'])
















