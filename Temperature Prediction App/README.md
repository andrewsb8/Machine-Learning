These are the files used for a very basic gunicorn/nginx web app that predicts the high/low daily temperature using the previous day's
high/low temperatures and a decision tree model. These files were placed on a remote Ubuntu server and in a virtual environment.

The files do the following:
1. execute_update: Bash script that runs yesterday's high/low temps into the trained model to determine today's high/low temps. So the data is updated daily.
2. temp_json_file.json: Output of above line
3. temp_info_update.py: python script that is carried out by execute_update
4. root_high(low)_temp_1day: Serialized trained decision tree models for high (low) temps
5. log.txt: Supposed to store all of the day's outputs from (3). Ended project before it got this far. Could easily just write an extra line to (3) to add this functionality but server is no longer active.
6. prediction_app.py: flask app managed by gunicorn/nginx that would send temp_json_file.json to end user when called on host machine
