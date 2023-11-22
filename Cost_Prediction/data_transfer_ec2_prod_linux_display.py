import sys
import os
import pymysql
import boto3
import csv
import datetime
import boto.s3.connection
from boto.s3.key import Key

def getdate(string):
    datestr = ["","",""]
    j = 0
    for i in range(len(string)):
        if j == 3:
            break
        if string[i] == '.':
            break
        if string[i] == '-':
            j += 1
        if (string[i] != '-' and string[i] != '.'):
            datestr[j] += string[i]
    try:
        for k in range(len(datestr)):
            datestr[k] = int(datestr[k])
        return datestr
    except ValueError:
        return [1,1,1999]

def closestdate(trial,rightnow):
    return ((30*rightnow.month)+(rightnow.day)+(365*rightnow.year))-((30*trial[0])+(trial[1])+(365*trial[2]))

output = open("transfer_log.txt",'r')
lines = output.readlines()
readlines = []
for line in lines:
    readlines.append(line.strip('\n'))
output.close()


now = datetime.datetime.now()

conn1 = boto.s3.connect_to_region('us-east-2',
                                  aws_access_key_id='',
                                  aws_secret_access_key='',
                                  is_secure = True,
                                  calling_format = boto.s3.connection.OrdinaryCallingFormat(),
                                  )

bucket = conn1.get_bucket('db-csv-files')
mintime = 1000000000
for key in bucket.list():
    yeah = getdate(key.name)
    mintry = closestdate(yeah,now)
    print(mintry)
    if mintry < mintime:
        mintime = mintry
        minkey = key

print(minkey.name)
found = 0
for h in range(len(readlines)):
    print(minkey.name, readlines[h])
    if minkey.name == readlines[h]:
        found = 1

count = 0
if found == 0:
    connection = pymysql.connect(host = '',
                                 user = 'root',
                                 password = '',
                                 db = 'Cost-Info')

    cursor = connection.cursor()

    transfer = boto3.client('s3',aws_access_key_id='',
                        aws_secret_access_key='')        

    bucket = 'db-csv-files'
    tmpdir = '/tmp/'
    transfer.download_file(bucket, minkey.name,tmpdir+minkey.name)

    cursor.execute("TRUNCATE TABLE Patient_Costs")
    with open(tmpdir+minkey.name, "r") as file:
        read = csv.reader(file, delimiter=',')
        for r in read:
            if r[1] != "DiagCode":
                if r[2] != '0':
                    print(r)
                    cursor.execute("insert into Patient_Costs values ('{0}','{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}','{9}','{10}','{11}')".format(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11]))
                    connection.commit()
                    count += 1

    connection.close()

writestuff = open("transfer_log.txt", 'w+')
print(len(readlines))
writestuff.write("*****************************************************************\n")
writestuff.write("Date and Time: %s\n" % (now))
writestuff.write("Program ran: data_transfer_ec2_test.py\n")
if found == 0:
    writestuff.write("Data imported from:\n%s\n" % (minkey.name))
    writestuff.write("Number of entries imported: %d\n" % (count))
if found == 1:
    writestuff.write("No new data to import.\n")
    writestuff.write("Last imported file: %s\n" % (minkey.name))
writestuff.write("\n\n")
for n in range(len(readlines)):
    writestuff.write(readlines[n])
    writestuff.write("\n")

writestuff.close()
