import json
import urllib.request
import boto3

s3 = boto3.client('s3')

def lambda_handler(event, context):
    url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
    filename = '/tmp/boston.csv'
    urllib.request.urlretrieve(url, filename)

    ### s3 へ格納
    s3.upload_file(filename, "aws", "data/boston.csv")

    return {
        'statusCode' : 200,
        'body' : json.dumps('Hello from Lambda!')
    }