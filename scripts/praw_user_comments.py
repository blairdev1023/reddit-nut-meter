import praw
import csv
import pandas as pd
import numpy as np

def get_user(name):
    '''
    returns a redditor object

    name - string
        name of user, gives an error if the name does not exist
    '''
    return r.redditor(name=name)

def comment_scrape(user):
    '''
    saves a csv of user comments, maximum of 1000 if possible

    user - user object (not a string this time)
    '''
    comms_list = []
    for comment in user.comments.new(limit=None):
        comms_list.append([
                          comment.body,
                          comment.score,
                          comment.subreddit.display_name
                          ])
    df =  pd.DataFrame(data=comms_list)
    df.columns = ['body', 'score', 'sub']
    df.to_csv('../user_data/%s.csv' % user.name, index=False)

def get_user_comments(user):
    '''
    returns a pandas df of the user comments
    the user must have already been scraped and saved as this method
    loads the data from the csv saved in /data

    user - user object (not a string this time)
    '''
    df = pd.read_csv('../user_data/%s.csv' % user.name)
    return df

if __name__ == '__main__':
    USERNAME = 'imnotanotherbot'
    USER_AGENT = 'getting comments'
    CLIENT_ID ='gH8SFf3Q2tklNA'
    CLIENT_SECRET = 'MnMxcuJmB_dpA8-aBR2Ian2aHyE'
    PASSWORD = 'tester'

    r = praw.Reddit(username = USERNAME,
                    user_agent = USER_AGENT,
                    client_id = CLIENT_ID,
                    client_secret = CLIENT_SECRET,
                    password = PASSWORD
                    )

    redditor_name = 'camille11325'
    user = get_user(redditor_name)
    comment_scrape(user)
    df = get_user_comments(user)
