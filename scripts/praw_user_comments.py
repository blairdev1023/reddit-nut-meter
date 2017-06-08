import praw
import pandas as pd
import numpy as np
import re
from time import time

def get_user(name):
    '''
    returns a redditor object

    name - string
        name of user, gives an error if the name does not exist
    '''
    return r.redditor(name=name)

def comment_scrape(user, direct, nut=False):
    '''
    saves a csv of user comments, maximum of 1000 if possible

    user - user object (not a string this time)
    direct - string
        'sup' - saves data in the data/sup directory
        'un_sup' - saves data in the data/un_sup directory
    nut - bool
        if False, saves data in data/<direct>/not_nuts directory
        if True, saves data in data/<direct>/nuts directory
    '''
    comms_list = []
    for comment in user.comments.new(limit=None):
        comms_list.append([
                          user.name,
                          clean_body(comment.body),
                          comment.score,
                          comment.subreddit.display_name
                          ])
    df =  pd.DataFrame(data=comms_list,
                       columns = ['name', 'body', 'score', 'sub'])
    if (direct == 'sup') & (nut == False):
        df.to_csv('../data/sup/not_nuts/%s.csv' % user.name, index=False)
    elif (direct == 'sup') & (nut == True):
        df.to_csv('../data/sup/nuts/%s.csv' % user.name, index=False)
    elif (direct == 'un_sup') & (nut == False):
        df.to_csv('../data/un_sup/not_nuts/%s.csv' % user.name, index=False)
    elif (direct == 'un_sup') & (nut == True):
        df.to_csv('../data/un_sup/nuts/%s.csv' % user.name, index=False)
    else:
        print('directory of \'sup\' or \'un_sup\' not specified, did not save')

def get_user_comments(user):
    '''
    returns a pandas df of the user comments
    the user must have already been scraped and saved as this method
    loads the data from the csv saved in /data

    user - user object (not a string this time)
    direct - string
        'sup' - sends this data to the data/sup/ directory
        'un_sup' - sends this data to the data/un_sup directory
    '''
    df = pd.read_csv('../data/%s.csv' % user.name)
    return df

def clean_body(text):
    '''
    returns a string of the "cleaned" body of a comment

    text - string
        the comment.body should be entered here
    '''
    text = text.lower()
    text = re.sub(r"\({3}[\S]+\){3}", "abcdefghijklmnopqrstuvxyz", text)
    text = re.sub(r"[^A-Za-z0-9']", " ", text)
    text = re.sub(r"abcdefghijklmnopqrstuvxyz", "((()))", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" i e ", " ie ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def supervised_nuts_scrape():
    '''
    gets all WN and MIS from pickled df's, pulls those users comments,
    cleans them, and saves them as a csv in the ../data/sup/nuts directory
    '''
    MIS_list = pd.read_pickle('pickles/dfMIS.pkl').values.tolist()
    WN_list = pd.read_pickle('pickles/dfWN.pkl').values.tolist()

    for name in MIS_list:
        user = get_user(name[0])
        comment_scrape(user, 'sup', nut=True)
    for name in WN_list:
        user = get_user(name[0])
        comment_scrape(user, 'sup', nut=True)

def unsupervised_nuts_scrape():
    '''
    gets the specified number of user names from the mentioned subs.
    data is saved in ../data/un_sup/nuts directory as a csv
    '''
    MIS_sub_dict = {'theredpill': 30,
                'pussypassdenied': 30,
                'pussypass': 13,
                'MGTOW': 13,
                'incels': 7,
                'redpillwomen': 7
                }
    WN_sub_dict = {'the_donald': 40,
               'conspiracy': 40,
               'uncensorednews': 10,
               'sjwhate': 8,
               'whiterights': 2,
               'pol': 2
               }
    MIS_user_list, WN_user_list = [], []

    print('geting user names')

    for sub in MIS_sub_dict.keys():
        n = MIS_sub_dict[sub]
        i = 0
        comments = r.subreddit(sub).comments(limit=100)
        for comment in comments:
            if comment == None:
                break
            author = comment.author.name
            if author.lower() == 'automoderator':
                break
            if author not in MIS_user_list:
                MIS_user_list.append(author)
                i += 1
                if i == n:
                    break

    for sub in WN_sub_dict.keys():
        n = WN_sub_dict[sub]
        i = 0
        comments = r.subreddit(sub).comments(limit=100)
        for comment in comments:
            if comment == None:
                break
            author = comment.author.name
            if author.lower() == 'automoderator':
                break
            if author not in WN_user_list:
                WN_user_list.append(author)
                i += 1
                if i == n:
                    break

    print('getting user comments')

    i, j = 0, 0
    n_MIS, n_WN, = len(MIS_user_list), len(WN_user_list)
    for name in MIS_user_list:
        now = round((time() - start)/60., 2)
        print('%s, %s/%s users retrieved %s Mins' % (name, i, n_MIS, now))
        user = get_user(name)
        comment_scrape(user, 'un_sup', nut=True)
        i += 1
    for name in WN_user_list:
        now = round((time() - start)/60., 2)
        print('%s, %s/%s users retrieved %s Mins' % (name, j, n_WN, now))
        user = get_user(name)
        comment_scrape(user, 'un_sup', nut=True)
        j += 1

def unsupervised_not_nuts_scrape():
    '''
    gets 20 user names from the mentioned subs.
    data is saved in ../data/un_sup/not_nuts directory as a csv
    '''
    not_nuts_subs = ['gaming',
                  'lifeprotips',
                  'eathporn',
                  'sports',
                  'history',
                  'food',
                  'get_motivated',
                  'creepy',
                  'technology',
                  'aww'
                  ]

    print('geting user names')

    not_nuts_users = []
    for sub in not_nuts_subs:
        n = 2
        i = 0
        comments = r.subreddit(sub).comments(limit=100)
        for comment in comments:
            if comment != None:
                author = comment.author.name
                if (author not in not_nuts_users) & (author.lower() != 'automoderator'):
                    not_nuts_users.append(author)
                    i += 1
                    if i == n:
                        break

    print('getting user comments')

    i = 0
    n = 20
    for name in not_nuts_users:
        now = round((time() - start)/60., 2)
        print('%s, %s/%s users retrieved %s Mins' % (name, i, n, now))
        user = get_user(name)
        comment_scrape(user, 'un_sup', nut=False)
        i += 1

def supervised_not_nuts_scrape():
    '''
    gets all normies from its pickled df, pulls those users comments,
    cleans them, and saves them as a csv in the ../data/sup/nuts directory
    '''
    normies_list = pd.read_pickle('pickles/not_nuts_df.pkl').values.tolist()

    for name in normies_list:
        user = get_user(name[0])
        comment_scrape(user, 'sup', nut=False)

if __name__ == '__main__':
    USERNAME = 'imnotanotherbot'
    USER_AGENT = 'getting comments' # put whatever here
    CLIENT_ID ='gH8SFf3Q2tklNA'
    CLIENT_SECRET = 'MnMxcuJmB_dpA8-aBR2Ian2aHyE'
    PASSWORD = 'tester'

    r = praw.Reddit(username = USERNAME,
                    user_agent = USER_AGENT,
                    client_id = CLIENT_ID,
                    client_secret = CLIENT_SECRET,
                    password = PASSWORD
                    )

    print('starting...')
    start = time()
    unsupervised_not_nuts_scrape()
    end = time()
    print('This took %s minutes' % round(float(end - start)/60, 2))
