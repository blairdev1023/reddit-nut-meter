import pandas as pd
import numpy as np
from time import time
from os import listdir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def get_master_df():
    '''
    returns the master dataframe of comments in order of supervised-nuts,
    supervised-not-nut, unsupervised-nuts, unsupervised-not-nuts.
    '''
    s_n = listdir('../data/sup/nuts')
    s_nn = listdir('../data/sup/not_nuts')
    us_n = listdir('../data/un_sup/nuts')
    us_nn = listdir('../data/un_sup/not_nuts')

    s_n.remove('users.txt')
    s_n.remove('user_info.txt')
    s_nn.remove('users.txt')

    csvs = s_n + s_nn + us_n + us_nn
    master = s_n.pop(0)
    master_df = pd.read_csv('../data/sup/nuts/%s' % master)
    for name in s_n:
        name_df = pd.read_csv('../data/sup/nuts/%s' % name)
        master_df = master_df.append(name_df, ignore_index=True)
    for name in s_nn:
        name_df = pd.read_csv('../data/sup/not_nuts/%s' % name)
        master_df = master_df.append(name_df, ignore_index=True)
    for name in us_n:
        name_df = pd.read_csv('../data/un_sup/nuts/%s' % name)
        master_df = master_df.append(name_df, ignore_index=True)
    for name in us_nn:
        name_df = pd.read_csv('../data/un_sup/not_nuts/%s' % name)
        master_df = master_df.append(name_df, ignore_index=True)

    return master_df

def get_tfidf_transform(master_df):
    '''

    '''
    master_df['body'] = master_df['body'].astype(str)
    print('Vectorizing')
    vectorizer = TfidfVectorizer(stop_words='english',
                                 lowercase=False,
                                 max_features=1000,
                                 min_df = 0.00001,
                                 max_df = 0.3
                                 )
    now = round((time() - start)/60., 2)
    print('Fitting, now: %s Mins' % now)
    vectorizer.fit(master_df['body'].values)
    now = round((time() - start)/60., 2)
    print('Transforming, now: %s Mins' % now)
    X = vectorizer.transform(master_df['body']).todense()
    nmf = NMF(verbose=1, n_components=20)
    now = round((time() - start)/60., 2)
    print('Fitting NMF, now: %s Mins' % now)
    nmf.fit(X)
    return nmf, X

if __name__ == '__main__':
    start = time()

    master_df = pd.read_pickle('pickles/master_df.pkl')
    nmf, X = get_tfidf_transform(master_df)

    end = time()
    print('This took %s minutes' % round((end - start)/60., 2))
