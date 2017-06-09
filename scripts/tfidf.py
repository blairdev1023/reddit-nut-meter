import pandas as pd
import numpy as np
from time import time
from os import listdir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pickle
from nltk.corpus import stopwords

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
    returns an nmf model and X numpy array. X is the tfidf matrix and nmf
    is the sklearn nmf model that is fitted with X

    master_df - pandas dataframe
        should be your running main dataframe
    '''
    master_df['body'] = master_df['body'].astype(str)
    print('Vectorizing')
    vectorizer = TfidfVectorizer(stop_words=stop,
                                 lowercase=False,
                                 max_features=2000,
                                 min_df = 0.000001,
                                 max_df = 0.1
                                 )
    now = round((time() - start)/60., 2)
    print('Fitting, now: %s Mins' % now)
    vectorizer.fit(master_df['body'].values)
    now = round((time() - start)/60., 2)
    print('Transforming, now: %s Mins' % now)
    X = vectorizer.transform(master_df['body']).todense()
    nmf = NMF(verbose=1, n_components=25)
    now = round((time() - start)/60., 2)
    print('Fitting NMF, now: %s Mins' % now)
    nmf.fit(X)
    now = round((time() - start)/60., 2)
    print('Done Fitting NMF, now: %s Mins' % now)
    return nmf, X

def append_topic_idx(master_df, nmf, X):
    '''
    Tranforms and fetches W from the nmf model using the training data in X.
    Since W is comments vs. topics we find the index of the max index and
    append that column to the master dataframe. Then the master dataframe
    is returned.

    master_df - pandas dataframe
        should be your running master
    nmf - sklearn nmf model
        needs to be already fitted with the same data, X
    X - pandas dataframe or numpy array
        the same data the nmf was trained on, used for transforming
    '''
    now = round((time() - start)/60., 2)
    print('Transforming X, now: %s Mins' % now)
    W = nmf.transform(X)
    now = round((time() - start)/60., 2)
    print('Appending indicies, now: %s Mins' % now)
    master_df['topic_idx'] = np.argmax(W, axis=1)
    return master_df

def get_stop_words():
    stop = stopwords.words('english') + ['will', 'would', 'one', 'get',
                                         'like', 'know', 'still', 'got']
    return stop

if __name__ == '__main__':
    start = time()

    stop = get_stop_words()
    master_df = pd.read_pickle('pickles/master_df.pkl')
    nmf, X = get_tfidf_transform(master_df)

    pickle.dump(nmf, open('pickles/nmf3.pkl', 'wb'))
    np.save(open('pickles/X3.pkl', 'wb'), X)

    master_df = append_topic_idx(master_df, nmf, X)
    master_df.to_pickle('pickles/master_df3.pkl')

    end = time()
    print('This took %s minutes' % round((end - start)/60., 2))
