import pandas as pd
import numpy as np
from time import time
from os import listdir
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pickle
from nltk.corpus import stopwords

def print_time(message):
    now = time() - start
    now = round(now % 60 + (now - now % 60), 2)
    print('%s, now: %sm%ss' % (message, round(now // 60), round(now % 60)))

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

def get_vectorizer_transform(master_df, mode):
    master_df['body'] = master_df['body'].astype(str)
    if mode.lower() == 'tfidf':
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=1000,
                                     max_features=10000,
                                     ngram_range=(1,2),
                                     stop_words=stop,
                                     lowercase=False
                                     )
    elif mode.lower() == 'bow':
        vectorizer = CountVectorizer(max_df=0.5, min_df=1000,
                                     max_features=10000,
                                     ngram_range=(1,2),
                                     stop_words=stop,
                                     lowercase=False
                                     )
    else:
        print("'mode' not valid. Try 'tfidf' or 'bow'")
    print_time('Fitting %s Vectorizer' % mode)
    vectorizer.fit(master_df['body'].values)
    print_time('Transforming %s Vector' % mode)
    W = vectorizer.transform(master_df['body'])
    return W

def get_model(X, mode, num_topics):
    '''
    returns an lda model. X is the tfidf matrix and nmf
    is the sklearn nmf model that is fitted with X

    master_df - pandas dataframe
        should be your running main dataframe
    '''

    print_time('Fitting %s Model' % mode)
    if mode.lower() == 'nmf':
        model = NMF(n_components=num_topics, init='nndsvda', verbose=1)
    elif mode.lower() == 'lda':
        model = LDA(n_topics=num_topics, max_iter=10, batch_size=1000)
    else:
        print("'mode' not valid. Try 'nmf' or 'lda'")
    model.fit(X)
    print_time('Done Fitting %s Model' % mode)
    return model

def append_topic_idx(master_df, model, X):
    '''
    Tranforms and fetches W from the nmf model using the tfidf/bow data in X.
    Since W is comments vs. topics we find the index of the max index and
    append that column to the master dataframe. Then the master dataframe
    is returned.

    master_df - pandas dataframe
        should be your running master
    model - sklearn nmf/lda model
        needs to be already fitted with the same data, X
    X - pandas dataframe or numpy array
        the same data the nmf was trained on, used for transforming
    '''
    print_time('Transforming X')
    W = model.transform(X)
    print_time('Appending Indicies')
    master_df['topic_idx'] = np.argmax(W, axis=1)
    return master_df

def get_stop_words():
    stop = stopwords.words('english') + ['will', 'would', 'one', 'get',
                                         'like', 'know', 'still', 'got']
    return stop

if __name__ == '__main__':
    start = time()

    num_topics = 50
    stop = get_stop_words()
    master_df = get_master_df()
    X_bow = get_vectorizer_transform(master_df, 'bow')
    X_tfidf = get_vectorizer_transform(master_df, 'tfidf')
    nmf = get_model(X_bow, 'nmf', num_topics)
    lda = get_model(X_tfidf, 'lda', num_topics)

    print_time('Pickling Models')
    with open('pickles/models/nmf__6_14__%s_words.pkl' % num_topics, 'wb') as f:
        pickle.dump(nmf, f)
    with open('pickles/models/lda__6_14__%s_words.pkl' % num_topics, 'wb') as f:
        pickle.dump(lda, f)
    print_time('Pickling X Datum')
    np.save(open('pickles/X/X_bow__6_14.npy', 'wb'), X_bow)
    np.save(open('pickles/X/X_tfidf__6_14.npy', 'wb'), X_tfidf)

    print_time('Labeling Topic Numbers To Comments, Then Pickling Master DF')
    master_df_nmf = append_topic_idx(master_df, nmf, X_bow)
    master_df_lda = append_topic_idx(master_df, lda, X_tfidf)
    master_df_nmf.to_pickle('pickles/masters/master_df__6_14__nmf.pkl')
    master_df_lda.to_pickle('pickles/masters/master_df__6_14__lda.pkl')

    print_time('\nFIN\nFIN\nFIN\nFIN\nFIN\n')
