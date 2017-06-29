import pandas as pd
import numpy as np
from time import time
from os import listdir
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pickle
from nltk.corpus import stopwords

def get_stop_words():
    return stopwords.words('english') + ['will', 'would', 'one', 'get',
                                         'like', 'know', 'still', 'got']

def print_time(message, start):
    now = time() - start
    now = round(now % 60 + (now - now % 60), 2)
    print('%s, now: %sm%ss' % (message, round(now // 60), round(now % 60)))

def get_master_dfs():
    '''
    Makes two pandas dataframes (df). The first df (master_df_train) consists
    of the comments in order of supervised-nuts, supervised-not-nuts,
    unsupervised-nuts, and unsupervised-not-nuts. The second df
    (master_df_test) consists of the comments from test-nuts and test-non-nuts.
    The test data is of course supervised.

    Returns: master_df_train, master_df_test
    '''
    s_n = listdir('../data/sup/nuts')
    s_nn = listdir('../data/sup/not_nuts')
    us_n = listdir('../data/un_sup/nuts')
    us_nn = listdir('../data/un_sup/not_nuts')
    t_n = listdir('../data/test/nuts')
    t_nn = listdir('../data/test/not_nuts')

    print_time('Getting Master DataFrames', start)
    master = s_n.pop(0)
    master_df_train = pd.read_csv('../data/sup/nuts/%s' % master)
    for name in s_n:
        name_df = pd.read_csv('../data/sup/nuts/%s' % name)
        master_df_train = master_df_train.append(name_df, ignore_index=True)
    for name in s_nn:
        name_df = pd.read_csv('../data/sup/not_nuts/%s' % name)
        master_df_train = master_df_train.append(name_df, ignore_index=True)
    for name in us_n:
        name_df = pd.read_csv('../data/un_sup/nuts/%s' % name)
        master_df_train = master_df_train.append(name_df, ignore_index=True)
    for name in us_nn:
        name_df = pd.read_csv('../data/un_sup/not_nuts/%s' % name)
        master_df_train = master_df_train.append(name_df, ignore_index=True)

    master = t_n.pop(0)
    master_df_test = pd.read_csv('../data/test/nuts/%s' % master)
    for name in t_n:
        name_df = pd.read_csv('../data/test/nuts/%s' % name)
        master_df_test = master_df_test.append(name_df, ignore_index=True)
    for name in t_nn:
        name_df = pd.read_csv('../data/test/not_nuts/%s' % name)
        master_df_test = master_df_test.append(name_df, ignore_index=True)

    master_df_train['body'] = master_df_train['body'].astype(str)
    master_df_test['body'] = master_df_test['body'].astype(str)

    return master_df_train, master_df_test

def vectorizer_fit_transform(master_dfs, mode):
    '''
    Takes the comments from the train df and fits them to your vectorizer
    Then transforms the vectorizer on the training data (X_train) and then on
    the testing data (X_test). The returned vectorizer is fit FYI.

    Arguement 'mode' should be either 'bow' or 'tfidf'

    Returns: vectorizer, (X_train, X_test)
    '''
    print_time('Fitting %s Vectorizer' % mode.upper(), start)
    if mode.lower() == 'tfidf':
        vectorizer = TfidfVectorizer(max_df=0.8, min_df=50,
                                     max_features=10000,
                                     ngram_range=(1,2),
                                     stop_words=stop,
                                     lowercase=False
                                     )
    elif mode.lower() == 'bow':
        vectorizer = CountVectorizer(max_df=0.8, min_df=50,
                                     max_features=10000,
                                     ngram_range=(1,2),
                                     stop_words=stop,
                                     lowercase=False
                                     )
    else:
        print("'mode' not valid. Try 'tfidf' or 'bow'")
    master_df_train, master_df_test = master_dfs
    vectorizer.fit(master_df_train['body'].values)
    print_time('Done Fitting %s Vectorizer, Transforming.'% mode.upper(), start)
    X_train = vectorizer.transform(master_df_train['body'])
    X_test = vectorizer.transform(master_df_test['body'])

    return vectorizer, (X_train, X_test)

def model_fit_transform(X, mode, n_topics):
    '''
    Takes the vectorizer transform matricies as inputs. Fits your model with
    X_train then transforms X_train as W_train. X_test is the transformed
    into W_test. The returned model is fit FYI.

    Arguement 'mode' should be set to either 'nmf' or 'lda'.

    Returns: model, (W_train, W_test)
    '''
    start_func = time()
    print_time('Fitting %s Model' % mode.upper(), start)
    if mode.lower() == 'nmf':
        model = NMF(n_components=n_topics, init='nndsvda', verbose=1)
    elif mode.lower() == 'lda':
        model = LDA(n_topics=n_topics, max_iter=10, batch_size=1000)
    else:
        print("Arguement 'mode' not valid. Try 'nmf' or 'lda'")
    X_train, X_test = X
    model.fit(X_train)
    print_time('Done Fitting %s, That Took' % mode.upper(), start_func)
    print_time('Transforming', start)
    W_train = model.transform(X_train)
    W_test = model.transform(X_test)

    return model, (W_train, W_test)

def append_topic_idx(master_dfs, W):
    '''
    Takes the train and test dfs and uses the transform matricies (W)
    to label each comment as a topic number (topic_idx). The topic idx is
    taken as the maximum of that comment's row in the W matrix. Returns
    master_df_train and master_df_test.
    '''
    print_time('Appending Indicies', start)
    df_train, df_test = master_dfs
    master_df_train, master_df_test = df_train.copy(), df_test.copy()
    W_train, W_test = W
    master_df_train['topic_idx'] = np.argmax(W_train, axis=1)
    master_df_test['topic_idx'] = np.argmax(W_test, axis=1)

    return master_df_train, master_df_test

def save_object(n_topics, master_dfs_nmf, master_dfs_lda,
                nmf, lda, vectorizer_tfidf, vectorizer_bow):
    '''
    Saves the master_dfs, models, and vectorizers in either /pickles/nmf or
    /pickles/lda depending on model type.
    '''
    print_time('Pickling', start)

    nmf_dir = 'pickles/nmf/%s_topics/' % n_topics
    lda_dir = 'pickles/lda/%s_topics/' % n_topics

    master_df_train_nmf, master_df_test_nmf = master_dfs_nmf
    master_df_train_nmf.to_pickle(nmf_dir + 'master_df_train.pkl')
    master_df_test_nmf.to_pickle(nmf_dir + 'master_df_test.pkl')
    master_df_train_lda, master_df_test_lda = master_dfs_lda
    master_df_train_lda.to_pickle(lda_dir + 'master_df_train.pkl')
    master_df_test_lda.to_pickle(lda_dir + 'master_df_test.pkl')

    pickle.dump(nmf, open(nmf_dir + 'model.pkl', 'wb'))
    pickle.dump(lda, open(lda_dir + 'model.pkl', 'wb'))
    pickle.dump(vectorizer_tfidf, open(nmf_dir + 'vectorizer.pkl', 'wb'))
    pickle.dump(vectorizer_bow, open(lda_dir + 'vectorizer.pkl', 'wb'))

    print_time('Done Pickling', start)

if __name__ == '__main__':
    # this script should be ran as: python topic_modeling.py n_topics
    # where n_topics should be an integar
    start = time()

    n_topics = int(sys.argv[1])
    stop = get_stop_words()
    master_dfs = get_master_dfs()
    # vectorizer_bow, X_bow = vectorizer_fit_transform(master_dfs, 'bow')
    # vectorizer_tfidf, X_tfidf = vectorizer_fit_transform(master_dfs, 'tfidf')
    # nmf, W_nmf = model_fit_transform(X_tfidf, 'nmf', n_topics)
    # lda, W_lda = model_fit_transform(X_bow, 'lda', n_topics)

    tfidf = pickle.load(open('pickles/nmf/%s_topics/vectorizer.pkl' % n_topics, 'rb'))
    bow = pickle.load(open('pickles/lda/%s_topics/vectorizer.pkl' % n_topics, 'rb'))
    X_train_tfidf = tfidf.transform(master_dfs[0]['body'])
    X_test_tfidf = tfidf.transform(master_dfs[1]['body'])
    X_train_bow = bow.transform(master_dfs[0]['body'])
    X_test_bow = bow.transform(master_dfs[1]['body'])
    nmf = pickle.load(open('pickles/nmf/%s_topics/model.pkl' % n_topics, 'rb'))
    lda = pickle.load(open('pickles/lda/%s_topics/model.pkl' % n_topics, 'rb'))
    W_train_nmf = nmf.transform(X_train_tfidf)
    W_test_nmf = nmf.transform(X_test_tfidf)
    W_train_lda = lda.transform(X_train_bow)
    W_test_lda = lda.transform(X_test_bow)
    W_nmf = (W_train_nmf, W_test_nmf)
    W_lda = (W_train_lda, W_test_lda)

    master_dfs_nmf = append_topic_idx(master_dfs, W_nmf)
    master_dfs_lda = append_topic_idx(master_dfs, W_lda)

    master_df_train_nmf, master_df_test_nmf = master_dfs_nmf
    master_df_train_lda, master_df_test_lda = master_dfs_lda

    master_df_train_nmf.to_pickle('pickles/nmf/%s_topics/master_df_train.pkl' % n_topics)
    master_df_test_nmf.to_pickle('pickles/nmf/%s_topics/master_df_test.pkl' % n_topics)
    master_df_train_lda.to_pickle('pickles/lda/%s_topics/master_df_train.pkl' % n_topics)
    master_df_test_lda.to_pickle('pickles/lda/%s_topics/master_df_test.pkl' % n_topics)
    # save_object(n_topics, master_dfs_nmf, master_dfs_lda,
                    # nmf, lda, vectorizer_tfidf, vectorizer_bow)

    print_time('\nFIN\nFIN\nFIN\nFIN\nFIN', start)
