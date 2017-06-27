import sys
from os import listdir
import pandas as pd
import numpy as np
from time import time
from topic_modeling import print_time
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report as class_report
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
from ModelThreshold import ModelThreshold

def z_score(observation, mean, std):
    '''
    Calculates the z-score associated with the observation value given a
    normal distribution of a known mean and standard deviation

    Returns: z_score
    '''
    return (observation - mean) / std

def load_split(topic_dfs_model):
    '''
    Loads and splits data by nmf/lda, y/X, train/test in that order.

    Returns: X_train, y_train, X_test, y_test
    '''
    topic_df_train, topic_df_test = topic_dfs_model
    df_train, df_test = topic_df_train.copy(), topic_df_test.copy()
    y_train = df_train.pop('is_nut').values
    y_test = df_test.pop('is_nut').values
    X_train, X_test = df_train.values, df_test.values

    return X_train, y_train, X_test, y_test

def get_name_lists():
    '''
    Returns 6 lists. Each one has the names of belonging to each group
    '''
    s_n = listdir('../data/sup/nuts')
    s_nn = listdir('../data/sup/not_nuts')
    us_n = listdir('../data/un_sup/nuts')
    us_nn = listdir('../data/un_sup/not_nuts')
    t_n = listdir('../data/test/nuts')
    t_nn = listdir('../data/test/not_nuts')

    # removes .csv extension
    s_n = list(map(lambda x: x[:-4], s_n))
    s_nn = list(map(lambda x: x[:-4], s_nn))
    us_n = list(map(lambda x: x[:-4], us_n))
    us_nn = list(map(lambda x: x[:-4], us_nn))
    t_n = list(map(lambda x: x[:-4], t_n))
    t_nn = list(map(lambda x: x[:-4], t_nn))

    for l in [s_n, s_nn, us_n, us_nn, t_n, t_nn]:
        if '.DS_S' in l:
            l.remove('.DS_S')

    return s_n, s_nn, us_n, us_nn, t_n, t_nn

def get_master_dfs(n_topics, names_lists):
    '''
    Unpickles master_dfs for both models. Each being a tuple of the
    training df and a testing df for that model.

    Returns: master_dfs_nmf, master_dfs_lda
    '''
    print_time('Getting Master DataFrames', start)

    nmf_dir = 'pickles/nmf/%s_topics/' % n_topics
    lda_dir = 'pickles/lda/%s_topics/' % n_topics
    master_df_nmf_train = pd.read_pickle(nmf_dir + 'master_df_train.pkl')
    master_df_nmf_test = pd.read_pickle(nmf_dir + 'master_df_test.pkl')
    master_df_lda_train = pd.read_pickle(lda_dir + 'master_df_train.pkl')
    master_df_lda_test = pd.read_pickle(lda_dir + 'master_df_test.pkl')

    s_n, s_nn, us_n, us_nn, t_n, t_nn = names_lists
    mask_nmf = master_df_nmf_train['name'].apply(lambda x:
                                                   (x in s_n) | (x in s_nn))
    mask_lda = master_df_lda_train['name'].apply(lambda x:
                                                   (x in s_n) | (x in s_nn))
    master_df_nmf_train = master_df_nmf_train[mask_nmf]
    master_df_lda_train = master_df_lda_train[mask_lda]

    master_dfs_nmf = (master_df_nmf_train, master_df_nmf_test)
    master_dfs_lda = (master_df_lda_train, master_df_lda_test)

    return master_dfs_nmf, master_dfs_lda

def get_grouby_by_dfs(master_dfs_model):
    '''
    Takes the training and testing dfs from master_dfs_model and aggregates the
    comment count per topic for each user. The returned dfs are indexed by name
    and have columns 'topic_idx' and 'count'.

    Returns: score_agg_df_train, score_agg_df_test
    '''
    print_time('Getting Group-By', start)

    master_df_train, master_df_test = master_dfs_model
    gb_train = master_df_train.groupby(['name', 'topic_idx'])
    gb_test = master_df_test.groupby(['name', 'topic_idx'])
    score_agg_df_train = gb_train['score'].aggregate(['count'])
    score_agg_df_test = gb_test['score'].aggregate(['count'])

    return score_agg_df_train, score_agg_df_test

def standardize_gb(score_agg_dfs, n_topics, names_lists):
    '''
    Standardizes the counts per topic for the training data and testing data
    seperately. The dfs returned have the names as indicies, topic indicies as
    columns, and the values are that users standardized count in that topic.
    Last column that is appeneded is 'is_nut', with a 1 if True.

    Returns: topic_df_train, topic_df_test
    '''
    # get standardized data
    print_time('Standardizing', start)
    score_agg_df_train, score_agg_df_test = score_agg_dfs
    df_train, df_test = score_agg_df_train.copy(), score_agg_df_test.copy()

    topic_agg = np.zeros((n_topics, 2))
    for i in range(n_topics):
        topic_agg[i, 0] = df_train.loc(axis=0)[:,i].mean()
        topic_agg[i, 1] = df_train.loc(axis=0)[:,i].std()

    for idx in df_train.index:
        mean, std = topic_agg[idx[1]]
        df_train.loc[idx] = z_score(df_train.loc[idx], mean, std)
    for idx in df_test.index:
        mean, std = topic_agg[idx[1]]
        df_test.loc[idx] = z_score(df_test.loc[idx], mean, std)

    # make dataframes of names vs. topic (topic_df)
    print_time('Making Topic DataFrame', start)
    names_train = df_train.index.get_level_values(0).unique().tolist()
    names_test = df_test.index.get_level_values(0).unique().tolist()
    topic_data_train = np.zeros((len(names_train), n_topics))
    topic_data_test = np.zeros((len(names_test), n_topics))

    for i in range(len(names_train)):
        for idx in df_train.loc[names_train[i]].index:
            topic_data_train[i, idx] = df_train.loc[(names_train[i], idx)]
    for i in range(len(names_test)):
        for idx in df_test.loc[names_test[i]].index:
            topic_data_test[i, idx] = df_test.loc[(names_test[i], idx)]

    topic_df_train = pd.DataFrame(data=topic_data_train, index=names_train)
    topic_df_test = pd.DataFrame(data=topic_data_test, index=names_test)

    # labeling nuts
    topic_df_train['is_nut'] = np.zeros(len(names_train))
    topic_df_test['is_nut'] = np.zeros(len(names_test))
    for name in names_lists[0]:
        topic_df_train.loc[name]['is_nut'] = 1.0
    for name in names_lists[4]:
        topic_df_test.loc[name]['is_nut'] = 1.0

    return topic_df_train, topic_df_test

def model_gridsearch(topic_dfs_model):
    '''
    Takes the topic_dfs and splits them into X & y. Then using X_train and
    y_train GridSearchs over an AdaBoostClassifier, GradientBoostingClassifier,
    and RandomForestClassifier with their respective paramaters. Returns fitted
    GridSearch objects.

    Not used in normal if_name_main block because GS objects were pickled in
    script and then the saving part of the code was deleted. Not needed since
    pickles exist.

    Returns: gs_abc, gs_gbc, gs_rfc
    '''
    print_time('Starting GridSearch', start)
    topic_df_train, topic_df_test = topic_dfs_model
    y_train = topic_df_train.pop('is_nut').values
    y_test = topic_df_test.pop('is_nut').values
    X_train, X_test = topic_df_train.values, topic_df_test.values

    abc = ABC()
    gbc = GBC()
    rfc = RFC()
    params_abc = {'n_estimators': [50, 100, 150],
                  'learning_rate': [0.01, 0.1, 0.5, 1.0]}
    params_gbc = {'n_estimators': [200, 250, 300, 350],
                  'learning_rate': [0.01, 0.1, 0.5, 1.0],
                  'max_depth': [2, 3],
                  'subsample': [0.05, 0.1, 0.5, 0.75, 1.0],
                  'max_features': ['auto', 'sqrt', 'log2']}
    params_rfc = {'n_estimators': [200, 250, 300, 350],
                  'max_features': ['auto', 'sqrt', 'log2']}
    gs_abc = GridSearchCV(abc, params_abc)
    gs_gbc = GridSearchCV(gbc, params_gbc)
    gs_rfc = GridSearchCV(rfc, params_rfc)

    grids = {'abc': gs_abc, 'gbc': gs_gbc, 'rfc': gs_rfc}

    for key in grids.keys():
        start_func = time()
        print_time('Starting %s GridSearch' % key, start)
        grids[key].fit(X_train, y_train)
        print_time('Done Searching for %s' % key, start_func)

    return gs_abc, gs_gbc, gs_rfc

def model_fit(topic_dfs_nmf, topic_dfs_lda, n_topics):
    '''
    Loads this script's n_topics GridSearch pickles. Using the GridSearch
    best_params_ attribute makes three models for nmf and lda. Those models are
    an ABC, GBC, and RFC. They are fitted with their respective training data
    and all models are pickled.
    '''
    print_time('Fitting and Saving Models')
    nmf_X_train, nmf_y_train = list(load_split(topic_dfs_nmf))[:2]
    lda_X_train, lda_y_train = list(load_split(topic_dfs_lda))[:2]

    # load GS objects
    nmf_dir = 'pickles/nmf/%s_topics/' % n_topics
    nmf_gs_abc = pickle.load(open(nmf_dir + 'gs_abc.pkl', 'rb'))
    nmf_gs_gbc = pickle.load(open(nmf_dir + 'gs_gbc.pkl', 'rb'))
    nmf_gs_rfc = pickle.load(open(nmf_dir + 'gs_rfc.pkl', 'rb'))
    lda_dir = 'pickles/lda/%s_topics/' % n_topics
    lda_gs_abc = pickle.load(open(lda_dir + 'gs_abc.pkl', 'rb'))
    lda_gs_gbc = pickle.load(open(lda_dir + 'gs_gbc.pkl', 'rb'))
    lda_gs_rfc = pickle.load(open(lda_dir + 'gs_rfc.pkl', 'rb'))

    # instantiates, fits, and saves models
    nmf_abc = ABC(**nmf_gs_abc.best_params_)
    nmf_gbc = GBC(**nmf_gs_gbc.best_params_)
    nmf_rfc = RFC(**nmf_gs_rfc.best_params_)
    lda_abc = ABC(**lda_gs_abc.best_params_)
    lda_gbc = GBC(**lda_gs_gbc.best_params_)
    lda_rfc = RFC(**lda_gs_rfc.best_params_)

    nmf_abc.fit(nmf_X_train, nmf_y_train)
    nmf_gbc.fit(nmf_X_train, nmf_y_train)
    nmf_rfc.fit(nmf_X_train, nmf_y_train)
    lda_abc.fit(lda_X_train, lda_y_train)
    lda_gbc.fit(lda_X_train, lda_y_train)
    lda_rfc.fit(lda_X_train, lda_y_train)

    pickle.dump(nmf_abc, open(nmf_dir + 'model_abc.pkl', 'wb'))
    pickle.dump(nmf_gbc, open(nmf_dir + 'model_gbc.pkl', 'wb'))
    pickle.dump(nmf_rfc, open(nmf_dir + 'model_rfc.pkl', 'wb'))
    pickle.dump(lda_abc, open(lda_dir + 'model_abc.pkl', 'wb'))
    pickle.dump(lda_gbc, open(lda_dir + 'model_gbc.pkl', 'wb'))
    pickle.dump(lda_rfc, open(lda_dir + 'model_rfc.pkl', 'wb'))

def pred_prob(topic_dfs_nmf, topic_dfs_lda, n_topics):
    '''
    Loads this script's n_topics abc/gbc/rfc model pickles. Returns predict_proba of each model using either nmf or lda X_test.

    Returns: y_probs_nmf, y_probs_lda
    '''
    print_time('Predicting Probabilities', start)

    nmf_dir = 'pickles/nmf/%s_topics/' % n_topics
    nmf_abc = pickle.load(open(nmf_dir + 'model_abc.pkl', 'rb'))
    nmf_gbc = pickle.load(open(nmf_dir + 'model_gbc.pkl', 'rb'))
    nmf_rfc = pickle.load(open(nmf_dir + 'model_rfc.pkl', 'rb'))
    lda_dir = 'pickles/lda/%s_topics/' % n_topics
    lda_abc = pickle.load(open(lda_dir + 'model_abc.pkl', 'rb'))
    lda_gbc = pickle.load(open(lda_dir + 'model_gbc.pkl', 'rb'))
    lda_rfc = pickle.load(open(lda_dir + 'model_rfc.pkl', 'rb'))

    nmf_X_test = list(load_split(topic_dfs_nmf))[2]
    lda_X_test = list(load_split(topic_dfs_lda))[2]

    # get probabilities
    nmf_abc_y_prob = nmf_abc.predict_proba(nmf_X_test)
    nmf_gbc_y_prob = nmf_gbc.predict_proba(nmf_X_test)
    nmf_rfc_y_prob = nmf_rfc.predict_proba(nmf_X_test)
    lda_abc_y_prob = lda_abc.predict_proba(lda_X_test)
    lda_gbc_y_prob = lda_gbc.predict_proba(lda_X_test)
    lda_rfc_y_prob = lda_rfc.predict_proba(lda_X_test)

    y_probs_nmf = (nmf_abc_y_prob, nmf_gbc_y_prob, nmf_rfc_y_prob)
    y_probs_lda = (lda_abc_y_prob, lda_gbc_y_prob, lda_rfc_y_prob)

    return y_probs_nmf, y_probs_lda

def plot_roc(mts, ax, models, thresholds=np.arange(0, 1.05, 0.05), roc=True):
    '''

    mts - list
        each mt in the this list should have data loaded into it along with relevant terms made via the in-class methods (see ModelThreshold.py)
    ax - matplotlib axes object
        the already instantiated axis that will have the plots made on it
    models - list of tuples
        each tuple: (topic, pred, n_topics) where topic is a string of 'NMF' or 'LDA', pred is a string of 'abc', 'gbc', or 'rfc', and n_topics is an int multiple of 25 between 25 and 150
    thresholds - list or iterable
        optional array of percent thresholds to be plotted for a roc, default is 0-1 in increments of 0.05
    roc - bool
        optional bool argument that if False, makes a 'normal' plot instead of a roc plot
    '''
    color_dict = {'nmf': 'r', 'lda': 'b'}
    ls_dict = {'abc': '--', 'gbc': ':', 'rfc': '-'}

    if roc == True:
        for model in models:
            topic = model[0].lower()
            pred = model[1]
            n_topics = model[2]
            if len(mts) > 1:
                mt = mts[int(n_topics / 25 - 1)]
            else:
                mt = mts[0]
            df = eval('mt.%s_%s_df' % (topic[:2], pred))

            x = df['fpr'].values
            y = df['tpr'].values
            label = '%s, %s, %s topics' % (topic.upper(),pred.upper(),n_topics)
            color = color_dict[topic]
            ls = ls_dict[pred]
            ax.plot(x, y, label=label,color=color, ls=ls,
                    marker='.')
        ax.plot(thresholds, thresholds, ls='--', color='k')

if __name__ == '__main__':
    start = time()

    n_topic = int(sys.argv[1])
    names_lists = get_name_lists()
    topic_numbers = [n_topic]
    mts = []

    for n_topics in topic_numbers:
        master_dfs_nmf, master_dfs_lda = get_master_dfs(n_topics, names_lists)

        score_agg_dfs_nmf = get_grouby_by_dfs(master_dfs_nmf)
        score_agg_dfs_lda = get_grouby_by_dfs(master_dfs_lda)

        topic_dfs_nmf = standardize_gb(score_agg_dfs_nmf, n_topics, names_lists)
        topic_dfs_lda = standardize_gb(score_agg_dfs_lda, n_topics, names_lists)

        y_probs_nmf, y_probs_lda = pred_prob(topic_dfs_nmf, topic_dfs_lda,
                                             n_topics)
        mt = ModelThreshold(y_probs_nmf, y_probs_lda,
                            topic_dfs_nmf, topic_dfs_lda)
        mt.confusion_terms(terms=['fpr', 'tpr', 'ppv', 'f1'])
        mts.append(mt)


    models = [('nmf', 'abc', n_topic),
              ('nmf', 'gbc', n_topic),
              ('nmf', 'rfc', n_topic),
              ('lda', 'abc', n_topic),
              ('lda', 'gbc', n_topic),
              ('lda', 'rfc', n_topic),              ]
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    plot_roc(mts, ax, models)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve of NMF and LDA models with %s topics' % n_topic)
    plt.legend(loc='best')
    plt.savefig('../images/%s_topics.png' % n_topic)
    plt.close()

    print_time('Done', start)
