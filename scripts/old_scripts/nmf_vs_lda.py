from topic_analysis import get_name_lists, get_grouby_by_df
from topic_analysis import get_val_dicts, get_dict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from time import time


def print_time(message):
    now = time() - start
    now = round(now % 60 + (now - now % 60), 2)
    print('%s, now: %sm%ss' % (message, round(now // 60), round(now % 60)))

def z_score(observation, mean, std):
    '''
    returns the z-score associated with the observation value given a
    normal distribution of a known mean and standard deviation
    '''
    return (observation - mean) / std

def get_master_df(n_topics):
    nmf_dir = 'pickles/masters/master_df__6_15__nmf__%s.pkl' % n_topics
    lda_dir = 'pickles/masters/master_df__6_15__lda__%s.pkl' % n_topics
    master_df_nmf = pd.read_pickle(nmf_dir)
    master_df_lda = pd.read_pickle(lda_dir)
    return master_df_nmf, master_df_lda

def get_mean_std_dicts(type_val_dicts):
    '''
    Returns 2 tuples, each containing 4 dictionaries. Each dictionary
    represents one of the 4 groups and has the key set to a topic # and each
    value to be the type mean/std/count depending on which tuple is examined
    '''
    s_n_val, s_nn_val, us_n_val, us_nn_val = type_val_dicts

    s_n_mean = {x: np.array(y).mean() for x, y in s_n_val.items()}
    s_nn_mean = {x: np.array(y).mean() for x, y in s_nn_val.items()}
    us_n_mean = {x: np.array(y).mean() for x, y in us_n_val.items()}
    us_nn_mean = {x: np.array(y).mean() for x, y in us_nn_val.items()}

    s_n_std = {x: np.array(y).std() for x, y in s_n_val.items()}
    s_nn_std = {x: np.array(y).std() for x, y in s_nn_val.items()}
    us_n_std = {x: np.array(y).std() for x, y in us_n_val.items()}
    us_nn_std = {x: np.array(y).std() for x, y in us_nn_val.items()}

    means_dicts = (s_n_mean, s_nn_mean, us_n_mean, us_nn_mean)
    stds_dicts = (s_n_std, s_nn_std, us_n_std, us_nn_std)

    return means_dicts, stds_dicts

def append_name(name, group_num, n_topics, score_agg_df, mean_std_dicts):
    '''
    Method for get_vector_dicts. For each topic_idx it finds the topic's
    descriptive stats and uses those to convert the count value to its
    standarized form.
    '''
    vector = {x: 0 for x in range(n_topics)}
    topic_idx = score_agg_df.loc[name].index.tolist()
    count = score_agg_df.loc[name]['count'].values.tolist()
    for idx, cnt in zip(topic_idx, count):
        mean = mean_std_dicts[0][group_num][idx]
        std = mean_std_dicts[1][group_num][idx]
        z = z_score(cnt, mean, std)
        vector[idx] = z
    return vector

def get_vector_dicts(master_df, score_agg_df, mean_std_dicts,
                     names_lists, n_topics):
    '''
    By group, gets the stanrdized topic vector for each user and stores it in
    a dictionary. Each key is a name and each name has ANOTHER dictionary
    from append_name function that has the key of that nested dictionary
    being a list of topic indicies and the values are that stanrdized
    topic counts
    '''
    s_n_vector, s_nn_vector, us_n_vector, us_nn_vector = {}, {}, {}, {}

    for names_list in names_lists:
        for name in names_list:
            if names_list == names_lists[0]:
                s_n_vector[name] = append_name(name, 0, n_topics,
                                                 score_agg_df, mean_std_dicts)
            elif names_list == names_lists[1]:
                s_nn_vector[name] = append_name(name, 1, n_topics,
                                                 score_agg_df, mean_std_dicts)
            elif names_list == names_lists[2]:
                us_n_vector[name] = append_name(name, 2, n_topics,
                                                 score_agg_df, mean_std_dicts)
            elif names_list == names_lists[3]:
                us_nn_vector[name] = append_name(name, 3, n_topics,
                                                 score_agg_df, mean_std_dicts)
            else:
                print('vector error')

    return s_n_vector, s_nn_vector, us_n_vector, us_nn_vector

def get_started(n_topic):
    '''
    This function was designed with the intention of looping through the values
    in n_topics_list. Uses a lot of functions from topic_analysis.py along with
    get_vector_dicts to get this script started.
    '''
    master_df_nmf, master_df_lda = get_master_df(n_topics)
    score_agg_df_nmf = get_grouby_by_df(master_df_nmf)
    score_agg_df_lda = get_grouby_by_df(master_df_lda)

    count_dict_nmf = get_dict(master_df_nmf, score_agg_df_nmf, 'count')
    count_dict_lda = get_dict(master_df_lda, score_agg_df_lda, 'count')
    count_val_dicts_nmf = get_val_dicts(count_dict_nmf, names_lists,
                                        n_topics, standardize=False)
    count_val_dicts_lda = get_val_dicts(count_dict_lda, names_lists,
                                        n_topics, standardize=False)

    count_mean_std_dicts_nmf = get_mean_std_dicts(count_val_dicts_nmf)
    count_mean_std_dicts_lda = get_mean_std_dicts(count_val_dicts_lda)

    vector_dicts_nmf = get_vector_dicts(master_df_nmf, score_agg_df_nmf,
                                        count_mean_std_dicts_nmf,
                                        names_lists, n_topics)
    vector_dicts_lda = get_vector_dicts(master_df_lda, score_agg_df_lda,
                                        count_mean_std_dicts_lda,
                                        names_lists, n_topics)
    return vector_dicts_nmf, vector_dicts_lda

def get_train_dfs(vector_dicts, n_topics):
    '''
    Returns two pandas dataframes, one for nmf and the other lda. Each df
    has the supervised nuts' and non-nuts' standardized count vectors by
    name along with the label 'is_nut' which is a 1 or 0 depending on nut
    status.
    '''
    vector_dicts_nmf, vector_dicts_lda = vector_dicts

    names_n = list(vector_dicts_nmf[0].keys())
    names_nn = list(vector_dicts_nmf[1].keys())
    X_n_nmf = np.zeros((len(names_n), n_topics))
    X_nn_nmf = np.zeros((len(names_nn), n_topics))
    X_n_lda = np.zeros((len(names_n), n_topics))
    X_nn_lda = np.zeros((len(names_nn), n_topics))

    # gets each X values, not labled as 1 or 0 yet
    for i in range(len(names_n)):
        name_dict = vector_dicts_nmf[0][names_n[i]]
        topic_idx = sorted(list(name_dict.keys()))
        cnt = np.array([name_dict[key] for key in topic_idx])
        for j in range(cnt.shape[0]):
            X_n_nmf[i][j] = cnt[j]

    for i in range(len(names_nn)):
        name_dict = vector_dicts_nmf[1][names_nn[i]]
        topic_idx = sorted(list(name_dict.keys()))
        cnt = np.array([name_dict[key] for key in topic_idx])
        for j in range(cnt.shape[0]):
            X_nn_nmf[i][j] = cnt[j]

    for i in range(len(names_n)):
        name_dict = vector_dicts_lda[0][names_n[i]]
        topic_idx = sorted(list(name_dict.keys()))
        cnt = np.array([name_dict[key] for key in topic_idx])
        for j in range(cnt.shape[0]):
            X_n_lda[i][j] = cnt[j]

    for i in range(len(names_nn)):
        name_dict = vector_dicts_lda[1][names_nn[i]]
        topic_idx = sorted(list(name_dict.keys()))
        cnt = np.array([name_dict[key] for key in topic_idx])
        for j in range(cnt.shape[0]):
            X_nn_lda[i][j] = cnt[j]

    # labeling users and combining dataframes
    df_X_n_nmf = pd.DataFrame(X_n_nmf, index=names_n)
    df_X_nn_nmf = pd.DataFrame(X_nn_nmf, index=names_nn)
    df_X_n_lda = pd.DataFrame(X_n_lda, index=names_n)
    df_X_nn_lda = pd.DataFrame(X_nn_lda, index=names_nn)

    df_X_n_nmf['is_nut'] = np.ones((len(names_n), 1))
    df_X_nn_nmf['is_nut'] = np.zeros((len(names_n), 1))
    df_X_n_lda['is_nut'] = np.ones((len(names_n), 1))
    df_X_nn_lda['is_nut'] = np.zeros((len(names_n), 1))

    df_nmf = df_X_n_nmf.append(df_X_nn_nmf)
    df_lda = df_X_n_lda.append(df_X_nn_lda)

    return df_nmf, df_lda

def model_predictions(X_nmf, y_nmf, X_lda, y_lda):
    abc_nmf = AdaBoostClassifier()
    gbc_nmf = GradientBoostingClassifier()
    rfc_nmf = RandomForestClassifier()
    abc_lda = AdaBoostClassifier()
    gbc_lda = GradientBoostingClassifier()
    rfc_lda = RandomForestClassifier()

    models_nmf = [abc_nmf, gbc_nmf, rfc_nmf]
    models_lda = [abc_lda, gbc_lda, rfc_lda]

    for model in models_nmf:
        model.fit(X_nmf, y_nmf)
    for model in models_lda:
        model.fit(X_lda, y_nmf)

    for model in models_nmf:
        print(model.score(X_nmf, y_nmf))
    for model in models_lda:
        print(model.score(X_lda, y_lda))

if __name__ == '__main__':
    start = time()

    # n_topics_list = [25, 50, 75, 100, 125, 150]
    n_topics_list = [25]
    names_lists = get_name_lists()

    for n_topics in n_topics_list:
        vectors_dicts = get_started(n_topics)
        df_nmf, df_lda = get_train_dfs(vectors_dicts, n_topics)
        y_nmf, y_lda = df_nmf.pop('is_nut').values, df_lda.pop('is_nut').values
        X_nmf, X_lda = df_nmf.values, df_lda.values

        model_predictions(X_nmf, y_nmf, X_lda, y_lda)


    print_time('Done')
