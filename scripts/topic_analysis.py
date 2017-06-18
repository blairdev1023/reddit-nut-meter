import pandas as pd
import numpy as np
from os import listdir
from time import time
import matplotlib.pyplot as plt
from math import log
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

def get_grouby_by_df(master_df):
    '''
    takes the master_df and groups on name first and then topic #. After that,
    the groupby is aggregated into a new df on count and sum.
    '''
    gb = master_df.groupby(['name', 'topic_idx'])
    score_agg_df = gb['score'].aggregate(['count', 'sum'])
    return score_agg_df

def get_dict(master_df, score_agg_df, col):
    '''
    returns a dictionary that has has all the names as keys as each value
    is a tuple of 2 entries. First being the topic #'s in a list and the second
    is another list of those topic #'s col for that user
    '''
    d = {}
    names_set = set(master_df['name'].values.tolist())
    for name in names_set:
        name_df = score_agg_df.loc[name]
        value = (name_df.index.tolist(), name_df[col].values.tolist())
        d[name] = value
    return d

def get_name_lists():
    '''
    returns 4 lists. Each one has the names of belonging to each group
    '''
    s_n = listdir('../data/sup/nuts')
    s_nn = listdir('../data/sup/not_nuts')
    us_n = listdir('../data/un_sup/nuts')
    us_nn = listdir('../data/un_sup/not_nuts')

    # removes non .csv files
    s_n.remove('users.txt')
    s_n.remove('user_info.txt')
    s_nn.remove('users.txt')

    # removes .csv extension
    s_n = list(map(lambda x: x[:-4], s_n))
    s_nn = list(map(lambda x: x[:-4], s_nn))
    us_n = list(map(lambda x: x[:-4], us_n))
    us_nn = list(map(lambda x: x[:-4], us_nn))

    return s_n, s_nn, us_n, us_nn

def get_val_dicts(type_dict, names_list, n_topics, standardize=False):
    '''
    Returns 4 dictionaries. Each dictionary represents one of the 4 groups and
    has its keys set to the topic #'s and each value is a list of the values
    stored in the type_dict.
    '''
    # make dicts for values tally per topic
    s_n, s_nn, us_n, us_nn = names_list

    s_n_val = {x:[] for x in range(n_topics)}
    s_nn_val = {x:[] for x in range(n_topics)}
    us_n_val = {x:[] for x in range(n_topics)}
    us_nn_val = {x:[] for x in range(n_topics)}

    # append each groups' user's values by topic
    for name in s_n:
        idx = type_dict[name][0]
        val = type_dict[name][1]
        for j in range(len(idx)):
            s_n_val[idx[j]].append(val[j])

    for name in s_nn:
        idx = type_dict[name][0]
        val = type_dict[name][1]
        for j in range(len(idx)):
            s_nn_val[idx[j]].append(val[j])

    for name in us_n:
        idx = type_dict[name][0]
        val = type_dict[name][1]
        for j in range(len(idx)):
            us_n_val[idx[j]].append(val[j])

    for name in us_nn:
        idx = type_dict[name][0]
        val = type_dict[name][1]
        for j in range(len(idx)):
            us_nn_val[idx[j]].append(val[j])

    if standardize == False:
        return s_n_val, s_nn_val, us_n_val, us_nn_val
    #### This is the end unless standardize=True ####

    # standardize by topic, regardless of group
    master_dict = s_n_val.copy()
    for d in [s_nn_val, us_n_val, us_nn_val]:
        for key in master_dict.keys():
            master_dict[key] = master_dict[key] + d[key]

    for key in sorted(master_dict.keys()):
        master_dict[key] = preprocessing.scale(master_dict[key])

    # re-parse out standardized values using indicies of varying size
    s_n_lens = [len(s_n_val[key]) for key in sorted(s_n_val.keys())]
    s_nn_lens = [len(s_nn_val[key]) for key in sorted(s_nn_val.keys())]
    us_n_lens = [len(us_n_val[key]) for key in sorted(us_n_val.keys())]

    s_nn_lens = [s_nn_lens[i] + s_n_lens[i] for i in range(len(s_n_val))]
    us_n_lens = [us_n_lens[i] + s_nn_lens[i] for i in range(len(s_n_val))]

    for key in s_n_val.keys():
        s_n_val[key] = master_dict[key][:s_n_lens[key]]
    for key in s_nn_val.keys():
        s_nn_val[key] = master_dict[key][s_n_lens[key]:s_nn_lens[key]]
    for key in s_nn_val.keys():
        us_n_val[key] = master_dict[key][s_nn_lens[key]:us_n_lens[key]]
    for key in s_nn_val.keys():
        us_nn_val[key] = master_dict[key][us_n_lens[key]:]

    return s_n_val, s_nn_val, us_n_val, us_nn_val

def get_mean_dicts(type_val_dicts):
    '''
    Returns 3 tuples, each containing 4 dictionaries. Each dictionary
    represents one of the 4 groups and has the key set to a topic # and each
    value to be the type mean/std/count depending on which tuple is examined
    '''
    s_n_val, s_nn_val, us_n_val, us_nn_val = type_val_dicts

    s_n_mean = {x: np.array(y).mean() for x, y in s_n_val.items()}
    s_nn_mean = {x: np.array(y).mean() for x, y in s_nn_val.items()}
    us_n_mean = {x: np.array(y).mean() for x, y in us_n_val.items()}
    us_nn_mean = {x: np.array(y).mean() for x, y in us_nn_val.items()}

    return s_n_mean, s_nn_mean, us_n_mean, us_nn_mean

def get_param_dicts():
    model_spread = get_model_spread(count_mean_dicts_nmf, count_mean_dicts_lda)
    nmf_sum, lda_sum = model_spread

    nmf_param = {'color': 'k',
                 'label': 'NMF, Spread per Topic: %s' % nmf_sum}
    lda_param = {'color': 'white', 'hatch': '//', 'edgecolor': 'k',
                 'label': 'LDA, Spread per Topic: %s' % lda_sum}

    return nmf_param, lda_param

def get_cos_sim(mean_std_sum_dicts, val_dicts, score_agg_df):
    '''
    Iterates through the Nuts and Non-Nuts indivudal user topic vectors and
    calculates the cos_sim W.R.T. both the Nut average topic vector and the
    Non_Nut Average.

    *** Note, vals_dicts needs to be unstandardized, use the the 'standardize'
    argument in get_val_dicts
    '''
    nut_topic_d = mean_std_sum_dicts[0][0]
    non_nut_topic_d = mean_std_sum_dicts[0][1]
    nut_keys = sorted(nut_topic_d.keys())
    non_nut_keys = sorted(non_nut_topic_d.keys())
    nut_topic_v = np.array([nut_topic_d[key] for key in nut_keys])
    non_nut_topic_v = np.array([non_nut_topic_d[key] for key in non_nut_keys])

    s_n_val, s_nn_val, us_n_val, us_nn_val = val_dicts

    all_val_dicts, mean_all, std_all = s_n_val.copy(), {}, {}
    for d in [s_nn_val, us_n_val, us_nn_val]:
        for key in all_val_dicts.keys():
            all_val_dicts[key] = all_val_dicts[key] + d[key]

    for key in sorted(all_val_dicts.keys()):
        std_all[key] = np.array(all_val_dicts[key]).std()
        mean_all[key] = np.array(all_val_dicts[key]).mean()


    return mean_all, std_all

def get_model_spread(nmf_dicts, lda_dicts):
    s_n_nmf, s_nn_nmf, us_n_nmf, us_nn_nmf = nmf_dicts
    s_n_lda, s_nn_lda, us_n_lda, us_nn_lda = lda_dicts

    plot_dict_nmf = {key: s_n_nmf[key]-s_nn_nmf[key] for key in s_n_nmf.keys()}
    plot_dict_lda = {key: s_n_lda[key]-s_nn_lda[key] for key in s_n_lda.keys()}

    nmf_vals = list(map(lambda x: abs(x), list(plot_dict_nmf.values())))
    lda_vals = list(map(lambda x: abs(x), list(plot_dict_lda.values())))
    nmf_sum = round(np.array(nmf_vals).mean(), 2)
    lda_sum = round(np.array(lda_vals).mean(), 2)

    return nmf_sum, lda_sum

def plot_dicts(mean_dicts, params_dict, place=0):
    x, y = [], []
    s_n_mean, s_nn_mean, us_n_mean, us_nn_mean = mean_dicts
    plot_dict = {key: s_n_mean[key]-s_nn_mean[key] for key in s_n_mean.keys()}
    for item in sorted(list(plot_dict.items())):
        x.append(item[0])
        y.append(item[1])

    x = np.array(x)
    width = 0.4
    ax.bar(x + (width * place), y, width=width, **params_dict)

    ax.set_title('NMF vs. LDA Nut Avg. Count Difference')
    ax.set_xlabel('Topic #')
    ax.set_ylabel('Count Difference (standardized)')
    ax.set_xticks(x)

def end_fig(save_dir):
    plt.legend()
    plt.savefig(save_dir)
    plt.close()

if __name__ == '__main__':
    start = time()

    names_list = get_name_lists()
    dir_nmf = 'pickles/masters/master_df__6_15__nmf__25.pkl'
    dir_lda = 'pickles/masters/master_df__6_15__lda__25.pkl'
    master_df_nmf = pd.read_pickle(dir_nmf)
    master_df_lda = pd.read_pickle(dir_lda)
    n_topics = master_df_lda['topic_idx'].nunique()

    score_agg_df_nmf = get_grouby_by_df(master_df_nmf)
    score_agg_df_lda = get_grouby_by_df(master_df_lda)

    count_dict_nmf = get_dict(master_df_nmf, score_agg_df_nmf, 'count')
    count_dict_lda = get_dict(master_df_lda, score_agg_df_lda, 'count')

    count_val_dicts_nmf = get_val_dicts(count_dict_nmf, names_list,
                                        n_topics, standardize=True)
    count_val_dicts_lda = get_val_dicts(count_dict_lda, names_list,
                                        n_topics, standardize=True)

    count_mean_dicts_nmf = get_mean_dicts(count_val_dicts_nmf)
    count_mean_dicts_lda = get_mean_dicts(count_val_dicts_lda)

    nmf_param, lda_param = get_param_dicts()

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)
    plot_dicts(count_mean_dicts_nmf, nmf_param)
    plot_dicts(count_mean_dicts_lda, lda_param, place=1)
    end_fig('../images/eda__6_15__diff__25.png')

    end = time()
    print('This took %s minutes' % round((end - start)/60., 2))
