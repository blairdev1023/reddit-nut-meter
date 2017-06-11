import pandas as pd
import numpy as np
from os import listdir
from time import time
import matplotlib.pyplot as plt
from math import log

def log_tai(tai):
    '''
    Returns a logged output of the TAI value. If the TAI value is negative the
    magnitude is logged, negated, and returned. If the TAI value is zero, then
    zero is returned.
    '''
    if tai < 0:
        return log(-tai)
    elif tai == 0:
        return 0
    else:
        return log(tai)

def get_grouby_by_df(master_df):
    '''
    takes the master_df and groups on name first and then topic #. After that,
    the groupby is aggregated into a new df on count and sum. Lastly that aggregate df creates a new column 'tai' that is the count multiplied by the sum
    '''
    gb = master_df.groupby(['name', 'topic_idx'])
    score_agg_df = gb['score'].aggregate(['count', 'sum'])
    score_agg_df['tai'] = score_agg_df['count'] * score_agg_df['sum']
    score_agg_df['tai'] = score_agg_df['tai'].apply(lambda x: log_tai(x))
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

def get_tai_tot_dicts(tai_dict, names_list):
    '''
    Returns 4 dictionaries. Each one represents one of the 4 groups and has
    the key set to a topic # and each value to be the total tai that group
    has in that topic #
    '''
    s_n, s_nn, us_n, us_nn = names_list

    s_n_tai_tot = [0 for x in range(20)]
    s_nn_tai_tot = [0 for x in range(20)]
    us_n_tai_tot = [0 for x in range(20)]
    us_nn_tai_tot = [0 for x in range(20)]

    for name in s_n:
        idx = tai_dict[name][0]
        tai = tai_dict[name][1]
        for j in range(len(idx)):
            s_n_tai_tot[idx[j]] += tai[j]

    for name in s_nn:
        idx = tai_dict[name][0]
        tai = tai_dict[name][1]
        for j in range(len(idx)):
            s_nn_tai_tot[idx[j]] += tai[j]

    for name in us_n:
        idx = tai_dict[name][0]
        tai = tai_dict[name][1]
        for j in range(len(idx)):
            us_n_tai_tot[idx[j]] += tai[j]

    for name in us_nn:
        idx = tai_dict[name][0]
        tai = tai_dict[name][1]
        for j in range(len(idx)):
            us_nn_tai_tot[idx[j]] += tai[j]

    # normalize
    s_n_tai_tot = [float(x) / sum(s_n_tai_tot) for x in s_n_tai_tot]
    s_nn_tai_tot = [float(x) / sum(s_nn_tai_tot) for x in s_nn_tai_tot]
    us_n_tai_tot = [float(x) / sum(us_n_tai_tot) for x in us_n_tai_tot]
    us_nn_tai_tot = [float(x) / sum(us_nn_tai_tot) for x in us_nn_tai_tot]

    return s_n_tai_tot, s_nn_tai_tot, us_n_tai_tot, us_nn_tai_tot

def get_val_dicts(type_dict, names_list):
    '''
    Returns 4 dictionaries. Each dictionary represents one of the 4 groups and
    has its keys set to topic #'s' and each value is a list of the values
    stored in the type_dict
    '''
    s_n, s_nn, us_n, us_nn = names_list

    s_n_val = {x:[] for x in range(20)}
    s_nn_val = {x:[] for x in range(20)}
    us_n_val = {x:[] for x in range(20)}
    us_nn_val = {x:[] for x in range(20)}

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

    return s_n_val, s_nn_val, us_n_val, us_nn_val

def get_mean_std_sum_dicts(type_val_dicts):
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

    s_n_std = {x: np.array(y).std() for x, y in s_n_val.items()}
    s_nn_std = {x: np.array(y).std() for x, y in s_nn_val.items()}
    us_n_std = {x: np.array(y).std() for x, y in us_n_val.items()}
    us_nn_std = {x: np.array(y).std() for x, y in us_nn_val.items()}

    s_n_sum = {x: np.array(y).sum() for x, y in s_n_val.items()}
    s_nn_sum = {x: np.array(y).sum() for x, y in s_nn_val.items()}
    us_n_sum = {x: np.array(y).sum() for x, y in us_n_val.items()}
    us_nn_sum = {x: np.array(y).sum() for x, y in us_nn_val.items()}

    mean_dicts = (s_n_mean, s_nn_mean, us_n_mean,us_nn_mean)
    std_dicts = (s_n_std, s_nn_std, us_n_std, us_nn_std)
    sum_dicts = (s_n_sum, s_nn_sum, us_n_sum, us_nn_sum)

    return mean_dicts, std_dicts, sum_dicts

def get_params_dicts():
    s_n_param = {'color': 'r', 'label':'Nuts (S)'}
    s_nn_param = {'color': 'b', 'label':'Not Nuts (S)'}
    us_n_param = {'color': 'r', 'label':'Nuts (U)', 'ls':'--'}
    us_nn_param = {'color': 'b', 'label':'Not Nuts (U)', 'ls':'--'}
    params_dicts = (s_n_param, s_nn_param, us_n_param, us_nn_param)
    return params_dicts

def plot_dicts(mean_std_sum_dicts, params_dicts):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    mean_dicts, std_dicts, sum_dicts = mean_std_sum_dicts
    mean_dicts = mean_dicts[:2]
    params_dicts = params_dicts[:2]
    for d in mean_dicts:
        # del d[0]
        # del d[4]
        del d[13]

    zipped_dicts = zip(mean_dicts, params_dicts)

    for mean_dict, param_dict in zipped_dicts:
        x, y = [], []
        for key, value in mean_dict.items():
            x.append(key)
            y.append(value)
        ax.plot(x, y, **param_dict)

    ax.set_title('Nut vs. Non-Nut Avg. T.A.I. per Topic with LDA')
    ax.set_xlabel('Topic #')
    ax.set_ylabel('T.A.I. Value')
    plt.legend()

    plt.savefig('../images/eda_log_tai.png')
    plt.close()

if __name__ == '__main__':
    start = time()

    names_list = get_name_lists()
    master_df = pd.read_pickle('pickles/master_df_lda.pkl')
    score_agg_df = get_grouby_by_df(master_df)

    tai_dict = get_dict(master_df, score_agg_df, 'tai')
    # count_dict = get_dict(master_df, score_agg_df, 'count')
    tai_val_dicts = get_val_dicts(tai_dict, names_list)
    # count_val_dicts = get_val_dicts(count_dict, names_list)
    tai_mean_std_sum_dicts = get_mean_std_sum_dicts(tai_val_dicts)
    # count_mean_std_sum_dicts = get_mean_std_sum_dicts(count_val_dicts)
    params_dicts = get_params_dicts()

    plot_dicts(tai_mean_std_sum_dicts, params_dicts)
    # plot_dicts(count_mean_std_sum_dicts, params_dicts)

    end = time()
    print('This took %s minutes' % round((end - start)/60., 2))
