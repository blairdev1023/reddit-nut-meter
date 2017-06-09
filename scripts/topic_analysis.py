import pandas as pd

def get_cti_dict(master_df, score_agg_df):
    MIS_names = pd.read_pickle('pickles/dfMIS.pkl').values.tolist()
    WN_names = pd.read_pickle('pickles/dfWN.pkl').values.tolist()

    cti_dict = {}
    names_set = set(master_df['name'].values.tolist())
    for name in names_set:
        name_df = score_agg_df.loc[name]
        value = zip(name_df.index.tolist(), name_df['cti'].values.tolist())
        cti_dict[name] = list(value)
    return cti_dict

if __name__ == '__main__':
    score_agg_df = pd.read_pickle('pickles/score_agg_df.pkl')
    master_df = pd.read_pickle('pickles/master_df3.pkl')

    cti_dict = get_cti_dict(master_df, score_agg_df)
