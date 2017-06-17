from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

def get_wordclouds(body_arr, i=None, saveas=None):
    body_list = [str(x) for x in body_arr.tolist()]
    text = ' '.join(body_list)
    wordcloud = WordCloud(width=700,
                          height=500,
                          stopwords=stop,
                          )
    wordcloud = wordcloud.generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if i != None:
        plt.savefig('../images/lda_round1/topic%s.png' % i)
        plt.close()
    elif type(saveas) == str:
        plt.savefig('../images/lda_round1/%s.png' % saveas)
        plt.close()
    else:
        plt.savefig('../images/lda_round1/temp.png' % saveas)
        plt.close()

def get_stop_words():
    stop = stopwords.words('english') + ['will', 'would', 'one', 'get',
                                         'like', 'know', 'still', 'got']
    return stop

if __name__ == '__main__':
    stop = get_stop_words()
<<<<<<< HEAD
    master_df = pd.read_pickle('pickles/masters/master_df__6_15__nmf__100.pkl')

    body_arr = master_df[master_df['topic_idx'] == 29]['body'].values
    # print('working on topic %s' % i)
    get_wordclouds(body_arr, 29)
=======
    master_df = pd.read_pickle('pickles/master_df_lda_tune.pkl')

    for i in range(20):
        body_arr = master_df[master_df['topic_idx'] == i]['body'].values
        print('working on topic %s' % i)
        get_wordclouds(body_arr, i)
>>>>>>> 373ba0f8d36ecebf6b0a3f397e64028dc8ddf8df
