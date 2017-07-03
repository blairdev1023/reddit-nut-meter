from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from os import sys

def get_wordclouds(body_arr, saveas=None):
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
    plt.savefig('../images/wordclouds/temp/%s_topic.png' % saveas)
    plt.close()

def get_stop_words():
    stop = stopwords.words('english') + ['will', 'would', 'one', 'get',
                                         'like', 'know', 'still', 'got']
    return stop

if __name__ == '__main__':
    stop = get_stop_words()
    master_df = pd.read_pickle('pickles/lda/50_topics/master_df_train.pkl')

    for i in range(50):
        body_arr = master_df[master_df['topic_idx'] == i]['body'].values
        print('working on topic %s' % i)
        get_wordclouds(body_arr, i)
