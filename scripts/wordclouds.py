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
        plt.savefig('../images/topic%s.png' % i)
        plt.close()
    elif type(saveas) == str:
        plt.savefig('../images/%s.png' % saveas)
        plt.close()
    else:
        plt.savefig('../images/temp.png' % saveas)
        plt.close()

def get_stop_words():
    stop = stopwords.words('english') + ['will', 'would', 'one', 'get',
                                         'like', 'know', 'still', 'got']
    return stop

if __name__ == '__main__':
    stop = get_stop_words()
    master_df = pd.read_pickle('pickles/master_df2.pkl')

    for i in range(25):
        body_arr = master_df[master_df['topic_idx'] == i]['body'].values
        print('working on topic %s' % i)
        get_wordclouds(body_arr, i)
