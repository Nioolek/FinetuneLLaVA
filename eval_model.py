import csv
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore', category=pd.errors.SettingWithCopyWarning)

csv_path = 'baseline2_train2.csv'
# pandas读取csv
# logo, img, logo是否在训练集，brand是否在训练集，是否是amazon验证集, result
df = pd.read_csv(csv_path)

df['brand'] = df['logo'].apply(lambda x: x.split('/')[0])

# 'logo', 'img_path','logo in train', 'brand in train', 'is amazom', 'result'

all_res = []

def replace_blank(a):
    a = a.rstrip('.').replace(' ', '').replace('_', '').lower()
    return a

def predict_process(x):
    if 'There are None.' in x:
        return 'none'
    else:
        x = sorted(replace_blank(x).split(','))
        return x

def get_logo_recall(logo_df, desc):
    # 是None或者list
    logo_df['predict'] = logo_df['result'].apply(lambda x: predict_process(x))
    # 是str
    logo_df['label'] = logo_df['logo'].apply(lambda x: replace_blank(x).split('/')[0].lower().replace(' ', '').replace('_', ''))

    logo_df['recall'] = logo_df.apply(lambda row: row['label'] in row['predict'] if row['predict'] else False, axis=1)

    class_counts = logo_df['logo'].value_counts()  # 每个类别的样本数
    recall_counts = logo_df[logo_df['recall']].groupby('logo').size()
    recall_rates = recall_counts / class_counts
    recall_rates = recall_rates.fillna(0)
    recall_rates_percent = recall_rates * 100
    print('\n\n')
    print('Recall of %s Logo:' % desc)
    print(recall_rates_percent)

    all_res.append(recall_rates_percent.mean())

def get_brand_recall(logo_df, desc):
    # 是None或者list
    logo_df['predict'] = logo_df['result'].apply(lambda x: predict_process(x))
    # 是str
    logo_df['label'] = logo_df['logo'].apply(
        lambda x: replace_blank(x).split('/')[0].lower().replace(' ', '').replace('_', ''))

    logo_df['recall'] = logo_df.apply(lambda row: row['label'] in row['predict'] if row['predict'] else False, axis=1)

    class_counts = logo_df['brand'].value_counts()  # 每个类别的样本数
    recall_counts = logo_df[logo_df['recall']].groupby('brand').size()
    recall_rates = recall_counts / class_counts
    recall_rates = recall_rates.fillna(0)
    recall_rates_percent = recall_rates * 100
    print('\n\n')
    print('Recall of %s Brand:' % desc)
    print(recall_rates_percent)

    all_res.append(recall_rates_percent.mean())

def get_amazon(logo_df):
    # predict出了品牌就不为none，否则为none
    def temp(row):
        if (row['predict'] == 'none') and (row['label'] == False):
            return True
        if (row['predict'] != 'none') and (row['label'] == True):
            return True
        return False
    logo_df['predict'] = logo_df['result'].apply(lambda x: predict_process(x))
    logo_df['label'] = logo_df['logo'].apply(lambda x: True if x == 'none_brand' else False)
    logo_df['recall'] = logo_df.apply(lambda row: temp(row), axis=1)
    print(logo_df.head())
    class_counts = logo_df['logo'].value_counts()  # 每个类别的样本数
    recall_counts = logo_df[logo_df['recall']].groupby('logo').size()
    recall_rates = recall_counts / class_counts
    recall_rates = recall_rates.fillna(0)
    recall_rates_percent = recall_rates * 100
    print('\n\n')
    print('Recall of Amazon Brand:')
    print(recall_rates_percent)


get_logo_recall(df[df['logo in train'] == True], 'Train')
get_brand_recall(df[df['brand in train'] == True], 'Train')
get_logo_recall(df[(df['logo in train'] == False) & (df['is amazom'] == False) & (df['brand in train'] == False)], 'Test')
get_amazon(df[df['is amazom'] == True])

print('ALL TRAIN LOGO RECALL: ', all_res[0])
print('ALL TRAIN BRAND RECALL: ', all_res[1])
print('ALL TEST LOGO RECALL: ', all_res[2])


