import csv
import numpy as np
import pandas as pd

import warnings

from logo_dict import brand_link, none_brand_label

warnings.filterwarnings(action='ignore', category=pd.errors.SettingWithCopyWarning)

# csv_path = 'result/baseline3_e3.csv'
# csv_path = 'result/baseline3.csv'
# csv_path = 'result/baseline1.csv'
# csv_path = 'result/baseline2.csv'
# csv_path = 'result/baseline2_train2.csv'

# csv_path = 'result/baseline3_lorar.csv'
# csv_path = 'result/baseline3_lorar1.csv'
# csv_path = 'result/baseline3_lorar2.csv'
# csv_path = 'result/baseline5.csv'
# csv_path = 'result/baseline6.csv'
# csv_path = 'result/baseline6_lorar.csv'
# csv_path = 'result/baseline6_lorar1.csv'
# csv_path = 'result/baseline6_lorar1_nodesc.csv'
# csv_path = 'result/baseline6_lorar2.csv'
# csv_path = 'result/baseline6_lorar.csv'
# csv_path = 'result/baseline6_lorar_nodesc.csv'
# csv_path = 'result/baseline7.csv'
# csv_path = 'result/baseline6_e3.csv'

csv_path = 'result/baseline6_lorar_nodesc.csv'
save_csv_path = 'result/c_baseline6_lorar_nodesc.csv'


# pandas读取csv
# logo, img, logo是否在训练集，brand是否在训练集，是否是amazon验证集, result
df = pd.read_csv(csv_path)

df['brand'] = df['logo'].apply(lambda x: x.split('/')[0])

# 'logo', 'img_path','logo in train', 'brand in train', 'is amazom', 'result'

all_res = []

def replace_blank(a):
    a = a.rstrip('.').replace(' ', '').replace('_', '').replace("'", "").lower()
    return a

def predict_process(x):
    if 'There are None.' in x:
        return 'none'
    else:
        x = sorted(replace_blank(x).split(','))
        return x


def judge(label, predict):
    if label in brand_link:
        brand_list = brand_link[label]
        # 将label与predict求交集
        if set(brand_list) & set(predict):
            return True
        else:
            return False
    elif label in predict:
        return True
    else:
        return False


def get_logo_recall(logo_df, desc):
    # 是None或者list
    logo_df['predict'] = logo_df['result'].apply(lambda x: predict_process(x))
    # 是str
    logo_df['label'] = logo_df['logo'].apply(lambda x: replace_blank(x).split('/')[0].lower().replace(' ', '').replace('_', ''))

    logo_df['recall'] = logo_df.apply(lambda row: judge(row['label'], row['predict']), axis=1)

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

    logo_df['recall'] = logo_df.apply(lambda row: judge(row['label'], row['predict']), axis=1)

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
    # print('\n\n')
    # print('Recall of Amazon Brand:')
    # print(recall_rates_percent)
    all_res.append(recall_rates_percent['none'])
    all_res.append(recall_rates_percent['none_brand'])

def get_amazon_label(logo_df):
    # 基于人工打标的数据计算recall
    def temp(row):
        predict = set(row['predict'])    # list
        label = set(row['label'])        # list
        if predict & label:
            return True
        else:
            return False
    # print(none_brand_label)
    logo_df['predict'] = logo_df['result'].apply(lambda x: predict_process(x))
    logo_df['label'] = logo_df['img_path'].apply(lambda x: none_brand_label[x])
    logo_df['recall'] = logo_df.apply(lambda row: temp(row), axis=1)
    logo_df.to_csv(save_csv_path, index=False)
    print(logo_df.head())
    class_counts = logo_df['logo'].value_counts()  # 每个类别的样本数
    recall_counts = logo_df[logo_df['recall']].groupby('logo').size()
    recall_rates = recall_counts / class_counts
    recall_rates = recall_rates.fillna(0)
    recall_rates_percent = recall_rates * 100
    print('\n\n')
    # print('Recall of Amazon Brand Label:')
    # print(recall_rates_percent)
    # print('!!!!', type(recall_rates_percent))
    all_res.append(recall_rates_percent.mean())


get_logo_recall(df[df['logo in train'] == True], 'Train')
get_brand_recall(df[df['brand in train'] == True], 'Train')
get_logo_recall(df[(df['logo in train'] == False) & (df['is amazom'] == False) & (df['brand in train'] == False)], 'Test')
get_amazon(df[df['is amazom'] == True])
get_amazon_label(df[(df['is amazom'] == True) & (df['logo'] == 'none_brand')])

print('ALL TRAIN LOGO RECALL: ', all_res[0])
print('ALL TRAIN BRAND RECALL: ', all_res[1])
print('ALL TEST LOGO RECALL: ', all_res[2])
print('NONE RECALL: ', all_res[3])
print('NONE BRAND RECALL: ', all_res[4])
print('NONE BRAND LABEL RECALL: ', all_res[5])
print(csv_path)


