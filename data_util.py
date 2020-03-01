#!/usr/bin/env python

"""
Project: Kaggle 2019 Data Science Bowl (DSB).
https://www.kaggle.com/c/data-science-bowl-2019

Utils related with data processing.

Author: Zhengyang Zhao
"""

import os
from sys import getsizeof
import time
from time import mktime
from datetime import datetime as dt
import re
import functools
from concurrent import futures
from multiprocessing import Pool, cpu_count
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def split_train_by_userID(train_df, train_label_df, target_dir):
    """
    Split the original train data into multiple csv files, each file only contains the 
    """
    train_users = train_df['installation_id'].unique()
    train_users_withlabel = train_label_df['installation_id'].unique()
    train_users_withAssessment = train_df[train_df['type'] == 'Assessment']['installation_id'].unique()
    print('Count of train users:', len(train_users))
    print('Count of train users with label:', len(train_users_withlabel))
    print('Count of train users taking Assessment:', len(train_users_withAssessment))
    
    threads = 4
    t0 = time.time()
    with futures.ThreadPoolExecutor(threads) as p:
        p.map(functools.partial(extract_userID, train_df=train_df, 
                                                train_users_withlabel=train_users_withlabel,
                                                target_dir=target_dir), 
                        [userID for userID in train_users_withAssessment])
    t1 = time.time()
    print('Finish splitting train_df. Time: {0:.1f} seconds.'.format((t1 - t0)))
    

    
def extract_userID(userID, train_df, train_users_withlabel, target_dir):
    """
    Extract one user from the original train data. And save the data of this user to a file.
    """
    if userID not in train_users_withlabel:
        # print("User {} did Assessment, but didn't submit.".format(userID))
        return
    user_df = train_df[train_df['installation_id'] == userID]
    user_df.to_csv(os.path.join(target_dir, 'train_installation_id_{}.csv'.format(userID)), index=False)

    
#################################################################
# Pre-Process data
#################################################################
"""
Three main pre-processing functions:
1. get_label_csv(): extract information of assessment sessions (label) from raw data.
2. get_process_csv(): extract information of all game sessions from raw data.
3. get_Xy_csv(): from the proc_df, build Xy_df.
"""
    
    
def get_label_csv(df, target_path, is_test=True):
    """
    From the original train/test dataframe, extract the Assessment results and save to train_labels or test_labels csv file.
    Note:
        If an Assessment is not submitted (with code 4100 or 4110), then ignore this Assessment.
        If is_test==True, and Assessment section only contains 1 row of code 2000, then create a row for this Assessment.
    """

    ass_df = df[df['type'] == 'Assessment']
    ass_df.set_index('game_session', inplace=True) # use 'game_session' as new index to speed up the searching in folloing for-loop
    ass_df = ass_df.sort_index()
    # Dataframe searching time:
    #    set the target col as index and sort_index << set the target col as index but do not sort_index < search the target col directly.
    sessions = ass_df.index.unique()
    
    list_game_session = []
    list_installation_id = []
    list_title = []
    list_start_time = []
    list_num_correct = [] 
    list_num_incorrect = []
    list_accuracy = []
    list_accuracy_group = []
    
    t0 = time.time()
    for sess in sessions:
        sess_df = ass_df.loc[[sess]]
        sess_df.sort_values(by=['timestamp'], inplace=True) 
        sess_game_session = sess
        sess_title = sess_df['title'].iloc[0]
        sess_start_time = sess_df['timestamp'].iloc[0]
        sess_installation_id = sess_df['installation_id'].iloc[0]
        sess_code_list = sess_df['event_code'].unique()
        if is_test and (sess_df.shape[0] == 1 and sess_df['event_code'].iloc[0] == 2000):
            list_game_session = list_game_session + [sess_game_session]
            list_installation_id = list_installation_id + [sess_installation_id]
            list_title = list_title + [sess_title]
            list_start_time = list_start_time + [sess_start_time]
            list_num_correct = list_num_correct + [np.nan]
            list_num_incorrect = list_num_incorrect + [np.nan]
            list_accuracy = list_accuracy + [np.nan]
            list_accuracy_group = list_accuracy_group + [np.nan]
            continue
        
        code = 4100
        if sess_title == 'Bird Measurer (Assessment)':
            code = 4110
        if code not in sess_code_list:
            continue
        
        temp_df = sess_df[sess_df['event_code'] == code]
        sess_num_correct = temp_df['event_data'].str.contains('true').sum()
        sess_num_incorrect = temp_df.shape[0] - sess_num_correct
        sess_accuracy = sess_num_correct / (sess_num_correct + sess_num_incorrect)
        sess_accuracy_group = 1
        if sess_accuracy == 1:
            sess_accuracy_group = 3
        elif sess_accuracy == 0.5:
            sess_accuracy_group = 2
        elif sess_accuracy == 0:
            sess_accuracy_group = 0
            
        list_game_session = list_game_session + [sess_game_session]
        list_installation_id = list_installation_id + [sess_installation_id]
        list_title = list_title + [sess_title]
        list_start_time = list_start_time + [sess_start_time]
        list_num_correct = list_num_correct + [sess_num_correct]
        list_num_incorrect = list_num_incorrect + [sess_num_incorrect]
        list_accuracy = list_accuracy + [sess_accuracy]
        list_accuracy_group = list_accuracy_group + [sess_accuracy_group]

    label_df = pd.DataFrame({'game_session': list_game_session,
                             'installation_id': list_installation_id, 
                             'title': list_title, 
                             'start_time': list_start_time,
                             'num_correct': list_num_correct, 
                             'num_incorrect': list_num_incorrect, 
                             'accuracy': list_accuracy, 
                             'accuracy_group': list_accuracy_group,
                             })
    label_df.sort_values(by=['installation_id', 'start_time'], inplace=True) 
    label_df.to_csv(target_path, index=False)
    t1 = time.time()
    print('Finished extracting labels. Time: {0:.1f} seconds.'.format((t1 - t0)))

    
def get_process_csv(df, users, target_path, is_test, cpus=cpu_count()//2):
    """
    Each session in train_df would generate one row with several features.
    """
    df = df[ (df['type'] == 'Game') | (df['type'] == 'Assessment')]
    df.set_index('installation_id', inplace=True) # use 'installation_id' as new index to speed up the searching 
    df = df.sort_index()
    
    t0 = time.time()
    with Pool(cpus) as p:
        ll = p.map(functools.partial(process_user, is_test=is_test), \
                     [ df.loc[[uid]] for uid in users if df.loc[[uid]].shape[0] > 0 ])
    proc_df = pd.concat(ll, ignore_index=True)
    proc_df.sort_values(by=['installation_id', 'start_time'], inplace=True) 
    if is_test:
        proc_df_nan = proc_df[(proc_df['type'] == 'Assessment') & (proc_df['f0'].isna()) ]
        proc_df_nan_sessions = proc_df_nan['game_session'].values
        proc_df_final_sessions = proc_df_nan.groupby('installation_id').tail(1)['game_session'].values
        proc_df_junk_sessions = [sess for sess in proc_df_nan_sessions if sess not in proc_df_final_sessions]
        proc_df = proc_df[~proc_df['game_session'].isin(proc_df_junk_sessions)]
        
    proc_df = add_percentile_info(proc_df)
    proc_df.to_csv(target_path, index=False)
    t1 = time.time()
    print('Finished extracting features. Time: {0:.1f} seconds.'.format((t1 - t0)))
    

def process_user(user_df, is_test):
    """
    Return
        df:
            columns=('game_session', 
                     'installation_id', 
                     'title', 
                     'start_time',
                     'start_time_parsed',
                     'type',
                     'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11'
                     )
            dtypes = ['object'] * 5 + ['float32'] * 12
    """
    
    sessions = user_df['game_session'].unique()
    sess_installation_id = user_df.index[0]
    user_df.set_index('game_session', inplace=True) # use 'game_session' as new index to speed up the searching 
    user_df = user_df.sort_index()
    
    list_game_session = []
    list_installation_id = []
    list_title = []
    list_start_time = []
    list_start_time_parsed = []
    list_type = []
    list_f0 = [] 
    list_f1 = []
    list_f2 = []
    list_f3 = []
    list_f4 = [] 
    list_f5 = []
    list_f6 = []
    list_f7 = []
    list_f8 = [] 
    list_f9 = []
    list_f10 = []
    list_f11 = []
    
    for sess in sessions:
        [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11] = [np.nan] * 12
        sess_df = user_df.loc[[sess]]
        sess_df.sort_values(by=['timestamp'], inplace=True) 
        sess_type  = sess_df['type'].iloc[0]
        sess_title = sess_df['title'].iloc[0]
        sess_start_time = sess_df['timestamp'].iloc[0]
        if sess_type == 'Assessment':
            [tot_attempt, duration, accuracy, accuracy_group, keep] = process_sess_Assessment(sess_df, is_test=is_test)
            if keep == False:
                continue
            [f0, f1, f2, f3] = [tot_attempt, duration, accuracy, accuracy_group]
        
        elif sess_title == 'All Star Sorting':
            # [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur]
            res = process_sess_game_City_02(sess_df)  # 10
            if res == [np.nan] * 10:
                continue
            [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = res
        
        elif sess_title == 'Air Show':
            # [tot_attampts, ave_acc, ave_duration]
            res = process_sess_game_City_10(sess_df)  # 3
            if res == [np.nan] * 3:
                continue
            [f0, f1, f2] = res
        
        elif sess_title == 'Crystals Rule':
            # [tot_round, lv1_att, lv1_acc, lv1_dur, lv2_att, lv2_acc, lv2_dur, lv3_att, lv3_acc, lv3_dur]
            res = process_sess_game_City_13(sess_df) # 10
            if res == [np.nan] * 10:
                continue
            [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = res
        
        elif sess_title == 'Scrub-A-Dub':
            # [lv3_att, lv3_acc, lv3_dur, lv6_att, lv6_acc, lv6_dur, lv9_att, lv9_acc, lv9_dur, lv12_att, lv12_acc, lv12_dur]
            res = process_sess_game_Peak_03(sess_df) #12
            if res == [np.nan] * 12:
                continue
            [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11] = res
        
        elif sess_title == 'Dino Drink':
            # [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur]
            res = process_sess_game_Peak_06(sess_df) # 10
            if res == [np.nan] * 10:
                continue
            [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = res
        
        elif sess_title == 'Bubble Bath':
            # [tot_attampts, ave_acc, ave_duration]
            res = process_sess_game_Peak_07(sess_df) # 3
            if res == [np.nan] * 3:
                continue
            [f0, f1, f2] = res
        
        elif sess_title == 'Dino Dive':
            # [tot_round, lv1_att, lv1_acc, lv1_dur, lv2_att, lv2_acc, lv2_dur]
            res = process_sess_game_Peak_09(sess_df) # 7
            if res == [np.nan] * 7:
                continue
            [f0, f1, f2, f3, f4, f5, f6] = res
        
        elif sess_title == 'Chow Time':
            # [tot_attampts, ave_acc, ave_duration]
            res = process_sess_game_Cave_01(sess_df)
            if res == [np.nan] * 3:
                continue
            [f0, f1, f2] = res
        
        elif sess_title == 'Happy Camel':
            # [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur]
            res = process_sess_game_Cave_07(sess_df)  # 10
            if res == [np.nan] * 10:
                continue
            [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = res
            
        elif sess_title == 'Leaf Leader':
            res = process_sess_game_Cave_09(sess_df)  # 10
            if res == [np.nan] * 3:
                continue
            [f0, f1, f2] = res
        
        elif sess_title == 'Pan Balance':
            # [tot_attampts, ave_acc, ave_duration]
            res = process_sess_game_Cave_12(sess_df) # 3
            if res == [np.nan] * 3:
                continue
            [f0, f1, f2] = res
            
        list_game_session += [sess]
        list_installation_id += [sess_installation_id]
        list_title += [sess_title]
        list_start_time = list_start_time + [sess_start_time]
        list_start_time_parsed += [mktime(dt.strptime(str(sess_start_time), \
                                      "%Y-%m-%dT%H:%M:%S.%fZ").timetuple())/3600]  #'2019-08-06T05:22:01.344Z'
        list_type += [sess_type]
        list_f0 += [f0] 
        list_f1 += [f1]
        list_f2 += [f2]
        list_f3 += [f3]
        list_f4 += [f4] 
        list_f5 += [f5]
        list_f6 += [f6]
        list_f7 += [f7]
        list_f8 += [f8] 
        list_f9 += [f9]
        list_f10 += [f10]
        list_f11 += [f11]

    df = pd.DataFrame({'game_session': list_game_session,
                       'installation_id': list_installation_id, 
                       'title': list_title, 
                       'start_time': list_start_time,
                       'start_time_parsed': list_start_time_parsed,
                       'type':list_type,
                       'f0': list_f0,
                       'f1': list_f1,
                       'f2': list_f2,
                       'f3': list_f3,
                       'f4': list_f4,
                       'f5': list_f5,
                       'f6': list_f6,
                       'f7': list_f7,
                       'f8': list_f8,
                       'f9': list_f9,
                       'f10': list_f10,
                       'f11': list_f11,
                      })
    return df


def add_percentile_info(proc_df):
    """
    Add new features onto proc_df:
    'percentile': for a certain session type, how many percent of trials this trial out-performs.
    'percentile_growth': (current_percentile - last_percentile of the same game) / sqrt(current_timestamp - last_timestamp).
    'time_from_last_trial': time from last play of the game (in hour). if no history, set 100000. 
    """
    
    session_percentileOn = {
    #     'All Star Sorting': ['f2', 'f1'],
    #     'Air Show': ['f2', 'f1'],
    #     'Crystals Rule': ['f2', 'f1'],
    #     'Scrub-A-Dub': ['f2', 'f1'],
    #     'Dino Drink': ['f2', 'f1'],
    #     'Bubble Bath': ['f2', 'f1'],
    #     'Dino Dive': ['f2', 'f1'],
    #     'Chow Time': ['f2', 'f1'],
    #     'Happy Camel': ['f2', 'f1'],
    #     'Leaf Leader': ['f2', 'f1'],
    #     'Pan Balance': ['f2', 'f1'],
        'Bird Measurer (Assessment)': ['f2', 'f1'],  # [accuracy, duration]
        'Cart Balancer (Assessment)': ['f2', 'f1'], 
        'Cauldron Filler (Assessment)': ['f2', 'f1'], 
        'Chest Sorter (Assessment)': ['f2', 'f1'], 
        'Mushroom Sorter (Assessment)': ['f2', 'f1'], 
    }

    proc_df_copy = proc_df.set_index('game_session')

    # Add 'percentile'
    for session_title in session_percentileOn:
        score_cols = session_percentileOn[session_title]
        keep_cols = list(proc_df.columns.values[:5]) + score_cols
        sess_df = proc_df[proc_df['title'] == session_title][keep_cols].dropna()
        sess_df['f1'] = -sess_df['f1']
    #     fig, ax = plt.subplots(1, 4, figsize=[25, 4])
    #     for group in range(4):
    #         ax[group].hist(sess_df[sess_df['f3'] == group]['f1'], bins=range(500, 50000, 500))
        sess_df.sort_values(by=score_cols, inplace=True)
        sess_df.reset_index(inplace=True)
        sess_df['percentile'] = sess_df.index / sess_df.shape[0]
        sess_df.set_index('game_session', inplace=True)
        proc_df_copy.loc[sess_df.index, 'f4'] = sess_df['percentile']
    proc_df_copy = proc_df_copy.reset_index().sort_values(by=['installation_id', 'start_time'])
    proc_df_copy = proc_df_copy.reset_index(drop=True)

    # Add 'percentile_growth' and 'time_from_last_trial'
    proc_df_copy.loc[:, 'time_from_last_trial'] = 100000
    for k, df in proc_df_copy.groupby(['installation_id', 'title']):
        if df.shape[0] == 1:
            continue
        for i in range(1, df.shape[0]):
            idx = df.index[i]
            time_diff = df['start_time_parsed'].iloc[i] - df['start_time_parsed'].iloc[i - 1]
            proc_df_copy.loc[idx, 'time_from_last_trial'] = time_diff
            if np.isnan(df['f4'].iloc[i]):
                continue
            perc_diff = df['f4'].iloc[i] - df['f4'].iloc[i - 1]
            perc_growth = perc_diff / math.sqrt(time_diff)
            proc_df_copy.loc[idx, 'f5'] = perc_growth
    
    return proc_df_copy


def get_Xy_csv(proc_df, Xy_csv_path, feature_specs_path):
    """
    From proc_df, generate Xy_df and save to csv.
    
    Note: in Xy_df, each assessment corresponds to a row. 
        Therefore, one user can generate multiple rows, if he took multiple assessments.
        total_rows == number of assessments in proc_df.
    
    Columns of Xy_df:
        installation_id,
        is_last_assessment (boolean),
        is_first_assessment (boolean),
        assessment_title,
        accuracy, 
        accuracy_group.
        [features],
        
        
        Accumulated features:
        - total assessments taken
        - total game sessions taken
        - accumulated accuracy percentile
        - whether has taken current assessment
    """
    
    t0 = time.time()
    feature_count = {'All Star Sorting': 10,
                    'Air Show': 3,
                    'Crystals Rule': 10,
                    'Scrub-A-Dub': 12,
                    'Dino Drink': 10,
                    'Bubble Bath': 3,
                    'Dino Dive': 7,
                    'Chow Time': 3,
                    'Happy Camel': 10,
                    'Leaf Leader': 3,
                    'Pan Balance': 3,
                    'Bird Measurer (Assessment)': 10,  # [count, dur_sum, dur_last, dur_mean, acc_sum, acc_last, acc_mean, group_sum, group_last, group_mean]
                    'Cart Balancer (Assessment)': 10,
                    'Cauldron Filler (Assessment)': 10,
                    'Chest Sorter (Assessment)': 10,
                    'Mushroom Sorter (Assessment)': 10,
                    'Assessment Overall': 8,  # [count, group_sum, group_mean, percentile_sum, percentile_mean, 
                                              #    perc_growth_sum, perc_growth_count, perc_growth_mean]
                    'others': 1, # [time_from_last]
                   }
    
    proc_df_assessment = proc_df[proc_df['type'] == 'Assessment']
    proc_df_assessment.loc[:, 'is_last_assessment'] = 0
    proc_df_assessment.loc[:, 'is_first_assessment'] = 0
    proc_df_assessment.loc[proc_df_assessment.groupby('installation_id').tail(1).index, 'is_last_assessment'] = 1
    proc_df_assessment.loc[proc_df_assessment.groupby('installation_id').head(1).index, 'is_first_assessment'] = 1
    
    Xy_df = pd.DataFrame()
    Xy_df['installation_id'] = proc_df_assessment['installation_id']
    Xy_df['is_last_assessment'] = proc_df_assessment['is_last_assessment']
    Xy_df['is_first_assessment'] = proc_df_assessment['is_first_assessment']
    Xy_df['assessment_title'] = proc_df_assessment['title']
    Xy_df['accuracy'] = proc_df_assessment['f2']
    Xy_df['accuracy_group'] = proc_df_assessment['f3']
    feature_col_start = Xy_df.shape[1]

    # prepare feature columns:
    feature_specs_df = pd.DataFrame(columns=['feature_name', 'feature_type', 'is_sum', 'contains_nan'])
    feature_count_df = pd.DataFrame.from_dict(feature_count, orient='index', columns=['column_count'])
    feature_count_df['column_start'] = feature_col_start
    for i in range(1, feature_count_df.shape[0]):
        feature_count_df['column_start'].iloc[i] = feature_count_df['column_start'].iloc[i - 1]  \
                                                   + feature_count_df['column_count'].iloc[i - 1]
    total_features = feature_count_df['column_start'].iloc[-1] + feature_count_df['column_count'].iloc[-1] - feature_col_start
    assessment_feature_start = feature_count_df.loc['Bird Measurer (Assessment)', 'column_start']
    assessment_feature_count = feature_count_df.loc['Bird Measurer (Assessment)', 'column_count']
    assessment_feature_end   = feature_count_df.loc['Assessment Overall', 'column_start']
    assessall_feature_start = feature_count_df.loc['Assessment Overall', 'column_start']
    assessall_feature_count = feature_count_df.loc['Assessment Overall', 'column_count']
    assessall_feature_end   = assessall_feature_start + assessall_feature_count
    for f in feature_count_df.index:
        for i in range(feature_count_df.loc[f, 'column_count']):
            col_name = '{}_{:02d}'.format(f, i)
            Xy_df.loc[:, col_name] = np.nan
            feature_specs_df.loc[feature_specs_df.shape[0], 'feature_name'] = col_name
    assert(feature_specs_df.shape[0] == total_features)
    assert(Xy_df.shape[1] == feature_col_start + total_features)
    feature_specs_df['is_sum'] = 0
    feature_specs_df['contains_nan'] = 1
    feature_specs_df['feature_type'] = ''
    sum_feature_indexes  = [ x - feature_col_start for x in \
                             list(range(assessment_feature_start+1, assessment_feature_end, assessment_feature_count)) \
                            + list(range(assessment_feature_start+4, assessment_feature_end, assessment_feature_count)) \
                            + list(range(assessment_feature_start+7, assessment_feature_end, assessment_feature_count)) \
                            + [assessall_feature_start + 1, 
                               assessall_feature_start + 3,
                               assessall_feature_start + 5]]
    count_feature_indexes = [ x - feature_col_start for x in \
                             list(range(assessment_feature_start, assessment_feature_end, assessment_feature_count)) \
                             + [assessall_feature_start, 
                                assessall_feature_start + 6]]
    feature_specs_df['is_sum'].iloc[sum_feature_indexes] = 1
    feature_specs_df['contains_nan'].iloc[sum_feature_indexes] = 0
    feature_specs_df['contains_nan'].iloc[count_feature_indexes] = 0
    Xy_df.loc[:, feature_specs_df[feature_specs_df['contains_nan'] == 0]['feature_name'].values] = 0
#     for f in feature_specs_df[feature_specs_df['is_list'] == 1]['feature_name'].values:
#         Xy_df.loc[:, f] = Xy_df.loc[:, f].map(lambda x : np.array([]))

    first_row_df = proc_df.groupby('installation_id').head(1).reset_index().set_index('installation_id')[['index']]
    # first_row_df: records the index of the first row of each installation_id in proc_df.
    temp_df = Xy_df.reset_index(drop=True)
    last_assessment_df = temp_df[temp_df['is_last_assessment'] == 1].reset_index().set_index('installation_id')[['index']]
    # last_assessment_df: records the row index of the last assessment in Xy_df.

    # fill in feature columns:
    for i in range(Xy_df.shape[0]):
        uid = Xy_df['installation_id'].iloc[i]
        is_user_first_assessment = (Xy_df['is_first_assessment'].iloc[i] == 1)
        is_user_last_assessment  = (Xy_df['is_last_assessment'].iloc[i] == 1)
        row_end = last_assessment_df.loc[uid, 'index'] + 1

        if is_user_first_assessment:
            proc_start_index = first_row_df.loc[uid, 'index']
        else:
            proc_start_index = Xy_df.index[i - 1]

        for j in range(proc_start_index, Xy_df.index[i]):
            sess_title = proc_df.loc[j, 'title']
            sess_type = proc_df.loc[j, 'type']
            fill_col_start = feature_count_df.loc[sess_title, 'column_start']
            fill_col_count = feature_count_df.loc[sess_title, 'column_count']
            fill_col_end = fill_col_start + fill_col_count
            fill_list = proc_df.iloc[j, 6:(6 + fill_col_count)].values
            if sess_type != 'Assessment':
                for k in range(fill_col_count):
                    if np.isnan(fill_list[k]):
                        continue

#                     col = Xy_df.columns[fill_col_start+k]
#                     arr1 = np.array(Xy_df[col].iloc[i:row_end])
#                     arr2 = np.array([fill_list[k]] * len(arr1))
#                     Xy_df[col].iloc[i:row_end] = arr2 # np.nanmean(np.array([arr1, arr2]), axis=0)
            
            if sess_type == 'Assessment':
                sess_duration = fill_list[1]  # duration of current assessment session
                sess_accuracy = fill_list[2]  # accuracy of current assessment session
                sess_accgroup = fill_list[3]  # accuracy_group of current assessment session
                sess_percentile = fill_list[4]  # percentile of current assessment session
                sess_perc_growth = fill_list[5]  # percentile_growth of current assessment session
                
                # fill in assessment features: # [count, dur_sum, dur_last, dur_mean, acc_sum, acc_last, acc_mean, group_sum, group_last, group_mean]
                [col0, col1, col2, col3, col4, col5, col6, col7, col8, col9] = Xy_df.columns[fill_col_start : fill_col_end]
                count = Xy_df[col0].iloc[i] + 1
                sum_duration = Xy_df[col1].iloc[i] + sess_duration
                sum_accuracy = Xy_df[col4].iloc[i] + sess_accuracy
                sum_accgroup = Xy_df[col7].iloc[i] + sess_accgroup
                Xy_df[col0].iloc[i:row_end] = count  
                Xy_df[col1].iloc[i:row_end] = sum_duration
                Xy_df[col2].iloc[i:row_end] = sess_duration         # last duration
                Xy_df[col3].iloc[i:row_end] = sum_duration / count  # mean duration
                Xy_df[col4].iloc[i:row_end] = sum_accuracy
                Xy_df[col5].iloc[i:row_end] = sess_accuracy         # last accuracy
                Xy_df[col6].iloc[i:row_end] = sum_accuracy / count  # mean accuracy
                Xy_df[col7].iloc[i:row_end] = sum_accgroup
                Xy_df[col8].iloc[i:row_end] = sess_accgroup         # last accgroup
                Xy_df[col9].iloc[i:row_end] = sum_accgroup / count  # mean accgroup
            
                # fill in assessment overall features: [count, group_sum, group_mean, percentile_sum, percentile_mean,
                #                                        perc_growth_sum, perc_growth_count, perc_growth_mean]
                [col0, col1, col2, col3, col4, col5, col6, col7] = Xy_df.columns[assessall_feature_start : assessall_feature_end]
                count = Xy_df[col0].iloc[i] + 1
                sum_accgroup = Xy_df[col1].iloc[i] + sess_accgroup
                sum_percentile = Xy_df[col3].iloc[i] + sess_percentile
                if ~np.isnan(sess_perc_growth):
                    sum_perc_growth = Xy_df[col5].iloc[i] + sess_perc_growth
                    count_perc_growth = Xy_df[col6].iloc[i] + 1
                else:
                    sum_perc_growth = Xy_df[col5].iloc[i]
                    count_perc_growth = Xy_df[col6].iloc[i]
                Xy_df[col0].iloc[i:row_end] = count  # count of all assessment sessions
                Xy_df[col1].iloc[i:row_end] = sum_accgroup
                Xy_df[col2].iloc[i:row_end] = sum_accgroup / count   # mean accuracy_group
                Xy_df[col3].iloc[i:row_end] = sum_percentile
                Xy_df[col4].iloc[i:row_end] = sum_percentile / count  # mean percentile
                Xy_df[col5].iloc[i:row_end] = sum_perc_growth
                Xy_df[col6].iloc[i:row_end] = count_perc_growth
                Xy_df[col7].iloc[i:row_end] = sum_perc_growth / count_perc_growth  # mean perc_growth
    
    # fill in feature_type in feature_specs_df
    feature_specs_df.set_index('feature_name', inplace=True)
    for sess_type in ['Bird Measurer (Assessment)', 'Cart Balancer (Assessment)', 'Cauldron Filler (Assessment)', 'Chest Sorter (Assessment)', 
                      'Mushroom Sorter (Assessment)']:
        feature_specs_df.loc[str(sess_type + '_00'), 'feature_type'] = 'assessment__count'
        feature_specs_df.loc[str(sess_type + '_01'), 'feature_type'] = 'assessment__dur_sum'
        feature_specs_df.loc[str(sess_type + '_02'), 'feature_type'] = 'assessment__dur_last'
        feature_specs_df.loc[str(sess_type + '_03'), 'feature_type'] = 'assessment__dur_mean'
        feature_specs_df.loc[str(sess_type + '_04'), 'feature_type'] = 'assessment__acc_sum'
        feature_specs_df.loc[str(sess_type + '_05'), 'feature_type'] = 'assessment__acc_last'
        feature_specs_df.loc[str(sess_type + '_06'), 'feature_type'] = 'assessment__acc_mean'
        feature_specs_df.loc[str(sess_type + '_07'), 'feature_type'] = 'assessment__group_sum'
        feature_specs_df.loc[str(sess_type + '_08'), 'feature_type'] = 'assessment__group_last'
        feature_specs_df.loc[str(sess_type + '_09'), 'feature_type'] = 'assessment__group_mean'

    feature_specs_df.loc['Assessment Overall_00', 'feature_type'] = 'assessment_overall__count'
    feature_specs_df.loc['Assessment Overall_01', 'feature_type'] = 'assessment_overall__group_sum'
    feature_specs_df.loc['Assessment Overall_02', 'feature_type'] = 'assessment_overall__group_mean'
    feature_specs_df.loc['Assessment Overall_03', 'feature_type'] = 'assessment_overall__perc_sum'
    feature_specs_df.loc['Assessment Overall_04', 'feature_type'] = 'assessment_overall__perc_mean'
    feature_specs_df.loc['Assessment Overall_05', 'feature_type'] = 'assessment_overall__perc_grow_sum'
    feature_specs_df.loc['Assessment Overall_06', 'feature_type'] = 'assessment_overall__perc_grow_count'
    feature_specs_df.loc['Assessment Overall_07', 'feature_type'] = 'assessment_overall__perc_grow_mean'
    
    Xy_df['others_00'] = proc_df_assessment['time_from_last_trial']
    feature_specs_df.loc['others_00', 'feature_type'] = 'time_from_last_trial'
    feature_specs_df.reset_index(inplace=True)
    
    t1 = time.time()
    print('Finish creating Xy_df. Time: {0:.1f} seconds.'.format((t1 - t0)))
    Xy_df.to_csv(Xy_csv_path, index=False)
    feature_specs_df.to_csv(feature_specs_path, index=False)
        

        
#################################################################
# Process individual sessions: Assessment sessions
#################################################################

def process_sess_Assessment(ass_df, is_test):
    """
    Return: [tot_attempt, accuracy, accuracy_group]
                tot_attempt = num_correct + num_incorrect
    
    If is_test and (ass_df.shape[0] == 1 and ass_df['event_code'].iloc[0] == 2000):
        return [np.nan] * 4
    """
    
    [tot_attempt, duration, accuracy, accuracy_group, keep] = [np.nan] * 4 + [False]
    
    if is_test and (ass_df.shape[0] == 1 and ass_df['event_code'].iloc[0] == 2000):
        return [tot_attempt, duration, accuracy, accuracy_group, True]
    
    code = 4100
    ass_title = ass_df['title'].iloc[0]
    ass_code_list = ass_df['event_code'].unique()
    if ass_title == 'Bird Measurer (Assessment)':
        code = 4110
    if code not in ass_code_list:   # Assessment not submitted.
        return [tot_attempt, duration, accuracy, accuracy_group, False]

    temp_df = ass_df[ass_df['event_code'] == code]
    duration = json.loads(temp_df['event_data'].iloc[-1])['game_time']
    num_correct = temp_df['event_data'].str.contains('true').sum()
    num_incorrect = temp_df.shape[0] - num_correct
    accuracy = num_correct / (num_correct + num_incorrect)
    accuracy_group = 1
    if accuracy == 1:
        accuracy_group = 3
    elif accuracy == 0.5:
        accuracy_group = 2
    elif accuracy == 0:
        accuracy_group = 0
    tot_attempt = num_correct + num_incorrect
    return [tot_attempt, duration, accuracy, accuracy_group, True]
    
    
#################################################################
# Process individual sessions: Game sessions
#################################################################


"""
Test code:

games = train_df[train_df['title'] == 'Crystals Rule']
print('Rows:', games.shape[0])
games_sessions = games['game_session'].unique()
print('Sessions:', len(games_sessions))
t0 = time.time()
for sess in games_sessions:
    sss = games[games['game_session'] == sess]
    process_sess_game_City_13(sss)
t1 = time.time()
print('Runtime: {0:.1f} seconds.'.format((t1 - t0)))

"""


def process_sess_game_City_02(game_df):
    """
    Treetop City: All Star Sorting (match the dinos' height with caves)
    
    ** ladders: 3 **
    Levels/Rounds: Round 1 (3 dino), Round 2 (4 dino), Round 3 (5 dino). 
        User can play it again (Round 4, Round 5, ...)
    
    Info to extract: 
        - attampts (each round corresponding to # dino correct attampt.)
        - accuracy (= correct_attampts / total_attampts. )
        - duration (if a round is not completed, leave duration blank)
        for:
            Round 1,
            Round 2,
            Round 3,
    
    Event-code:
        Assess code for each level: 2030
        Attampt correct: 3021 (note to remove the 2 attampts in round0)
        Attampt incorrect: 3020
        
    Runtime on train_df:
        Rows: 509344
        Sessions: 5917
        Runtime: 261.4 seconds.
    """
    
    [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur] = [np.nan] * 10
    ass_df = game_df[game_df['event_code'] == 2030]
    tot_misses = game_df[game_df['event_code'] == 3020].shape[0]
    tot_attempts = tot_misses + game_df[game_df['event_code'] == 3021].shape[0]
    if tot_attempts <= 0:
        return [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur]
    
    tot_round = ass_df.shape[0]
    df = pd.DataFrame(columns=['round', 'misses', 'attempts', 'duration'])
    miss_list = [0] * 100
    miss_df = game_df[game_df['event_code'] == 3020]
    for i in range(miss_df.shape[0]):
        msg = json.loads(miss_df['event_data'].iloc[i])
        miss_list[msg['round']] += 1
    
    for i in range(ass_df.shape[0]):
        msg = json.loads(ass_df['event_data'].iloc[i])
        df.loc[i] = [i + 1, 
                     miss_list[i + 1], 
                     miss_list[i + 1] + 1, 
                     msg['duration']]
    if tot_attempts - df['attempts'].sum() > 0:
        df.loc[df.shape[0]] = [df.shape[0] + 1, 
                               tot_misses - df['misses'].sum(), 
                               tot_attempts - df['attempts'].sum(), 
                               np.nan]
    while df.shape[0] < 3:
        df.loc[df.shape[0]] = [df.shape[0] + 1, np.nan, np.nan, np.nan]
    df['acc'] = 1 - df['misses'] / df['attempts']

    # If user plays again, take average of Round 1 & 4, Round 2 & 5, ...
    df_ave = pd.DataFrame(columns=['round', 'ave_attempts', 'ave_acc', 'ave_duration'])
    for i in range(3):
        df_ave.loc[i] = [i + 1, 
                         df['attempts'][i::3].mean(),
                         df['acc'][i::3].mean(),
                         df['duration'][i::3].mean()]

#     if tot_round < 3:
#         print(df)
#         print(df_ave)
#         print('\n')
    
    rd1_att = df_ave['ave_attempts'][0]
    rd2_att = df_ave['ave_attempts'][1]
    rd3_att = df_ave['ave_attempts'][2]
    rd1_acc = df_ave['ave_acc'][0]
    rd2_acc = df_ave['ave_acc'][1]
    rd3_acc = df_ave['ave_acc'][2]
    rd1_dur = df_ave['ave_duration'][0]
    rd2_dur = df_ave['ave_duration'][1]
    rd3_dur = df_ave['ave_duration'][2]
    return [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur]


def process_sess_game_City_10(game_df):
    """
    Treetop City: Air Show (dino jump from a distance, compete with the other team)
    
    ** no ladder **
    Levels/Rounds: Round 1 -- Round 2 -- Round 3 -- win. 
        Round 0 is for practice and doesn't generate 2030 code.
        User can play it again (Round 4, Round 5, ...)
    
    Info to extract: 
        - tot_attampts (correct_attampts = 2030 counts, incorrect_attampts = tot_misses)
        - average_accuracy (correct_attampts / tot_attampts)
        - average_duration
    
    Event-code:
        Assess code for each level: 2030
        
    Runtime on train_df:
        Rows: 306239
        Sessions: 3400
        Runtime: 47.7 seconds.
    """
    
    [tot_attampts, ave_acc, ave_duration] = [np.nan] * 3
    ass_df = game_df[game_df['event_code'] == 2030]
    if ass_df.shape[0] > 0:
        correct_attampts = ass_df.shape[0]
        tot_misses = 0
        total_duration = 0
        for i in range(ass_df.shape[0]):
            msg = json.loads(ass_df['event_data'].iloc[i])
            tot_misses += msg['misses']
            total_duration += msg['duration']
        tot_attampts = correct_attampts + tot_misses
        ave_acc = correct_attampts / tot_attampts
        ave_duration = total_duration / tot_attampts
    return [tot_attampts, ave_acc, ave_duration]



def process_sess_game_City_13(game_df):
    """
    Treetop City: Crystals Rule (measure the length of a crystal with different objects)
    
    ** ladders: 3 **
    Levels/Rounds: Round 1-3 is easy, Round 4-6 is harder, Round 7-9 is hardest. 
        User can play it again.
    
    Info to extract:
        - tot_round
        - ave_attampts (for each round, attempts = misses + 1)
        - ave_accuracy 
        - ave_duration (if a round is not completed, leave duration blank)
        for:
            Round 1-3,
            Round 4-6,
            Round 7-9,
    
    Event-code:
        Assess code for each level: 2030
        
    Runtime on train_df:
        Rows: 453852
        Sessions: 3145
        Runtime: 100.8 seconds.
    """
    
    [tot_round, lv1_att, lv1_acc, lv1_dur, lv2_att, lv2_acc, lv2_dur, lv3_att, lv3_acc, lv3_dur] = [np.nan] * 10
    ass_df = game_df[game_df['event_code'] == 2030]
    if ass_df.shape[0] <= 0:
        return [tot_round, lv1_att, lv1_acc, lv1_dur, lv2_att, lv2_acc, lv2_dur, lv3_att, lv3_acc, lv3_dur]
    
    tot_round = ass_df.shape[0]
    df = pd.DataFrame(columns=['round', 'misses', 'attempts', 'duration'])
    for i in range(ass_df.shape[0]):
        msg = json.loads(ass_df['event_data'].iloc[i])
        df.loc[i] = [i + 1, msg['misses'], msg['misses'] + 1, msg['duration']]
    df['acc'] = 1 - df['misses'] / df['attempts']

    # If user plays again, take average 
    df_ave = pd.DataFrame(columns=['level', 'ave_attempts', 'ave_acc', 'ave_duration'])
    df_ave.loc[0] = [1, 
                     df['attempts'].iloc[[i for i in range(df.shape[0]) if i % 9 < 3]].mean(),
                     df['acc'].iloc[[i for i in range(df.shape[0]) if i % 9 < 3]].mean(),
                     df['duration'].iloc[[i for i in range(df.shape[0]) if i % 9 < 3]].mean()]
    df_ave.loc[1] = [2, 
                     df['attempts'].iloc[[i for i in range(df.shape[0]) if (i % 9 >= 3) and (i % 9 < 6)]].mean(),
                     df['acc'].iloc[[i for i in range(df.shape[0]) if  (i % 9 >= 3) and (i % 9 < 6)]].mean(),
                     df['duration'].iloc[[i for i in range(df.shape[0]) if  (i % 9 >= 3) and (i % 9 < 6)]].mean()]
    df_ave.loc[2] = [3, 
                     df['attempts'].iloc[[i for i in range(df.shape[0]) if i % 9 >= 6]].mean(),
                     df['acc'].iloc[[i for i in range(df.shape[0]) if i % 9 >= 6]].mean(),
                     df['duration'].iloc[[i for i in range(df.shape[0]) if i % 9 >= 6]].mean()]
    
#     if tot_round > 9:
#         print(df)
#         print(df_ave)
#         print('\n')
    
    lv1_att = df_ave['ave_attempts'][0]
    lv2_att = df_ave['ave_attempts'][1]
    lv3_att = df_ave['ave_attempts'][2]
    lv1_acc = df_ave['ave_acc'][0]
    lv2_acc = df_ave['ave_acc'][1]
    lv3_acc = df_ave['ave_acc'][2]
    lv1_dur = df_ave['ave_duration'][0]
    lv2_dur = df_ave['ave_duration'][1]
    lv3_dur = df_ave['ave_duration'][2]
    return [tot_round, lv1_att, lv1_acc, lv1_dur, lv2_att, lv2_acc, lv2_dur, lv3_att, lv3_acc, lv3_dur]



def process_sess_game_Peak_03(game_df):
    """
    Magma Peak: Scrub-A-Dub (match animals' size with tub, soap and water)

    ** ladders: 4 **
    Levels/Rounds: level 1-3: 2 animals; level 4-6: 3 animals; ... max 5 animals
    
    Info to extract: 
        - attampts (each level corresponding to 1 correct attampt.)
        - accuracy (= correct_attampts / total_attampts)
        - duration (if not complete each 3-level, leave duration blank)
        for:
            Level 1-3 (2 animals),
            Level 4-6 (3 animals),
            Level 7-9 (4 animals),
            Level 10-12 (5 animals),
               (Level beyond 12: ignored).
    
    Event-code:
        Assess code for each level: 2050
        Attampt correct: 3021
        Attampt incorrect: 3020
        
    Runtime on train_df:
        Rows: 1016837
        Sessions: 6235
        Runtime: 352.5 seconds.
    """
    [lv3_att, lv3_acc, lv3_dur, lv6_att, lv6_acc, lv6_dur, lv9_att, lv9_acc, lv9_dur, lv12_att, lv12_acc, lv12_dur] = [np.nan] * 12
    ass_df = game_df[game_df['event_code'] == 2050]
    df = pd.DataFrame(columns=['level', 'misses', 'attempts', 'duration'])
    for i in range(ass_df.shape[0]):
        msg = json.loads(ass_df['event_data'].iloc[i])
        df.loc[i] = [i + 1, msg['misses'], msg['misses'] + 1, msg['duration']]
    res_misses = game_df[game_df['event_code'] == 3020].shape[0]
    res_attempts = res_misses + game_df[game_df['event_code'] == 3021].shape[0]
    if res_attempts <= 0:
        return [lv3_att, lv3_acc, lv3_dur, lv6_att, lv6_acc, lv6_dur, lv9_att, lv9_acc, lv9_dur, lv12_att, lv12_acc, lv12_dur]
    
    if ass_df.shape[0] < 3:
        lv3_att = res_attempts
        lv3_mis = res_misses
        lv3_acc = 1 - lv3_mis / lv3_att
    elif ass_df.shape[0] < 6:
        lv3_att = df['attempts'][0:3].sum()
        lv3_mis = df['misses'][0:3].sum()
        lv3_acc = 1 - lv3_mis / lv3_att
        lv3_dur = df['duration'][0:3].sum()
        if res_attempts - lv3_att > 0:
            lv6_att = res_attempts - lv3_att
            lv6_mis = res_misses - lv3_mis
            lv6_acc = 1 - lv6_mis / lv6_att
    elif ass_df.shape[0] < 9:
        lv3_att = df['attempts'][0:3].sum()
        lv3_mis = df['misses'][0:3].sum()
        lv3_acc = 1 - lv3_mis / lv3_att
        lv3_dur = df['duration'][0:3].sum()
        lv6_att = df['attempts'][3:6].sum()
        lv6_mis = df['misses'][3:6].sum()
        lv6_acc = 1 - lv6_mis / lv6_att
        lv6_dur = df['duration'][3:6].sum()
        if res_attempts - lv3_att - lv6_att > 0:
            lv9_att = res_attempts - lv3_att - lv6_att
            lv9_mis = res_misses - lv3_mis - lv6_mis
            lv9_acc = 1 - lv9_mis / lv9_att
    elif ass_df.shape[0] < 12:
        lv3_att = df['attempts'][0:3].sum()
        lv3_mis = df['misses'][0:3].sum()
        lv3_acc = 1 - lv3_mis / lv3_att
        lv3_dur = df['duration'][0:3].sum()
        lv6_att = df['attempts'][3:6].sum()
        lv6_mis = df['misses'][3:6].sum()
        lv6_acc = 1 - lv6_mis / lv6_att
        lv6_dur = df['duration'][3:6].sum()
        lv9_att = df['attempts'][6:9].sum()
        lv9_mis = df['misses'][6:9].sum()
        lv9_acc = 1 - lv9_mis / lv9_att
        lv9_dur = df['duration'][6:9].sum()
        if res_misses - lv3_mis - lv6_mis - lv9_mis > 0:
            lv12_att = res_attempts - lv3_att - lv6_att - lv9_att
            lv12_mis = res_misses - lv3_mis - lv6_mis - lv9_mis
            lv12_acc = 1 - lv12_mis / lv12_att
    else: 
        lv3_att = df['attempts'][0:3].sum()
        lv3_mis = df['misses'][0:3].sum()
        lv3_acc = 1 - lv3_mis / lv3_att
        lv3_dur = df['duration'][0:3].sum()
        lv6_att = df['attempts'][3:6].sum()
        lv6_mis = df['misses'][3:6].sum()
        lv6_acc = 1 - lv6_mis / lv6_att
        lv6_dur = df['duration'][3:6].sum()
        lv9_att = df['attempts'][6:9].sum()
        lv9_mis = df['misses'][6:9].sum()
        lv9_acc = 1 - lv9_mis / lv9_att
        lv9_dur = df['duration'][6:9].sum()
        lv12_att = df['attempts'][9:12].sum()
        lv12_mis = df['misses'][9:12].sum()
        lv12_acc = 1 - lv12_mis / lv12_att
        lv12_dur = df['duration'][9:12].sum()
        
    return [lv3_att, lv3_acc, lv3_dur, lv6_att, lv6_acc, lv6_dur, lv9_att, lv9_acc, lv9_dur, lv12_att, lv12_acc, lv12_dur]



def process_sess_game_Peak_06(game_df):
    """
    Magma Peak: Dino Drink
    
    ** ladders: 3 **
    Levels/Rounds: Round 1 (2 dino), Round 2 (3 dino), Round 3 (5 dino). 
        User can play it again (Round 4, Round 5, ...)
    
    Info to extract: 
        - tot_round
        - attampts (each round corresponding to # dino correct attampt.)
        - accuracy (= correct_attampts / total_attampts. )
        - duration (if a round is not completed, leave duration blank)
        for:
            Round 1,
            Round 2,
            Round 3,
    
    Event-code:
        Assess code for each level: 2030
        Attampt correct: 3021 (note to remove the 2 attampts in round0)
        Attampt incorrect: 3020
        
    Runtime on train_df:
        Rows: 492916
        Sessions: 4998
        Runtime: 154.6 seconds.
    """
    
    [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur] = [np.nan] * 10
    ass_df = game_df[game_df['event_code'] == 2030]
    tot_misses = game_df[game_df['event_code'] == 3020].shape[0]
    tot_attempts = tot_misses + game_df[game_df['event_code'] == 3021].shape[0]
    if tot_attempts <= 0:
        return [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur]
    
    tot_round = ass_df.shape[0]
    df = pd.DataFrame(columns=['round', 'misses', 'attempts', 'duration'])
    correct_required = [2, 3, 6] * 30
    for i in range(ass_df.shape[0]):
        msg = json.loads(ass_df['event_data'].iloc[i])
        df.loc[i] = [i + 1, msg['misses'], msg['misses'] + correct_required[i], msg['duration']]
    if tot_attempts - df['attempts'].sum() > 0:
        df.loc[df.shape[0]] = [df.shape[0] + 1, tot_misses - df['misses'].sum(), tot_attempts - df['attempts'].sum(), np.nan]
    while df.shape[0] < 3:
        df.loc[df.shape[0]] = [df.shape[0] + 1, np.nan, np.nan, np.nan]

    # If user plays again, take average of Round 1 & 4, Round 2 & 5, ...
    df_ave = pd.DataFrame(columns=['round', 'tot_misses', 'tot_attempts', 'ave_attempts', 'ave_duration'])
    for i in range(3):
        df_ave.loc[i] = [i + 1, 
                         df['misses'][i::3].sum(), 
                         df['attempts'][i::3].sum(), 
                         df['attempts'][i::3].mean(), 
                         df['duration'][i::3].mean()]
        if np.isnan(df['attempts'].loc[i]):
            df_ave['tot_misses'][i] = np.nan
            df_ave['tot_attempts'][i] = np.nan
    df_ave['acc'] = 1 - df_ave['tot_misses'] / df_ave['tot_attempts']
    
#     if tot_round > 12:
#         print(df)
#         print(df_ave)
    
    rd1_att = df_ave['ave_attempts'][0]
    rd2_att = df_ave['ave_attempts'][1]
    rd3_att = df_ave['ave_attempts'][2]
    rd1_acc = df_ave['acc'][0]
    rd2_acc = df_ave['acc'][1]
    rd3_acc = df_ave['acc'][2]
    rd1_dur = df_ave['ave_duration'][0]
    rd2_dur = df_ave['ave_duration'][1]
    rd3_dur = df_ave['ave_duration'][2]
    return [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur]



def process_sess_game_Peak_07(game_df):
    """
    Magma Peak: Bubble Bath (how many times it need to fill in the bath with a small container.)

    ** no ladder **
    Levels/Rounds: No ladder. So just take the average of all rounds.
    
    Info to extract: 
        - tot_attampts (how many rounds)
        - average_accuracy (total_misses / attampts)
        - average_duration
    
    Event-code:
        Assess code for each level: 2030
        Attampt incorrect: 3020
        
    Runtime on train_df:
        Rows: 458972
        Sessions: 4177
        Runtime: 86.8 seconds.
    """
    
    [tot_attampts, ave_acc, ave_duration] = [np.nan] * 3
    ass_df = game_df[game_df['event_code'] == 2030]
    if ass_df.shape[0] > 0:
        tot_attampts = ass_df.shape[0]
        misses = 0
        total_duration = 0
        for i in range(ass_df.shape[0]):
            msg = json.loads(ass_df['event_data'].iloc[i])
            misses += msg['misses']
            total_duration += msg['duration']
        ave_acc = 1 - misses / tot_attampts
        ave_duration = total_duration / tot_attampts
    return [tot_attampts, ave_acc, ave_duration]
        


def process_sess_game_Peak_09(game_df):
    """
    Magma Peak: Dino Dive
    
    ** ladders: 2 **
    Levels/Rounds: Round 1, Round 2, Round 3, Round 4. (Round 0 is introduction, not count.)
        Round 1, Round 2 are easier; Round 3, Round 4 are harder.
        User can play it again (Round 4, Round 5, ...)
    
    Info to extract: 
        - tot_round
        - attampts (each round corresponding to 1 correct attampt.)
        - accuracy (= correct_attampts / total_attampts. If a round is not completed, then the accuracy must be 0.)
        - duration (if a round is not completed, leave duration blank)
        for:
            Round 1 & 2,
            Round 3 & 4,
    
    Event-code:
        Assess code for each level: 2030
        Attampt correct: 3021 (note to remove the 2 attampts in round0)
        Attampt incorrect: 3020
        
    Runtime on train_df:
        Rows: 427655
        Sessions: 3894
        Runtime: 122.1 seconds.
    """
    
    [tot_round, lv1_att, lv1_acc, lv1_dur, lv2_att, lv2_acc, lv2_dur] = [np.nan] * 7
    ass_df = game_df[game_df['event_code'] == 2030]
    tot_misses = game_df[game_df['event_code'] == 3020].shape[0]
    tot_attempts = tot_misses + game_df[game_df['event_code'] == 3021].shape[0] - 2  # remove the 2 attampts in round0
    if tot_attempts <= 0:
        return [tot_round, lv1_att, lv1_acc, lv1_dur, lv2_att, lv2_acc, lv2_dur]
    
    tot_round = ass_df.shape[0]
    df = pd.DataFrame(columns=['round', 'misses', 'attempts', 'duration'])
    for i in range(ass_df.shape[0]):
        msg = json.loads(ass_df['event_data'].iloc[i])
        df.loc[i] = [i + 1, msg['misses'], msg['misses'] + 1, msg['duration']]
    if tot_attempts - df['attempts'].sum() > 0:
        df.loc[df.shape[0]] = [df.shape[0] + 1, tot_misses - df['misses'].sum(), tot_attempts - df['attempts'].sum(), np.nan]
    while df.shape[0] < 4:
        df.loc[df.shape[0]] = [df.shape[0] + 1, np.nan, np.nan, np.nan]
    df['acc'] = 1 - df['misses'] / df['attempts']

    # If user plays again, take average of Round 1 & 5, Round 2 & 6, ...
    df_ave = pd.DataFrame(columns=['level', 'ave_attempts', 'ave_acc', 'ave_duration'])
    df_ave.loc[0] = [1, 
                     df['attempts'].iloc[[i for i in range(df.shape[0]) if i % 4 < 2]].mean(),
                     df['acc'].iloc[[i for i in range(df.shape[0]) if i % 4 < 2]].mean(),
                     df['duration'].iloc[[i for i in range(df.shape[0]) if i % 4 < 2]].mean()]
    df_ave.loc[1] = [2, 
                     df['attempts'].iloc[[i for i in range(df.shape[0]) if i % 4 >= 2]].mean(),
                     df['acc'].iloc[[i for i in range(df.shape[0]) if i % 4 >= 2]].mean(),
                     df['duration'].iloc[[i for i in range(df.shape[0]) if i % 4 >= 2]].mean()]
    
#     if tot_round == 4:
#         print(df)
#         print(df_ave)
#         print('\n')
    
    lv1_att = df_ave['ave_attempts'][0]
    lv2_att = df_ave['ave_attempts'][1]
    lv1_acc = df_ave['ave_acc'][0]
    lv2_acc = df_ave['ave_acc'][1]
    lv1_dur = df_ave['ave_duration'][0]
    lv2_dur = df_ave['ave_duration'][1]
    return [tot_round, lv1_att, lv1_acc, lv1_dur, lv2_att, lv2_acc, lv2_dur]

    
def process_sess_game_Cave_01(game_df):
    """
    Crystal Caves: Chow Time (balance the scale by adding/removing food)
    
    ** no ladder **
    Levels/Rounds:
    
    Info to extract: 
        - tot_attampts (correct_attampts = 2030 counts, incorrect_attampts = tot_misses)
        - average_accuracy (correct_attampts / tot_attampts)
        - average_duration
    
    Event-code:
        Assess code for each level: 2030
        
    Runtime on train_df:
        Rows: 1150974
        Sessions: 10196
        Runtime: 480.4 seconds.
    """
    [tot_attampts, ave_acc, ave_duration] = [np.nan] * 3
    ass_df = game_df[game_df['event_code'] == 2030]
    if ass_df.shape[0] > 0:
        correct_attampts = ass_df.shape[0]
        tot_misses = 0
        total_duration = 0
        for i in range(ass_df.shape[0]):
            msg = json.loads(ass_df['event_data'].iloc[i])
            tot_misses += msg['misses']
            total_duration += msg['duration']
        tot_attampts = correct_attampts + tot_misses
        ave_acc = correct_attampts / tot_attampts
        ave_duration = total_duration / tot_attampts
    return [tot_attampts, ave_acc, ave_duration]

    
    
def process_sess_game_Cave_07(game_df):
    """
    Crystal Caves: Happy Camel (use scale to find the heaviest)
    
    ** ladders: 3 **
    Levels/Rounds: Round 1 (2 bowls), Round 2 (3 bowls), Round 3 (4 bowls). 
        User can play it again (Round 4, Round 5, ...)
    
    Info to extract: 
        - tot_round
        - attampts (each round corresponding 1 correct attampt.)
        - accuracy (= correct_attampts / total_attampts. )
        - duration (if a round is not completed, leave duration blank)
        for:
            Round 1,
            Round 2,
            Round 3,
    
    Event-code:
        Assess code for each level: 2030
        Attampt correct: 3021
        Attampt incorrect: 3020
        
    Runtime on train_df:
        Rows: 311543
        Sessions: 3689
        Runtime: 143.3 seconds.
    """
    
    [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur] = [np.nan] * 10
    ass_df = game_df[game_df['event_code'] == 2030]
    tot_misses = game_df[game_df['event_code'] == 3020].shape[0]
    tot_attempts = tot_misses + game_df[game_df['event_code'] == 3021].shape[0]
    if tot_attempts <= 0:
        return [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur]
    
    tot_round = ass_df.shape[0]
    df = pd.DataFrame(columns=['round', 'misses', 'attempts', 'duration'])
    for i in range(ass_df.shape[0]):
        msg = json.loads(ass_df['event_data'].iloc[i])
        df.loc[i] = [i + 1, msg['misses'], msg['misses'] + 1, msg['duration']]
    if tot_attempts - df['attempts'].sum() > 0:
        df.loc[df.shape[0]] = [df.shape[0] + 1, 
                               tot_misses - df['misses'].sum(), 
                               tot_attempts - df['attempts'].sum(), 
                               np.nan]
    while df.shape[0] < 3:
        df.loc[df.shape[0]] = [df.shape[0] + 1, np.nan, np.nan, np.nan]
    df['acc'] = 1 - df['misses'] / df['attempts']

    # If user plays again, take average of Round 1 & 4, Round 2 & 5, ...
    df_ave = pd.DataFrame(columns=['round', 'ave_attempts', 'ave_acc', 'ave_duration'])
    for i in range(3):
        df_ave.loc[i] = [i + 1, 
                         df['attempts'][i::3].mean(),
                         df['acc'][i::3].mean(),
                         df['duration'][i::3].mean()]

#     if tot_round > 6:
#         print(df)
#         print(df_ave)
#         print('\n')
    
    rd1_att = df_ave['ave_attempts'][0]
    rd2_att = df_ave['ave_attempts'][1]
    rd3_att = df_ave['ave_attempts'][2]
    rd1_acc = df_ave['ave_acc'][0]
    rd2_acc = df_ave['ave_acc'][1]
    rd3_acc = df_ave['ave_acc'][2]
    rd1_dur = df_ave['ave_duration'][0]
    rd2_dur = df_ave['ave_duration'][1]
    rd3_dur = df_ave['ave_duration'][2]
    return [tot_round, rd1_att, rd1_acc, rd1_dur, rd2_att, rd2_acc, rd2_dur, rd3_att, rd3_acc, rd3_dur]
    

    
def process_sess_game_Cave_09(game_df):
    """
    Leaf Leader: Leaf Leader (Tug of war)
    
    ** no ladder **
    Levels/Rounds:
    
    Info to extract: 
        - tot_attampts (correct_attampts = 2030 counts, incorrect_attampts = tot_misses)
        - average_accuracy (correct_attampts / tot_attampts)
        - average_duration
    
    Event-code:
        Assess code for each level: 2030
        
    Runtime on train_df:
        Rows: 282104
        Sessions: 3378
        Runtime: 48.8 seconds.
    """
    [tot_attampts, ave_acc, ave_duration] = [np.nan] * 3
    ass_df = game_df[game_df['event_code'] == 2030]
    if ass_df.shape[0] > 0:
        correct_attampts = ass_df.shape[0]
        tot_misses = 0
        total_duration = 0
        for i in range(ass_df.shape[0]):
            msg = json.loads(ass_df['event_data'].iloc[i])
            tot_misses += msg['misses']
            total_duration += msg['duration']
        tot_attampts = correct_attampts + tot_misses
        ave_acc = correct_attampts / tot_attampts
        ave_duration = total_duration / tot_attampts
    return [tot_attampts, ave_acc, ave_duration]


    
def process_sess_game_Cave_12(game_df):
    """
    Crystal Caves: Pan Balance (scale and weights)
    
    ** no ladder **
    Levels/Rounds:
    
    Info to extract: 
        - tot_attampts (correct_attampts = 2030 counts, incorrect_attampts = tot_misses)
        - average_accuracy (correct_attampts / tot_attampts)
        - average_duration
    
    Event-code:
        Assess code for each level: 2030
        
    Runtime on train_df:
        Rows: 384857
        Sessions: 3370
        Runtime: 56.2 seconds.
    """
    [tot_attampts, ave_acc, ave_duration] = [np.nan] * 3
    ass_df = game_df[game_df['event_code'] == 2030]
    if ass_df.shape[0] > 0:
        correct_attampts = ass_df.shape[0]
        tot_misses = 0
        total_duration = 0
        for i in range(ass_df.shape[0]):
            msg = json.loads(ass_df['event_data'].iloc[i])
            tot_misses += msg['misses']
            total_duration += msg['duration']
        tot_attampts = correct_attampts + tot_misses
        ave_acc = correct_attampts / tot_attampts
        ave_duration = total_duration / tot_attampts
    return [tot_attampts, ave_acc, ave_duration]
    
    

#################################################################
# Process individual sessions: Activity sessions
#################################################################



#################################################################
# Process individual sessions: Clip sessions
#################################################################