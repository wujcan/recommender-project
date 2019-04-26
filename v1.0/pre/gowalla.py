#!/usr/bin/env python
import numpy as np
import pandas as pd
import csv
from predata3 import PreData

def get_user_social_dict(path):
    u_social_dict = {}

    u_lines = open(path).readlines()
    for user in u_lines:
        temp = user.strip().split('\t')
        u_id = int(temp[0])
        f_id = int(temp[1])

        if u_id not in u_social_dict.keys():
            u_social_dict[u_id] = []
        u_social_dict[u_id].append(f_id)
    print('load user social network done.')
    return u_social_dict

def get_ratings(path):
    ratings = []
    ui_dict = {}

    lines = open(path, 'r').readlines()
    for line in lines:
        temp = line.strip().split('\t')
        u_id = int(temp[0])
        i_id = int(temp[-1])

        stamp = temp[1]
        ui_key = '%d-%d' % (u_id, i_id)
        if (ui_key in ui_dict.keys() and ui_dict[ui_key] < stamp) or (ui_key not in ui_dict.keys()):
            ui_dict[ui_key] = stamp
    print('remove duplicated user-item pairs')

    sorted_ui_dict = sorted(ui_dict.items(), key=lambda kv: kv[1])
    for ui in sorted_ui_dict:
        temp = ui[0].split('-')
        ratings.append([int(temp[0]), int(temp[1])])
    ratings = np.array(ratings)
    print('load ratings done')
    return ratings

def get_interactions(ratings):
    user_dict, item_dict = {}, {}
    for line in ratings:
        u_id = int(line[0])
        i_id = int(line[1])

        if u_id in user_dict.keys():
            user_dict[u_id].append(i_id)
        else:
            user_dict[u_id] = [i_id]

        if i_id in item_dict.keys():
            item_dict[i_id].append(u_id)
        else:
            item_dict[i_id] = [u_id]
    print('get user_dict and item_dict done')
    return user_dict, item_dict


if __name__ == '__main__':
    dataset = 'gowalla'
    n_u_f, n_i_f = 10, 10
    path = 'D:/PycharmProjects/codes_sigir19_depfm/Data/%s' % dataset

    user_social_dict = get_user_social_dict('D:/PycharmProjects/codes_sigir19_depfm/Data/%s/loc-gowalla_edges.txt' % dataset)
    ratings = get_ratings('D:/PycharmProjects/codes_sigir19_depfm/Data/%s/loc-gowalla_totalCheckins.txt' % dataset)

    user_dict, item_dict = get_interactions(ratings)

    predata = PreData(user_dict=user_dict, item_dict=item_dict, path=path,
                      user_social_dict=user_social_dict, n_u_f=n_u_f, n_i_f=n_i_f)



