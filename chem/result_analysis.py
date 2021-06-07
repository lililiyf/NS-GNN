import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os
import seaborn as sns
import copy


def analysis():
    # construct result dict
    dataset_list = ["muv", "bace", "bbbp", "clintox", "hiv", "sider", "tox21", "toxcast"]
    #dataset_list = ["muv", "bace", "bbbp", "clintox", "hiv", "toxcast"]
    # 10 random seed
    # seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    seed_list = [0]
    config_list = []
    config_list.append("param_masking_struct_6")
    # config_list.append("gin_nopretrain")
    # # config_list.append("gin_infomax")
    # # config_list.append("gin_edgepred")
    # config_list.append("gin_masking")
    # config_list.append("gin_contextpred")
    # config_list.append("gin_new_masking")
    # config_list.append("gin_new_masking_2")

    result_mat = np.zeros((len(seed_list), len(config_list), len(dataset_list)))
    help(open)

    with open("result_summary", "rb") as f:
        result = pickle.load(f)
        result_mat = result['result_mat']
    # print(result_mat)
    # print(dataset_list)
    # print(config_list)

    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('Sheet1')
    worksheet.write(0, 0, 'test_data')  # 不带样式的写入
    for i in range(len(dataset_list)):
        worksheet.write(1,i+2,dataset_list[i])
    for j in range(len(config_list)):
        worksheet.write(j+2,1,config_list[j])

    for j in range(len(config_list)):
        for i in range(len(dataset_list)):
            print(result_mat[0][j][i])
            worksheet.write(j+2,i+2,result_mat[0][j][i]*100)

    workbook.save('data_test3.xls')


if __name__ == "__main__":
    analysis()
