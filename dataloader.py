# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
DATA_ROOT = "./"

class DataLoader(object):
    def __init__(self):
        pass

    def get_data(self, path, beginWith=0):
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            rows = rows[beginWith:]
        return rows

    def write_to_file(self, data, filepath, split=","):
        newStr = ''
        result = data
        for row in result:
            for t in row[:-1]:
                if type(t) == str or not np.isnan(t):
                    newStr += str(t)
                newStr += split
            if type(t) == str or not np.isnan(row[-1]):
                newStr += str(row[-1])
            newStr += '\n'

        with open(filepath, "w") as outputFile:
            outputFile.write(newStr)
            outputFile.close()


def loadDataset(dataset, mr):
    seeds = [0,1,2,3,4]
    file_root = "./datasets_missing/"
    dataloader = DataLoader()
    mat = dataloader.get_data(os.path.join(file_root, dataset, "origin_data.csv"))
    mat = np.array(mat[1:], dtype=np.float32)
    mat_missing_list = []
    for seed in seeds:
        data = dataloader.get_data(os.path.join(file_root, dataset, f"mis_data_{mr}_{seed}.csv"))
        data = np.array(data[1:])
        data[data==""] = np.nan
        data = data.astype(np.float32)
        mat_missing_list.append(data)
    return mat, mat_missing_list


