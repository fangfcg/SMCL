import SMC
import SMCL
from utils import *
import time


def rmse(mat, imputed_mat, mask):
    error = 0
    count = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mask[i, j]:
                error += pow(mat[i, j] - imputed_mat[i, j], 2)
                count += 1

    if count == 0:
        return np.nan
    return np.sqrt(error / count)


def geo_distance(X, i, j):
    dis_x = (X[i][0] - X[j][0]) ** 2
    dis_y = (X[i][1] - X[j][1]) ** 2
    return np.sqrt(dis_x + dis_y)

def sparseList(mat,threshold, num):
    n = mat.shape[0]
    points = []
    for i in range(0, n):
        count = 0
        for j in range(0, n):
            dis = geo_distance(mat, i, j)
            if dis < threshold:
                 count += 1
        if count < num:
            points.append(i)
    return points

def SMCTest(mat, mat_, lmd, landmarkrow=[], method=1, k=6, iter=500, p=5, return_v = False):
    smc = SMC.SMC(mat_, k=k, alpha=0.0005, beta=0.0005, iterations=iter, lmd=lmd, p=p)
    start = time.time()
    training_process = smc.train(method=method)
    end = time.time()
    X_hat = smc.full_matrix()
    X_comp = smc.replace_nan(X_hat)
    mask = np.isnan(mat_)
    if return_v:
        return rmse(X_comp, mat, mask), X_comp, training_process, (end - start) * 1000, smc.V
    return rmse(X_comp, mat, mask), X_comp, training_process, (end-start) * 1000


def SMCLTest(mat, mat_, lmd, landmarkrow=[], method=1, k=6, iter=500, cluster=False, p=5, return_v= False):
    smcl = SMCL.SMCL(mat_, k=k, alpha=0.001, beta=0.001, iterations=iter, lmd=lmd, cluster=cluster, p=p)
    start = time.time()
    training_process = smcl.train(method=method)
    end = time.time()
    X_hat = smcl.full_matrix()
    X_comp = smcl.replace_nan(X_hat)
    mask = np.isnan(mat_)
    if return_v:
        return rmse(X_comp, mat, mask), X_comp, training_process, (end - start) * 1000, smcl.V
    return rmse(X_comp, mat, mask), X_comp, training_process, (end-start) * 1000





