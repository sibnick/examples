# https://stats.stackexchange.com/questions/410468/online-update-of-pearson-coefficient
import math
import torch


def pearson_corr(n, xn1, yn1, data):
    n1 = 1.0/(1 + n)
    mean_xn1 = data[:, 1] + n1 * (xn1[:] - data[:, 1])
    mean_yn1 = data[:, 2] + n1 * (yn1 - data[:, 2])
    nn1 = data[:, 3] + (xn1 - data[:, 1]) * (yn1 - mean_yn1)
    dn1 = data[:, 4] + (xn1 - data[:, 1]) * (xn1 - mean_xn1)
    en1 = data[:, 5] + (yn1 - data[:, 2]) * (yn1 - mean_yn1)
    r = nn1 / torch.sqrt(1e-6 + dn1 * en1)
    data[:, 0] = r + 1
    data[:, 1] = mean_xn1
    data[:, 2] = mean_yn1
    data[:, 3] = nn1
    data[:, 4] = dn1
    data[:, 5] = en1
    return r

def pearson_corr_scalar(n, xn1, yn1, data):
    n1 = 1.0 / (1 + n)
    (_, mean_xn0, mean_yn0, nn, dn, en) = data
    mean_xn1 = mean_xn0 + n1 * (xn1 - mean_xn0)
    mean_yn1 = mean_yn0 + n1 * (yn1 - mean_yn0)
    nn1 = nn + (xn1 - mean_xn0)*(yn1 - mean_yn1)
    dn1 = dn + (xn1 - mean_xn0)*(xn1 - mean_xn1)
    en1 = en + (yn1 - mean_yn0)*(yn1 - mean_yn1)
    r = nn1/math.sqrt(dn1 * en1)
    data[:] = torch.stack(list((r, mean_xn1, mean_yn1, nn1, dn1, en1)), dim=0)
    return r

x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
y = [10, 20, 30, 40, 50, 40, 30, 20, 10, 0, -10, -20, -30, -40, -50]

data = torch.zeros((3, 6))
for idx in range(len(x)):
    xx = torch.FloatTensor([x[idx], x[idx]+1, x[idx]])
    print(pearson_corr(3, xx, y[idx], data))
