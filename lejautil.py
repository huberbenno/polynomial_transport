import numpy as np

def leja_points_1d(n) :
    r = [1]
    if n > 0 : r.append(-1)
    if n > 1 : r.append(0)
    for j in range(3, n+1) :
        if j % 2 == 0 :
            r.append(-r[j-1])
        else :
            r.append(np.sqrt((r[int((j+1)/2)] + 1) / 2))
    return np.array(r).reshape((len(r),1))

def leja_points(multiset, d) :
    indices = [m.GetVector() for m in multiset.GetAllMultiIndices()]
    r = leja_points_1d(max([max(m) for m in indices]))
    p = np.zeros((len(indices), d))
    for i in range(len(indices)) :
        for j in range(min(len(indices[i]), d)) :
            p[i,j] = r[indices[i][j]]
    return p

if __name__ == '__main__' :
    print(leja_points_1d(10))
