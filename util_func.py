import numpy as np



def cos_dist(x, y):
    norm_x = x / (np.linalg.norm(x, axis=1, keepdims=True)+0.000001)
    norm_y = y / (np.linalg.norm(y, axis=1, keepdims=True)+0.000001)
    return 1 - np.matmul(norm_x, norm_y.T)

def categorical_dist(featmtx,categorylabels,categorynames):
    objdistmtx=cos_dist(featmtx,featmtx)#distance_matrix(featmtx,featmtx)
    catdistmtx=np.zeros((len(categorynames),len(categorynames)))
    for i in range(len(categorynames)):
        tmpind_i=(categorylabels== #['category']
                  categorynames[i]).to_numpy()
        for j in range(len(categorynames)):
            tmpind_j=(categorylabels== #['category']
                      categorynames[j]).to_numpy()
            catdistmtx[i,j]=np.mean(objdistmtx[tmpind_i,:][:,tmpind_j])
    return catdistmtx