from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show
from logistic_regression import process_data, generate_dataset
import numpy as np

def cov(data):
    N = data.shape[1]
    C = empty((N, N))
    for j in range(N):
        C[j, j] = mean(data[:, j] * data[:, j])
        for k in range(N):
            C[j, k] = C[k, j] = mean(data[:, j] * data[:, k])
    return C

def pca(data, pc_count = None):
    """
        Principal component analysis using eigenvalues
    """
    d_1 = mean(data, 0)
    d_2 = std(data, 0)
    data -= mean(data, 0)
    data /= std(data, 0)
    C = cov(data)
    E, V = eigh(C)
    key = argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = dot(V.T, data.T).T
    return [U, [d_1, d_2, V]]

def pca_reconstruct(data, recst):
    data -= recst[0] # mean(data, 0)
    data /= recst[1] # std(data, 0) 
    U = dot(recst[2].T, data.T).T
    return U
  


def quad_mapping(data, k = 3):
  '''
    Mapping the original data to quadratic space, i.e.
    [X1, X2,..., Xn] -> [X1^2, X1X2, X1X3,..., X2^2,...,xn^2 ]
  '''
  [num, dim] = data.shape
  feature = np.zeros([num, k*dim], dtype = float)
  for idx in range(num):
    for i in range(dim):
      for pw in range(k):
        feature[idx, k*i+pw]  = np.power(data[idx, i],pw+1)
  return feature[:,:-k]



def add_const_column(data, k = 1):
  '''
    add a constant column vector for a datamatrix
    default constant = 1
  '''
  [num, dim] = data.shape
  new_data = np.zeros([num,dim+1], dtype = float)
  new_data[:, 0:dim] = data
  for i in range(num):
    new_data[i, dim] = k
  return new_data

def get_train_feature(data, dim = 100):
  print 'Mapping Data to High Dimensional Space: 10 dims...'
  tmp = quad_mapping(data)
  print 'PCA Selecting features: '+str(dim) +' features...'
  [tmp, rcnst] = pca(tmp, dim)
  train_X = add_const_column(tmp)
  return [train_X, rcnst]

def get_test_feature(data, rcnst):
  print 'Mapping Data to High Dimensional Space: 10 dims...'
  tmp = quad_mapping(data)
  print 'PCA Reconstruction...'
  tmp = pca_reconstruct(tmp, rcnst)
  test_X = add_const_column(tmp)
  return test_X




# Test section

if __name__ == '__main__' :
  """ test data """
  df = process_data('data/spam_train.csv')
  [data, labels] = generate_dataset(df)

  """ visualize """
  feature = quad_mapping(data)
  trans = pca(data[:,0:57], 3)[0]
  fig, (ax1, ax2) = subplots(1, 2)
  for i in range(300):
    if int(labels[i]) == 0:
      ax1.scatter(data[i, 0], data[i, 1], c = 'r')
      ax2.scatter(trans[i, 0], trans[i, 1], c = 'r')
    if int(labels[i]) == 1:
      ax1.scatter(data[i, 0], data[i, 1], c = 'b')
      ax2.scatter(trans[i, 0], trans[i, 1], c = 'b')
  ax1.set_title('Data Before PCA')
  ax2.set_title('Data After PCA')
  #show()
   
  trans_feature = pca(feature[:,:-3],100)[0]
  fig_2, (ax1_2, ax2_2) = subplots(1, 2)
  for i in range(300):
    if int(labels[i]) == 0:
      ax1_2.scatter(trans[i, 0], trans[i, 1], c = 'r')
      ax2_2.scatter(trans_feature[i, 0], trans_feature[i, 1], c = 'r')
    if int(labels[i]) == 1:
      ax1_2.scatter(trans[i, 0], trans[i, 1], c = 'b')
      ax2_2.scatter(trans_feature[i, 0], trans_feature[i, 1], c = 'b')
  ax1_2.set_title('Data Before poly-mapping')
  ax2_2.set_title('Data After poly-mapping')
  
  show()

