import tensorflow as tf
from scipy.spatial.distance import mahalanobis
import numpy as np
import time
import itertools

# initialize data
num_obs = 100
num_var = 2
data = np.zeros([num_obs,num_var])

# column 0 var is indicator
# column 1 var is correlated

for i in range(num_obs):
    data[i][0] = np.random.randint(2) 
    time.sleep(0.01)
    data[i][1] = data[i][0]*50 + np.random.randint(51)


V = np.cov(data,rowvar=0)
Vm = np.matrix(V)

x = data[0]
y = data[1]

def maha(x,y,V):
    return np.sqrt((x-y).transpose() * V.I * (x-y).reshape(len(x),1))

def tf_maha(x,y,V):
    num_var = len(x)
    with tf.Session() as sess:
        xx = tf.placeholder(tf.types.float32,[num_var])
        yy = tf.placeholder(tf.types.float32,[num_var])
        vv = tf.placeholder(tf.types.float32,[num_var,num_var])
        vivi = tf.matrix_inverse(vv)
        diff = tf.sub(xx,yy)
        dt = tf.reshape(tf.transpose(diff),[1,len(x)])
        ds = tf.reshape(diff,[len(x),1])
        M1 = tf.matmul(dt,vivi)
        M2 = tf.matmul(M1,ds)
        output = tf.sqrt(M2)
        ans = sess.run([output],feed_dict = {xx:x,
                                       yy:y,
                                       vv:V})
    return ans

tf_maha(x,y,Vm)

def cov(data):
    '''
    Compute covariance matrix of data.
    Each row is an observation.
    Each column is a variable.
    '''
    num_obs, num_vars = data.shape
    means = np.mean(data,axis=0)
    diffs = data - means
    cov = np.zeros([num_vars,num_vars])
    for v1,v2 in itertools.combinations_with_replacement(range(num_vars),2):
        cov[v1,v2] = (1/float(num_obs-1))*np.sum(diffs[:,v1] * diffs[:,v2])
        cov[v2,v1] = cov[v1,v2]
    return cov

def tf_cov(data):
    '''
    Compute covariance matrix of data in TensorFlow
    '''
    num_obs, num_vars = data.shape
    with tf.Session() as sess:
        dd = tf.placeholder(tf.types.float64,[num_obs,num_vars])
        means = tf.reduce_mean(dd, reduction_indices=[0])
        diffs = tf.sub(dd,means)
        cov = (1/float(num_obs-1)) * tf.matmul(tf.transpose(diffs),diffs)
        ans = sess.run([cov],feed_dict = {dd:data})
    return ans

def tf_maha(data):
    '''
    Compute mahalanobis distance of every point to the mean
    '''
    num_obs, num_vars = data.shape
    with tf.Session() as sess:
        dd = tf.placeholder(tf.types.float64,[num_obs,num_vars])
        means = tf.reduce_mean(dd, reduction_indices=[0])
        diffs = tf.sub(dd,means)
        cov = (1/float(num_obs-1)) * tf.matmul(tf.transpose(diffs),diffs)
        vivi = tf.matrix_inverse(cov)
        dt = tf.transpose(diffs)
        ds = diffs
        M1 = tf.matmul(ds,vivi)
        M2 = tf.matmul(M1,dt)
        z = tf.pack([M2[i,i] for i in range(num_obs)])
        output = tf.sqrt(z)
        ans = sess.run([output],feed_dict = {dd:data})
    return ans


