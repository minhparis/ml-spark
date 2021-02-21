import numpy as np
from numpy.linalg import norm
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn import datasets
import time

def grad(x, w):
  """Calculation of gradient
    Args:
        x: dataset
        w: weight vector
    Returns:
        gradient: gradient of dataset with the weight vector
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  return  2*(train_x.dot(w)-train_y).dot(train_x)

def sgd(x, w, nb_iter_sgd, lr):
  """Stochastic gradient descent
    Args:
        x: dataset
        w: weight vector
        nb_iter_sgd: number of iterations for SGD
        lr: steep
    Returns:
        wsub: weight optimized with SGD
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  N = train_x.shape[0]
  wsub = w
  for i in range(nb_iter_sgd):
    np.random.seed(int(time.time())) 
    random_idx_list = np.random.permutation(N)
    for j in range(N):
      idx = random_idx_list[j]
      X = train_x[idx]
      y = train_y[idx]
      grads = 2*(sum(X*w)-y)*X
      wsub -= lr * grads / norm(grads)
  return wsub

def batch_sgd(x, w, nb_iter_sgd, lr, batch_size):
    """Batch Stochastic gradient descent
    Args:
        x: dataset
        w: weight vector
        nb_iter_sgd: number of iterations for SGD
        lr: steep
        batch_size: number of batch
    Returns:
        wsub: weight optimized with SGD
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  N = train_x.shape[0]
  inds = list(range(N))
  wsub = w
  for i in range(nb_iter_sgd):
    np.random.seed(int(time.time())) 
    random_idx_list = np.random.permutation(N)
    
    batch_indices = [inds[i:(i+batch_size)] for i in range(0, len(inds), batch_size)]
    for indices in batch_indices:
      grads_sum = 0
      
      for j in indices:
        idx = random_idx_list[j]
        X = train_x[idx]
        y = train_y[idx]
        grads_sum += 2*(sum(X*w)-y)*X
      
      wsub -= lr * grads_sum / (norm(grads_sum)*batch_size)
  return wsub
  
def MSE(w, x, y):
    """Calculation of loss function (Mean squared error)
    Args:
        x: dataset
        w: weight vector
        y: label of dataset
    Returns:
        loss value
    """
  return sum((x.dot(w)-y)**2)/len(x)

def sgd_mom(x, w, nb_iter_sgd, lr, gamma):
  """SGD moment
    Args:
        x: dataset
        w: weight vector
        nb_iter_sgd: number of iterations for SGD
        lr: steep
        gamma: <1, gamma of moment
    Returns:
        wsub: weight optimized with SGD moment
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  N = train_x.shape[0]
  wsub = w
  v_t = 0
  for i in range(nb_iter_sgd):
    np.random.seed(int(time.time())) 
    random_idx_list = np.random.permutation(N)
    for j in range(N):
      idx = random_idx_list[j]
      X = train_x[idx]
      y = train_y[idx]
      
      grads = 2*(sum(X*wsub)-y)*X
      # update momentum
      v_t = gamma*v_t + lr*grads/ norm(grads)
      wsub -= v_t
  return wsub

def nesterov(x, w, nb_iter_sgd, lr, gamma):
  """Nesterov
    Args:
        x: dataset
        w: weight vector
        nb_iter_sgd: number of iterations for SGD
        lr: steep
        gamma: <1
    Returns:
        wsub: weight optimized with Nesterov
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  N = train_x.shape[0]
  wsub = w
  v_t = 0
  for i in range(nb_iter_sgd):
    np.random.seed(int(time.time())) 
    random_idx_list = np.random.permutation(N)
    for j in range(N):
      idx = random_idx_list[j]
      X = train_x[idx]
      y = train_y[idx]
      
      w_t = wsub - gamma*v_t
      grads = 2*(sum(X*w_t)-y)*X
      # update momentum
      v_t = gamma*v_t + lr*grads/ norm(grads)
      
      wsub -= v_t
  return wsub

def adagrad(x, w, nb_iter_sgd, lr):
  """Adagrad optimization
    Args:
        x: dataset
        w: weight vector
        nb_iter_sgd: number of iterations for SGD
        lr: steep
    Returns:
        wsub: weight optimized with Adagrad
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  N = train_x.shape[0]
  wsub = w
  grad_squared = 0
  for i in range(nb_iter_sgd):
    np.random.seed(int(time.time())) 
    random_idx_list = np.random.permutation(N)
    for j in range(N):
      idx = random_idx_list[j]
      X = train_x[idx]
      y = train_y[idx]
      
      grads = 2*(sum(X*wsub)-y)*X
      grads = grads/np.linalg.norm(grads)
      grad_squared += grads * grads
      
      wsub -= (lr / np.sqrt(grad_squared)) * grads
  return wsub

def adadelta(x, w, nb_iter_sgd, lr, gamma, eps):
  """Adadelta
    Args:
        x: dataset
        w: weight vector
        nb_iter_sgd: number of iterations for SGD
        lr: steep
        gamma: <1
        eps: eps coeff
    Returns:
        wsub: weight optimized with Adadelta
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  N = train_x.shape[0]
  wsub = w
  g_t = 0
  e_d_theta2_0 = 0
  e_d_theta2_k = 0
  eg2t_0 = 0
  eg2t_k = 0
  for i in range(nb_iter_sgd):
    np.random.seed(int(time.time())) 
    random_idx_list = np.random.permutation(N)
    for j in range(N):
      idx = random_idx_list[j]
      X = train_x[idx]
      y = train_y[idx]
      
      eg2t_0 = eg2t_k
      e_d_theta2_0 = e_d_theta2_k
      
      g_t = 2*(sum(X*wsub)-y)*X
      g_t = g_t/np.linalg.norm(g_t)
      eg2t_k = gamma*eg2t_0 + (1-gamma)*(g_t**2) #equation 10
      
      d_theta_t = - lr * g_t / ((eg2t_k+eps)**0.5) #equation 14
      e_d_theta2_k = (gamma*e_d_theta2_0 + (1-gamma)*(d_theta_t**2) ) #equation 15
      
      delta_w = -((e_d_theta2_0+eps)**0.5/(eg2t_k+eps)**0.5)*g_t #equation 17
      
      wsub += delta_w
      
  return wsub


def rmsprop(x, w, nb_iter_sgd, lr):
  """RMSProp Calculation
    Args:
        x: dataset
        w: weight vector
        nb_iter_sgd: number of iterations for SGD
        lr: steep
    Returns:
        wsub: weight optimized with RMSProp
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  N = train_x.shape[0]
  wsub = w
  grad_squared = 0
  for i in range(nb_iter_sgd):
    np.random.seed(int(time.time())) 
    random_idx_list = np.random.permutation(N)
    for j in range(N):
      idx = random_idx_list[j]
      X = train_x[idx]
      y = train_y[idx]
      
      grads = 2*(sum(X*wsub)-y)*X
      grads = grads/np.linalg.norm(grads)
      grad_squared = 0.9 * grad_squared + 0.1 * grads * grads
      
      wsub -= (lr / np.sqrt(grad_squared)) * grads
  return wsub

def adam(x, w, nb_iter_sgd, lr, b1, b2, eps):
  """Adam optmization
    Args:
        x: dataset
        w: weight vector
        nb_iter_sgd: number of iterations for SGD
        lr: steep
        b1: 1st coeff of beta
        b2: 2nd coeff of beta
        eps: eps coeff
    Returns:
        wsub: weight optimized with Adam
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  N = train_x.shape[0]
  wsub = w
  v_t = 0
  m_t = 0
  for i in range(nb_iter_sgd):
    np.random.seed(int(time.time())) 
    random_idx_list = np.random.permutation(N)
    for j in range(N):
      idx = random_idx_list[j]
      X = train_x[idx]
      y = train_y[idx]
      
      g_t = 2*(sum(X*wsub)-y)*X
      g_t = g_t/np.linalg.norm(g_t)
      m_t = b1*m_t + (1- b1)*g_t
      v_t = b2*v_t + (1-b2)*g_t*g_t
      
      mc_t = m_t/(1-b1**(j+1)) #j+1 = nombre d'iteraions t
      vc_t = v_t/(1-b2**(j+1))
      
      wsub -= lr * mc_t / ((vc_t**0.5) + eps)
  return wsub

def adamax(x, w, nb_iter_sgd, lr, b1, b2):
  """Adamax optmization
    Args:
        x: dataset
        w: weight vector
        nb_iter_sgd: number of iterations for SGD
        lr: steep
        b1: 1st coeff of beta
        b2: 2nd coeff of beta
    Returns:
        wsub: weight optimized with Adam
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  N = train_x.shape[0]
  wsub = w
  v_t = 0
  m_t = 0
  for i in range(nb_iter_sgd):
    np.random.seed(int(time.time())) 
    random_idx_list = np.random.permutation(N)
    for j in range(N):
      idx = random_idx_list[j]
      X = train_x[idx]
      y = train_y[idx]
      
      g_t = 2*(sum(X*wsub)-y)*X
      g_t = g_t/np.linalg.norm(g_t)
      
      u_t = max(LA.norm(b2*v_t),LA.norm(g_t))
      v_t = b2*v_t + (1-b2)*g_t*g_t
      
      m_t = b1*m_t + (1- b1)*g_t
      mc_t = m_t/(1-b1**(j+1))
  
      wsub -= lr * mc_t / u_t
  return wsub

def nadam(x, w, nb_iter_sgd, lr, b1, b2, eps):
  """Nadam optmization
    Args:
        x: dataset
        w: weight vector
        nb_iter_sgd: number of iterations for SGD
        lr: steep
        b1: 1st coeff of beta
        b2: 2nd coeff of beta
        eps: eps coeff
    Returns:
        wsub: weight optimized with Adam
    """
  x = np.array(x)
  train_x = db.data[:, :2]
  train_y = db.target
  N = train_x.shape[0]
  wsub = w
  v_t = 0
  m_t = 0
  for i in range(nb_iter_sgd):
    np.random.seed(int(time.time())) 
    random_idx_list = np.random.permutation(N)
    for j in range(N):
      idx = random_idx_list[j]
      X = train_x[idx]
      y = train_y[idx]
      
      g_t = 2*(sum(X*wsub)-y)*X
      g_t = g_t/np.linalg.norm(g_t)
      
      u_t = max(LA.norm(b2*v_t),LA.norm(g_t))
      v_t = b2*v_t + (1-b2)*g_t*g_t
      vc_t = v_t/(1-b2**(j+1))
      
      m_t = b1*m_t + (1- b1)*g_t
      mc_t = m_t/(1-b1**(j+1))
  
      wsub -= (lr/(vc_t**0.5 + eps))*(b1*mc_t + (1-b1)*g_t/(1-b1**(j+1)))
  return wsub

##################
###LOAD DATASET###
##################

db = datasets.load_iris()
#db = datasets.fetch_california_housing()

#########################
###SPARK CONFIGURATION###
#########################
spark = SparkSession \
    .builder \
    .appName("Python Spark create RDD example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
sc = spark.sparkContext
df = sc.parallelize(db).cache().glom()

#################################
###INITIZATION OF OPTIMIZATION###
#################################
w =  np.ones(2)
nb_iter = 100
nb_iter_sgd = 5
lr = 0.002

t1 = time.time()
MSE_list = []

train_x = db.data[:, :2]
train_y = db.target

#Mini batch SGD
batch_size = 50

#SGD Moment parameters
gamma = 0.01

#Adam parameters
b1 = 0.9
b2 = 0.999
eps = 10**-8
  

##############
###TRAINING###
##############

for i in range(nb_iter):
    w_v =  sc.broadcast(w)

    #Calculation of weight vector by map reduce
    w = df.map(lambda x: batch_sgd(x, w, nb_iter_sgd, lr, batch_size)).mean() 

    #Calculation of loss
    MSE_list.append(MSE(w,train_x,train_y))

    #Print result
    if i%10 == 9:
        print(MSE(w,train_x,train_y))
        print("time  : ", time.time() - t1)
        t1 = time.time()


#################
###SHOW RESULT###
#################
plt.plot(MSE_list)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.title('Batch SGD - Small dataset')
plt.show()