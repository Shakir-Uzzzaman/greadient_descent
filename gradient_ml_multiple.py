import numpy as np
import copy,math
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train=np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict(x,w,b):
    p = np.dot(x, w) + b     
    return p
x_vec=X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
f_wb=predict(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
def compute_cost(X,y,w,b):
    m=X.shape[0]
    cost=0.0
    for i in range(m):
        f_wb_i=np.dot(X[i],w)+b
        cost=cost+ (f_wb_i-y[i])**2
    cost = cost / (2 * m) 
    return cost
cost=compute_cost(X_train,y_train,w_init,b_init)
print(f'Cost at optimal w : {cost}')

def compute_gradient(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        err=(np.dot(X[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j]=dj_dw[j]+err*X[i,j]
            dj_db=dj_db+err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db
tem_dj_dw,tem_dj_db=compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tem_dj_db}')
print(f'dj_dw at initial w,b: \n {tem_dj_dw}')
def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function,alpha,itt):
    j_history=[]
    w=copy.deepcopy(w_in)
    b=b_in
    for i in range(itt):
        dj_dw,dj_db=gradient_function(X,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        if i<=1000:
            j_history.append(cost_function(X,y,w,b))
        
    return w,b,j_history
init_w=np.zeros_like(w_init)
init_b=0
itt=1000
alpha=5e-7
w_final, b_final, J_hist = gradient_descent(X_train, y_train, init_w, init_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, itt)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
 
            


       
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



        