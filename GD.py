# Importing various packages
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from autograd import grad

n = 100
x = 2*np.random.rand(n,1)
x = np.sort(x, axis=0)
y = 4 + 3*x + 5*x*x + np.random.randn(n,1)

##Own inversion
X = np.c_[np.ones((n,1)), x, x*x]
theta_linreg = np.linalg.inv(X.T@X)@(X.T@y)
print("Own inversion")
print(theta_linreg)
H = (2.0/n)*(X.T@X)


theta = np.random.randn(3,1)
eta = 0.1  #learning rate
print("eta = ", eta)
Niterations = 1000

def CostOLS(y,X,theta):
    return np.sum((y-X @ theta)**2)

def Gradient_Descent(X, y, theta, eta, Niterations):
    for iter in range(Niterations):
        gradients = 2.0/n*X.T @ ((X @ theta)-y)
        theta -= eta*gradients
    return theta

def Gradient_Descent_momentum(X, y, theta, eta, Niterations, change = 0.0, delta_momentum = 0.3):
    for iter in range(Niterations):
        gradients = 2.0/n*X.T @ ((X @ theta)-y)
        new_change = eta*gradients + delta_momentum*change
        theta -= new_change
        change = new_change
    return theta

def Stochastic_Gradient_Descent(X, y, theta, eta, n_epochs, M, m, t0, t1):
    for epoch in range(n_epochs):
        for i in range (m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (2.0/M)*xi.T @ ((xi @ theta)-yi)
            eta = t0/((epoch*m+i)+t1)
            theta = theta - eta*gradients
    return theta

def Stochastic_Gradient_Descent_momentum(X, y, theta, eta, n_epochs, M, m, t0, t1, change = 0.0, delta_momentum = 0.3):
    for epoch in range (n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (2.0/M)*xi.T @ ((xi @ theta)-yi)
            eta = t0/((epoch*m+i)+t1)
            # calculate update
            new_change = eta*gradients+delta_momentum*change
            # take a step
            theta -= new_change
            # save the change
            change = new_change
print("theta from own sdg with momentum")
print(theta)

def GD_AdaGrad(X, y, theta, eta, Niterations, epsilon = 1e-8):
    Giter = 0.0
    training_gradient = grad(CostOLS)
    for iter in range (Niterations):
        gradients = training_gradient(y, X, theta)
        Giter += gradients**2
        update = eta*gradients/(np.sqrt(Giter)+epsilon)
        theta -= update
    return theta

def GDM_AdaGrad(X, y, theta, eta, Niterations, epsilon = 1e-8, change = 0.0, delta_momentum = 0.3):
    Giter = 0.0
    training_gradient = grad(CostOLS)
    for iter in range (Niterations):
        gradients = training_gradient(y, X, theta)
        Giter += gradients**2
        update = eta*gradients/(np.sqrt(Giter)+epsilon)
        new_change = update + delta_momentum*change
        theta -= new_change
        change = new_change
    return theta

def SGD_AdaGrad (X, y, theta, eta, n_epochs, M, m, t0, t1, epsilon = 1e-8):
    Giter = 0.0
    training_gradient = grad(CostOLS)
    for epoch in range (n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (1.0/M)*training_gradient(yi, xi, theta)
            Giter += gradients **2
            eta = t0/t0/((epoch*m+i)+t1)
            update = eta*gradients/(np.sqrt(Giter)+epsilon)
            theta -= update
    return theta

def SGDM_AdaGrad (X, y, theta, eta, n_epochs, M, m, t0, t1, epsilon = 1e-8, change = 0.0, delta_momentum = 0.3):
    Giter = 0.0
    training_gradient = grad(CostOLS)
    for epoch in range (n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (1.0/M)*training_gradient(yi, xi, theta)
            Giter += gradients **2
            eta = t0/t0/((epoch*m+i)+t1)
            update = eta*gradients/(np.sqrt(Giter)+epsilon)
            new_change = update + delta_momentum*change
            theta -= new_change
            change = new_change
    return theta

def GD_RMSProp (X, y, theta, eta, Niterations, epsilon = 1e-6, rho = 0.99)
    Giter = 0.0
    training_gradient = grad(CostOLS)
    for iter in range (Niterations):
        gradients = training_gradient(y, X, theta)
        Giter = rho*Giter + (1-rho)*gradients**2
        update = eta*gradients/(np.sqrt(Giter)+epsilon)
        theta -= update
    return theta

def GDM_RMSProp (X, y, theta, eta, Niterations, epsilon = 1e-6, rho = 0.99, change = 0.0, delta_momentum = 0.3):
    Giter = 0.0
    training_gradient = grad(CostOLS)
    for iter in range (Niterations):
        gradients = training_gradient(y, X, theta)
        Giter = rho*Giter + (1-rho)*gradients**2
        update = eta*gradients/(np.sqrt(Giter)+epsilon)
        new_change = update + delta_momentum*change
        theta -= new_change
        change = new_change
    return theta

def SGD_RMSProp (X, y, theta, eta, n_epochs, M, m, t0, t1, epsilon = 1e-6, rho = 0.99):
    Giter = 0.0
    training_gradient = grad(CostOLS)
    for epoch in range (n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (1.0/M)*training_gradient(yi, xi, theta)
            Giter = rho*Giter + (1-rho)*gradients**2
            eta = t0/((epoch*m+i)+t1)
            update = eta*gradients/(np.sqrt(Giter)+epsilon)
            theta -= update
    return theta

def SGDM_RMSProp (X, y, theta, eta, n_epochs, M, m, t0, t1, epsilon = 1e-6, rho = 0.99, change = 0.0, delta_momentum = 0.3):
    Giter = 0.0
    training_gradient = grad(CostOLS)
    for epoch in range (n_epochs):
        for i in range (m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (1.0/M)*training_gradient(yi, xi, theta)
            Giter += rho*Giter + (1-rho)*gradients**2
            eta = t0/t0/((epoch*m+i)+t1)
            update = eta*gradients/(np.sqrt(Giter)+epsilon)
            new_change = update + delta_momentum*change
            theta -= new_change
            change = new_change
    return theta

def GD_Adam (X, y, theta, Niterations, eta = 0.01, delta = 1e-7, beta1 = 0.9, beta2 = 0.999):
    first_moment = 0.0
    second_moment = 0.0
    training_gradient = grad(CostOLS)
    for iter in range (Niterations):
        gradients = training_gradient(y, X, theta)
        first_moment = beta1*first_moment + (1-beta1)*gradients
        second_moment = beta2*second_moment + (1-beta2)*gradients**2
        first_term = first_moment/(1-beta1**(iter+1))
        second_term = second_moment/(1-beta2**(iter+1))
        update = eta*first_term/(np.sqrt(second_term)+delta)
        theta -= update
    return theta

def GDM_Adam (X, y, theta, Niterations, eta = 0.01, delta = 1e-7, beta1 = 0.9, beta2 = 0.999, change = 0.0, delta_momentum = 0.3):
    first_moment = 0.0
    second_moment = 0.0
    training_gradient = grad(CostOLS)
    for iter in range (Niterations):
        gradients = training_gradient(y, X, theta)
        first_moment = beta1*first_moment + (1-beta1)*gradients
        second_moment = beta2*second_moment + (1-beta2)*gradients**2
        first_term = first_moment/(1-beta1**(iter+1))
        second_term = second_moment/(1-beta2**(iter+1))
        update = eta*first_term/(np.sqrt(second_term)+delta)
        new_change = update + delta_momentum*change
        theta -= new_change
        change = new_change
    return theta
    
def SGD_Adam (X, y, theta, eta, n_epochs, M, m, t0, t1, eta = 0.01, delta = 1e-7, beta1 = 0.9, beta2 = 0.999, change = 0.0, delta_momentum = 0.3):
    first_moment = 0.0
    second_moment = 0.0
    training_gradient = grad(CostOLS)
    for epoch in range (n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (1.0/M)*training_gradient(yi, xi, theta)
            first_moment = beta1*first_moment + (1-beta1)*gradients
            second_moment = beta2*second_moment + (1-beta2)*gradients**2
            first_term = first_moment/(1-beta1**(epoch*m+i+1))
            second_term = second_moment/(1-beta2**(epoch*m+i+1))
            eta = t0/((epoch*m+i)+t1)
            update = eta*first_term/(np.sqrt(second_term)+delta)
            theta -= update
    return theta
            


theta = Gradient_Descent(X, y, theta, eta, Niterations)
thetaGDM = Gradient_Descent_momentum(X, y, theta, eta, Niterations)

xnew = 2*np.random.rand(n,1)
xnew = np.sort(x, axis=0)
Xnew = np.c_[np.ones((n,1)), xnew, xnew*xnew]
ypredict = Xnew.dot(theta)
ypredict2 = Xnew.dot(theta_linreg)

plt.plot(xnew, ypredict, "r-", label = "Own GD")
plt.plot(xnew, ypredict2, "b-", label = "Own inversion")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 30.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Random numbers ')
plt.legend()
plt.show()