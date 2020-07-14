import numpy as np
import  matplotlib.pyplot as plt

plot_x = np.linspace(-1, 6, 141)    # 损失直线的141个点
plot_y = (plot_x-2.5) **2 -1

def dj(theta):
    return 2*(theta - 2.5)

def J(theta):
    try:
        return (theta - 2.5) ** 2 - 1
    except:
        return float('inf')

def gradient_descent(initial_theta, eta, n_iters= 1e4, epsilon=1e-8):
    theta = initial_theta
    theta_history.append(initial_theta)
    i_iter = 0

    '''最多执行n_iters次数'''
    while i_iter < n_iters:
        gradient = dj(theta)
        last_theta = theta
        theta = theta - gradient * eta
        theta_history.append(theta)
        if abs(J(theta) - J(last_theta)) < epsilon:
            break
        i_iter += 1

def plot_theta_history():
    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color='red', marker='+')
    plt.show()

theta_history = []
eta = 0.01
gradient_descent(0, eta)
plot_theta_history()
print(len(theta_history))
