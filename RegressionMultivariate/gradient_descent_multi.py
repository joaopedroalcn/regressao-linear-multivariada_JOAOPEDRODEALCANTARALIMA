# Functions/gradient_descent_multi.py
"""
@file gradient_descent_multi.py
@brief Performs gradient descent for multivariate regression.
@details Este módulo contém uma função para executar o gradiente descendente
          para regressão linear multivariada, atualizando os parâmetros θ
          iterativamente para minimizar a função de custo.
@author Joao Pedro de Alcântara Lima
e-mail: joao.alcantara@discente.ufma.br
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from compute_cost_multi import compute_cost_multi

def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        error = X @ theta - y
        gradient = (1 / m) * (X.T @ error)
        theta = theta - alpha * gradient
        J_history[i] = compute_cost_multi(X, y, theta)

    return theta, J_history

def gradient_descent_multi_with_history(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    n = theta.shape[0]

    J_history = np.zeros(num_iters)
    theta_history = np.zeros((num_iters + 1, n))
    theta_history[0] = theta.copy()

    for i in range(num_iters):
        error = X @ theta - y
        gradient = (1 / m) * (X.T @ error)
        theta = theta - alpha * gradient
        J_history[i] = compute_cost_multi(X, y, theta)
        theta_history[i + 1] = theta.copy()

    return theta, J_history, theta_history
