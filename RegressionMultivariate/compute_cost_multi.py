# Functions/compute_cost_multi.py
"""
@file compute_cost_multi.py
@brief Computes the cost for multivariate linear regression.
@details Este módulo contém uma função para calcular o custo de um modelo de regressão linear
          multivariada utilizando a função de custo de erro quadrático médio.
@author Joao Pedro de Alcântara Lima
e-mail: joao.alcantara@discente.ufma.br
"""

import numpy as np

def compute_cost_multi(X, y, theta):
    """
    Calcula o custo para regressão linear multivariada.

    A função de custo é definida como:
        J(θ) = (1 / (2m)) * (Xθ - y)ᵀ (Xθ - y)

    :param (ndarray) X: Matriz de features incluindo o termo de intercepto (shape: m × n+1).
    :param (ndarray) y: Vetor de valores alvo (shape: m,).
    :param (ndarray) theta: Vetor de parâmetros (shape: n+1,).
    :return (float): Valor do custo calculado.
    """
    # Número de exemplos de treinamento
    m = y.shape[0]

    # Predições do modelo linear: h(θ) = X @ θ
    predictions = X @ theta

    # Erros entre as predições e os valores reais
    errors = predictions - y

    # Cálculo do custo J(θ)
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)

    return cost
