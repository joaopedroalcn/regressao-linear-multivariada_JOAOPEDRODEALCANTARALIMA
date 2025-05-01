# Functions/normal_eqn.py
"""
@file normal_eqn.py
@brief Calcula os parâmetros θ usando a Equação Normal.
@details Este módulo contém uma função para calcular os parâmetros de um modelo
          de regressão linear utilizando a equação normal.
@author Joao Pedro de Alcântara Lima
e-mail: joao.alcantara@discente.ufma.br
"""

import numpy as np


def normal_eqn(X, y):
    """
    Resolve os parâmetros θ utilizando a equação normal.

    A equação normal é definida como:
        θ = (XᵀX)⁻¹ Xᵀ y

    :param (ndarray) X: Matriz de features com bias, onde cada linha é uma amostra
                        e cada coluna é uma feature (shape: m × n+1).
    :param (ndarray) y: Vetor de valores alvo (shape: m,).
    :return (ndarray): Vetor de parâmetros θ (shape: n+1,).
    """
    # Calcula os parâmetros θ utilizando a equação normal
    # A pseudo-inversa de XᵀX pode ser calculada com np.linalg.pinv
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta
