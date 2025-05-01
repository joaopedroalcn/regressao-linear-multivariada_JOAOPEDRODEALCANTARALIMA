# Functions/feature_normalize.py
"""
@file features_normalizes.py
@brief Funções para normalização de features em datasets.
@details Este módulo contém funções para normalizar as features de um dataset
          utilizando diferentes abordagens, como média e desvio padrão, ou
          mínimo e máximo.
@author Joao Pedro de Alcântara Lima
e-mail: joao.alcantara@discente.ufma.br
"""
import numpy as np

def features_normalize_by_std(X):
    """
    Normaliza as features de um dataset para média zero e desvio padrão unitário.
    Matematicamente, a fórmula utilizada é:
        X_norm = (X - mu) / sigma
    """
    # Calcula a média de cada feature (coluna)
    mu = np.mean(X, axis=0)

    # Calcula o desvio padrão de cada feature (coluna)
    sigma = np.std(X, axis=0)

    # Evita divisão por zero
    sigma[sigma == 0] = 1

    # Normaliza as features
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def features_normalizes_by_min_max(X):
    """
    Normaliza as features de um dataset para o intervalo [0, 1].
    Matematicamente, a fórmula utilizada é:
        X_norm = (X - min) / (max - min)
    """
    # Calcula o mínimo de cada feature (coluna)
    min_val = np.min(X, axis=0)

    # Calcula o máximo de cada feature (coluna)
    max_val = np.max(X, axis=0)

    # Calcula o denominador e evita divisão por zero
    denom = max_val - min_val
    denom[denom == 0] = 1

    # Normaliza as features
    X_norm = (X - min_val) / denom

    return X_norm, min_val, max_val
