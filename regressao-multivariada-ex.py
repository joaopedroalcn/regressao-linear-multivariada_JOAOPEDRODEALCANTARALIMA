# regressao-multivariada-ex.py
"""
@file regressao-multivariada-ex.py
@brief Multivariate linear regression exercise with gradient descent and normal equation.
@details Este script executa um fluxo de trabalho completo para regressão linear multivariada,
          incluindo normalização de features, cálculo de parâmetros via gradiente descendente
          e equação normal, além de comparação de custos.
@author Joao Pedro de Alcântara Lima
e-mail: joao.alcantara@discente.ufma.br
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from RegressionMultivariate.features_normalize import features_normalize_by_std
from RegressionMultivariate.features_normalize import features_normalizes_by_min_max
from RegressionMultivariate.compute_cost_multi import compute_cost_multi
from RegressionMultivariate.gradient_descent_multi import gradient_descent_multi
from RegressionMultivariate.gradient_descent_multi import gradient_descent_multi_with_history
from RegressionMultivariate.normal_eqn import normal_eqn

def costs_from_history(X_b: np.ndarray, y: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Calcula o custo J(θ) para cada θ em *thetas*."""
    return np.array([compute_cost_multi(X_b, y, th) for th in thetas])

def main():
    # 1) Cria pasta de figuras
    os.makedirs("Figures", exist_ok=True)

    # 2) Carrega dados
    data = np.loadtxt('Data/ex1data2.txt', delimiter=',')
    X = data[:, :2]  # Primeiras duas colunas são features
    y = data[:, 2]   # Terceira coluna é o target
    m = len(y)       # Número de exemplos de treinamento

    print('Primeiros 10 exemplos de treinamento:')
    print(np.column_stack((X[:10], y[:10])))

    
    # 3) Normaliza features
    X_norm, mu, sigma = features_normalize_by_std(X)
    X_b = np.column_stack((np.ones(m), X_norm))  # Adiciona coluna de bias
    
    print('\nParâmetros de normalização:')
    print(f'Média (mu): {mu}')
    print(f'Desvio Padrão (sigma): {sigma}')

    # 4) Gradient Descent Multivariado
    alpha = 0.01
    num_iters = 400
    theta_gd = np.zeros(X_b.shape[1])  # Inicializa theta com zeros
    
    theta_gd, J_history = gradient_descent_multi(
        X_b, y, theta_gd, alpha, num_iters
    )
    
    print('\nTheta via Gradient Descent:')
    print(theta_gd)

    # 4a) Plot de convergência (GD)
    plt.figure()
    plt.plot(np.arange(1, num_iters + 1), J_history, 'b-', linewidth=2)
    plt.xlabel('Iteração')
    plt.ylabel('Custo J(θ)')
    plt.title('Convergência do Gradiente (Multivariada)')
    plt.grid(True)
    plt.savefig('Figures/convergencia_custo_multi.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figures/convergencia_custo_multi.svg', format='svg', bbox_inches='tight')
    plt.show()

    # 5) Predição com GD
    example = np.array([1650, 3])  # features originais
    example_norm = (example - mu) / sigma  # normaliza
    x_pred = np.concatenate(([1], example_norm))  # adiciona bias
    price_gd = x_pred @ theta_gd  # faz a predição
    print(f'\nPreço previsto (GD) para [1650,3]: ${price_gd:.2f}')

    # 6) Equação Normal
    X_ne = np.column_stack((np.ones(m), X))  # X original com bias
    theta_ne = normal_eqn(X_ne, y)
    
    example = np.array([1, 1650, 3])  # features originais com bias
    price_ne = example @ theta_ne  # faz a predição
    print('\nTheta via Equação Normal:')
    print(theta_ne)
    print(f'Preço previsto (NE) para [1650,3]: ${price_ne:.2f}')

    # --- Comparação de custos ---
    cost_ne_errado = compute_cost_multi(X_b, y, theta_ne)
    print(f'\n[CUSTO ERRADO] Custo usando θ_ne em X_NORMALIZADO (X_b): {cost_ne_errado:.2f}')

    cost_ne_correto = compute_cost_multi(X_ne, y, theta_ne)
    print(f'[CUSTO CORRETO] Custo usando θ_ne em X_ORIGINAL (X_ne): {cost_ne_correto:.2f}')

    plt.figure()
    plt.plot(np.arange(1, num_iters + 1), J_history,
             'b-', label='Gradiente Descendente')
    plt.hlines(cost_ne_correto, 1, num_iters,
               colors='r', linestyles='--',
               label='Equação Normal (correto)')
    plt.hlines(cost_ne_errado, 1, num_iters, colors='k', linestyles=':', label='NE (errado)')
    plt.xlabel('Iteração')
    plt.ylabel('Custo J(θ)')
    plt.title('GD vs Normal Equation')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figures/convergencia_custo_vs_ne.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figures/convergencia_custo_vs_ne.svg', format='svg', bbox_inches='tight')
    plt.show()

    # -------------- Visualizações 3D / Contorno para multivariada ----------------------------
    theta_gd, J_history, theta_history = gradient_descent_multi_with_history(
        X_b, y, np.zeros(X_b.shape[1]), alpha, num_iters
    )
    
    # Normaliza theta_ne para mesma escala que theta_gd
    theta_ne_norm = np.zeros_like(theta_ne)
    theta_ne_norm[0] = theta_ne[0] + np.sum((mu / sigma) * theta_ne[1:])
    theta_ne_norm[1:] = theta_ne[1:] * sigma

    # 7) Contorno J(θ1, θ2) (θ0 fixo em θ_gd[0]). Malha de custo centrada no ótimo
    t1_hist, t2_hist = theta_history[:, 1], theta_history[:, 2]
    max_dev1 = np.max(np.abs(t1_hist - theta_ne_norm[1]))
    max_dev2 = np.max(np.abs(t2_hist - theta_ne_norm[2]))
    span1 = span2 = 1.5 * max(max_dev1, max_dev2)  # mesma escala nos 2 eixos

    t1_vals = np.linspace(theta_ne_norm[1] - span1, theta_ne_norm[1] + span1, 120)
    t2_vals = np.linspace(theta_ne_norm[2] - span2, theta_ne_norm[2] + span2, 120)
    T1, T2 = np.meshgrid(t1_vals, t2_vals)

    J_mesh = np.zeros_like(T1)
    for i in range(T1.shape[0]):
        for j in range(T1.shape[1]):
            J_mesh[i, j] = compute_cost_multi(X_b, y, [theta_ne_norm[0], T1[i, j], T2[i, j]])

    # 8) Superfície J(θ1, θ2) + trajetória GD + NE (normalizado)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T1, T2, J_mesh, cmap="viridis", alpha=0.85, linewidth=0)
    ax.plot(t1_hist, t2_hist, costs_from_history(X_b, y, theta_history), "r.-", label="Trajetória GD")
    ax.scatter(theta_ne_norm[1], theta_ne_norm[2], compute_cost_multi(X_b, y, theta_ne_norm),
               s=80, marker="x", color="black", linewidths=2, label="NE (norm)")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Custo J(θ)")
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$"); ax.set_zlabel("Custo J(θ)")
    ax.set_title("Superfície J(θ1, θ2)")
    ax.view_init(elev=30, azim=-60)
    ax.legend()
    fig.savefig("Figures/superficie_GD_vs_NE.png", dpi=300)

    # 8a) Contorno J(θ1, θ2) + trajetória GD + NE (normalizado)
    from matplotlib.colors import LogNorm
    plt.figure(figsize=(7, 5))
    levels = np.logspace(np.log10(J_mesh.min()), np.log10(J_mesh.max()), 60)
    cf = plt.contourf(T1, T2, J_mesh, levels=levels, norm=LogNorm(), cmap="viridis")
    plt.colorbar(cf, label="Custo J(θ)")
    plt.plot(t1_hist, t2_hist, "r.-", ms=2, label="Trajetória GD")
    plt.scatter(theta_ne_norm[1], theta_ne_norm[2], s=80, marker="x", color="black", label="NE (norm)")
    plt.xlabel(r"$\theta_1$"); plt.ylabel(r"$\theta_2$")
    plt.title("Contorno J(θ1, θ2)"); plt.legend()
    plt.savefig("Figures/contorno_GD_vs_NE.png", dpi=300)
    plt.show()

    # 9) Plano de regressão ajustado + pontos originais (3‑D)
    fig2 = plt.figure(figsize=(7, 5))
    ax2 = fig2.add_subplot(111, projection="3d")

    # Pontos originais
    ax2.scatter(X[:, 0], X[:, 1], y, c="red", marker="x", label="Dados de treino")

    # Plano de regressão
    f1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 40)
    f2_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 40)
    F1, F2 = np.meshgrid(f1_vals, f2_vals)

    # Converte θ_gd (normalizado) para escala original
    theta_gd_orig = np.zeros_like(theta_ne)
    theta_gd_orig[1:] = theta_gd[1:] / sigma
    theta_gd_orig[0] = theta_gd[0] - np.sum((mu / sigma) * theta_gd[1:])

    Z = theta_gd_orig[0] + theta_gd_orig[1] * F1 + theta_gd_orig[2] * F2
    surf2 = ax2.plot_surface(
        F1, F2, Z, alpha=0.5, cmap="viridis", rstride=1, cstride=1
    )

    ax2.set_xlabel("Tamanho (pés²)")
    ax2.set_ylabel("Quartos")
    ax2.set_zlabel("Preço (US$)")
    ax2.set_title("Ajuste da Regressão Linear Multivariada")
    ax2.view_init(elev=25, azim=-135)
    
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color="red", marker="x", linestyle="", label="Dados de treino"),
        Patch(facecolor=surf2.get_facecolor()[0], edgecolor="none", alpha=0.5, label="Plano GD"),
    ]
    ax2.legend(handles=handles)
    fig2.tight_layout()
    fig2.savefig("Figures/ajuste_regressao_multivariada.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()