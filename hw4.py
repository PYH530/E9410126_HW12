import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def solve_heat_equation():
    L = 1.0
    T_max = 1.0
    dx = 0.1
    dt = 0.005  # 調整時間步長以滿足穩定性條件

    Nx = int(L / dx) + 1
    Nt = int(T_max / dt) + 1

    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T_max, Nt)

    alpha = dt / dx**2  # 應滿足 alpha <= 0.5

    p = np.zeros((Nt, Nx))

    # 初始條件
    p[0, :] = np.cos(2 * np.pi * x)
    dpdt0 = 2 * np.pi * np.sin(2 * np.pi * x)
    p[1, 1:-1] = p[0, 1:-1] + dt * dpdt0[1:-1]

    # 邊界條件
    p[:, 0] = 1
    p[:, -1] = 2

    # 時間迴圈
    for n in range(1, Nt - 1):
        for i in range(1, Nx - 1):
            p[n + 1, i] = p[n, i] + alpha * (p[n, i + 1] - 2 * p[n, i] + p[n, i - 1])

    # 儲存數據
    df = pd.DataFrame(p, columns=[f"x={xval:.2f}" for xval in x])
    df.insert(0, "time", t)
    df.to_csv("hw4_heat_equation_result.csv", index=False)

    # 繪圖
    X, T = np.meshgrid(x, t)
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, T, p, levels=20, cmap='plasma')
    plt.colorbar(cp)
    plt.title('hw4_Heat Equation: p(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.savefig("hw4_heat_equation_result.png", dpi=300)
    plt.show()

solve_heat_equation()
