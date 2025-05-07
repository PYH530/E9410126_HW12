import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def setup():
    """參數設定與初始化"""
    Δt = 0.01  # Reduced time step for stability
    Δr = 0.1
    K = 0.1
    t_end = 10  # Total time to simulate

    # 穩定性檢查
    if Δt > (Δr**2) / (2*K):
        print(f"警告：Δt={Δt} 可能過大，可能導致不穩定。建議 Δt < {(Δr**2)/(2*K):.4f}")

    r = np.arange(0.5, 1.0 + Δr, Δr)
    t = np.arange(0, t_end + Δt, Δt)  # Increased number of time steps
    nr, nt = len(r), len(t)
    T = np.zeros((nr, nt))

    # 初始條件
    T[:, 0] = 200 * (r - 0.5)

    # 邊界條件 (T(1,t) = 100 + 40t)
    for n in range(nt):
        T[-1, n] = 100 + 40 * min(t[n], 10)  # 限制t≤10

    return r, t, T, Δr, Δt, K

def forward_method():
    """前向差分法（顯式）"""

    r, t, T, dr, dt, K = setup()
    nt = len(t)
    nr = len(r)

    for n in range(nt - 1):
        for i in range(1, nr - 1):
            d2T = (T[i+1, n] - 2*T[i, n] + T[i-1, n]) / dr**2
            dTdr = (T[i+1, n] - T[i-1, n]) / (2*dr)
            T[i, n+1] = T[i, n] + dt * (K * (d2T + (1/r[i]) * dTdr))

        # Neumann邊界條件 (dT/dr + 3T = 0 at r=0.5)
        T[0, n+1] = T[1, n+1] / (1 + 3*dr)

    return r, t, T

def backward_method():
    """後向差分法（隱式）"""

    r, t, T, dr, dt, K = setup()
    nt = len(t)
    nr = len(r)
    α = K * dt / dr**2

    for n in range(nt - 1):
        A = np.zeros((nr, nr))
        b = np.zeros(nr)

        for i in range(1, nr - 1):
            ri = r[i]
            A[i, i-1] = -α + (K * dt) / (2 * dr * ri)
            A[i, i] = 1 + 2*α
            A[i, i+1] = -α - (K * dt) / (2 * dr * ri)
            b[i] = T[i, n]

        # Neumann邊界條件
        A[0, 0] = 1 + 3*dr
        A[0, 1] = -1
        b[0] = 0

        # Dirichlet邊界條件
        A[-1, -1] = 1
        b[-1] = 100 + 40 * min(t[n+1], 10)

        T[:, n+1] = solve(A, b)

    return r, t, T

def crank_nicolson():
    """Crank-Nicolson方法"""

    r, t, T, dr, dt, K = setup()
    nt = len(t)
    nr = len(r)
    α = K * dt / (2 * dr**2)

    for n in range(nt - 1):
        A = np.zeros((nr, nr))
        b = np.zeros(nr)

        for i in range(1, nr - 1):
            ri = r[i]
            A[i, i-1] = -α + (K * dt) / (4 * dr * ri)
            A[i, i] = 1 + 2*α
            A[i, i+1] = -α - (K * dt) / (4 * dr * ri)

            b[i] = α * (T[i+1, n] - 2*T[i, n] + T[i-1, n]) + T[i, n] + \
                   (K * dt / (4 * dr * ri)) * (T[i+1, n] - T[i-1, n])

        # Neumann邊界條件
        A[0, 0] = 1 + 3*dr
        A[0, 1] = -1
        b[0] = 0

        # Dirichlet邊界條件
        A[-1, -1] = 1
        b[-1] = 100 + 40 * min(t[n+1], 10)

        T[:, n+1] = solve(A, b)

    return r, t, T

def plot_comparison():
    """三種方法對比可視化"""

    methods = {
        'Forward Difference': forward_method,
        'Backward Difference': backward_method,
        'Crank-Nicolson': crank_nicolson
    }

    plt.figure(figsize=(18, 6))
    results = {}

    for idx, (name, method) in enumerate(methods.items(), 1):
        r, t, T = method()
        results[name] = (r, t, T)

        plt.subplot(1, 3, idx)
        contour = plt.contourf(t, r, T, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Temperature')
        plt.title(name)
        plt.xlabel('Time (t)')
        plt.ylabel('Radius (r)')

    plt.tight_layout()
    plt.savefig('hw2_method_comparison.png', dpi=300)
    plt.show()

    # 動態可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    for name in methods.keys():
        line, = ax.plot([], [], label=name)
        lines.append(line)

    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 300)
    ax.set_xlabel('Radius (r)')
    ax.set_ylabel('Temperature (T)')
    ax.set_title('hw2_Temperature Distribution Evolution')
    ax.legend()
    ax.grid()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for line, (name, (r, t, T)) in zip(lines, results.items()):
            line.set_data(r, T[:, frame])
        ax.set_title(f'Temperature at t = {t[frame]:.1f}s')
        return lines

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)

    # 儲存為GIF
    try:
        ani.save('hw2_temperature_evolution.gif', writer='pillow', fps=10)
        print("動畫已儲存為 temperature_evolution.gif")
    except ImportError:
        print("請安裝 pillow 套件以儲存動畫：pip install pillow")

    return results

if __name__ == "__main__":
    results = plot_comparison()
    print("計算完成！結果已儲存為 method_comparison.png 和 temperature_evolution.gif")