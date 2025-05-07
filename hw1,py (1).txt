import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 設定區間和網格
pi = np.pi
h = 0.1 * pi
Nx = int(pi / h) + 1
Ny = int((pi / 2) / h) + 1

x = np.linspace(0, pi, Nx)
y = np.linspace(0, pi/2, Ny)

u = np.zeros((Nx, Ny))

# 邊界條件
for j in range(Ny):
    u[0, j] = np.cos(y[j])          # u(0, y)
    u[-1, j] = -np.cos(y[j])        # u(pi, y)
for i in range(Nx):
    u[i, 0] = np.cos(x[i])          # u(x, 0)
    u[i, -1] = 0                    # u(x, pi/2)

# 右手邊 f(x,y) = x*y
X, Y = np.meshgrid(x, y, indexing='ij')
f = X * Y

# Jacobi 迭代
u_new = u.copy()
for iteration in range(1000):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u_new[i, j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - h**2 * f[i,j])
    if np.linalg.norm(u_new - u, ord=np.inf) < 1e-4:
        break
    u[:] = u_new

# 印出數值（四捨五入）
u_rounded = np.round(u, 2)
print("Numerical solution u(x,y):")
print(u_rounded)

# 存成 CSV
df = pd.DataFrame(u_rounded, index=[f"x={round(val,2)}" for val in x], columns=[f"y={round(val,2)}" for val in y])
df.to_csv("hw1_u_values.csv")

# 繪圖並存檔
plt.figure(figsize=(8,5))
contour = plt.contourf(x, y, u.T, 20, cmap='viridis')
plt.colorbar(label='u(x, y)')
plt.title('hw1_Solution of Poisson equation')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("hw1_u_contour.png", dpi=300)
plt.show()
