from firedrake import *
import numpy as np

# 1. 定义网格和空间
mesh = UnitIntervalMesh(40)
V = FunctionSpace(mesh, "CG", 1)

# 2. 定义初值（优化变量）
q0 = Function(V, name="Initial Condition")
q0.assign(0.0)  # 初始猜测

# 3. 定义模型参数
dt = 0.01
T = 1.0
nt = int(T/dt)
u = Constant(1.0)  # 速度

# 4. 观测数据（假设已知）
obs_times = [0.2, 0.5, 0.8]
obs_data = [...]  # 观测值

# 5. 定义目标函数
def objective(q0):
    q = Function(V)
    q.assign(q0)
    J = 0.0
    t = 0.0
    for n in range(nt):
        # 前向推进一步（如Forward Euler）
        v = TestFunction(V)
        F = (q - q0)/dt*v*dx + u*dot(grad(q), v)*dx
        solve(F == 0, q)
        t += dt
        if np.isclose(t, obs_times, atol=dt/2).any():
            # 观测算子H为插值或投影
            J += assemble((q - obs_data[n])**2 * dx)
    return J

# 6. 自动微分与优化
from firedrake_adjoint import *
J = objective(q0)
control = Control(q0)
reduced_functional = ReducedFunctional(J, control)
opt_q0 = minimize(reduced_functional)