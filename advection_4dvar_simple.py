from firedrake import *
from firedrake_adjoint import *
import numpy as np

# 设置参数
nx = 50   # 空间网格数
nt = 20   # 时间步数
L = 1.0   # 空间长度
T = 0.3   # 总时间
u_vel = 1.0  # 对流速度
dt = T / nt

# 创建网格和函数空间
mesh = IntervalMesh(nx, L)
V = FunctionSpace(mesh, "CG", 1)
x = SpatialCoordinate(mesh)

# 定义真实初始条件（高斯脉冲）
def true_initial_condition(x):
    return exp(-50 * (x[0] - 0.3)**2)

# 生成真实解
def generate_true_solution():
    """生成真实解作为参考"""
    q_true = Function(V)
    q_true.interpolate(true_initial_condition(x))
    
    # 时间推进（解析解：q(x,t) = q(x-ut,0)）
    true_solutions = []
    for n in range(nt + 1):
        t = n * dt
        q_t = Function(V)
        # 解析解：q(x,t) = q(x-ut,0)
        q_t.interpolate(exp(-50 * ((x[0] - u_vel * t) % L - 0.3)**2))
        true_solutions.append(q_t.copy(deepcopy=True))
    
    return true_solutions

# 生成观测数据
def generate_observations(true_solutions):
    """生成稀疏观测数据"""
    obs_locations = [0.2, 0.5, 0.8]  # 观测位置
    obs_freq = 5  # 观测频率
    
    observations = []
    obs_times = []
    obs_positions = []
    
    for n in range(0, nt + 1, obs_freq):
        t = n * dt
        q_t = true_solutions[n]
        
        for pos in obs_locations:
            # 添加观测噪声
            noise = 0.01 * np.random.randn()
            obs_value = q_t.at(pos) + noise
            
            observations.append(obs_value)
            obs_times.append(t)
            obs_positions.append(pos)
    
    return observations, obs_times, obs_positions

# 生成背景场
def generate_background():
    """生成背景场（先验估计）"""
    q_bg = Function(V)
    # 背景场：稍微偏移的高斯
    q_bg.interpolate(0.8 * exp(-50 * (x[0] - 0.35)**2))
    return q_bg

# 4D-Var 目标函数
def objective_function(q0, observations, obs_times, obs_positions, q_bg):
    """4D-Var 目标函数"""
    q = Function(V)
    q.assign(q0)
    
    # 背景项
    J_bg = 0.5 * assemble((q - q_bg)**2 * dx)
    
    # 观测项
    J_obs = 0.0
    obs_idx = 0
    
    for n in range(nt + 1):
        t = n * dt
        
        # 检查当前时间步是否有观测
        while obs_idx < len(obs_times) and abs(obs_times[obs_idx] - t) < dt/2:
            pos = obs_positions[obs_idx]
            obs_val = observations[obs_idx]
            
            # 计算观测误差
            pred_val = q.at(pos)
            J_obs += 0.5 * (pred_val - obs_val)**2
            
            obs_idx += 1
        
        # 时间推进（Forward Euler）
        if n < nt:
            v = TestFunction(V)
            F = (q - q0)/dt*v*dx + u_vel*q.dx(0)*v*dx
            solve(F == 0, q)
            q0.assign(q)
    
    return J_bg + J_obs

# 主程序
def main():
    print("=== 对流方程 4D-Var 数据同化（简化版）===")
    
    # 1. 生成真实解
    print("生成真实解...")
    true_solutions = generate_true_solution()
    
    # 2. 生成观测数据
    print("生成观测数据...")
    observations, obs_times, obs_positions = generate_observations(true_solutions)
    print(f"观测数量: {len(observations)}")
    print(f"观测时间: {[f'{t:.3f}' for t in obs_times]}")
    print(f"观测位置: {obs_positions}")
    
    # 3. 生成背景场
    print("生成背景场...")
    q_bg = generate_background()
    
    # 4. 初始猜测
    q0 = Function(V)
    q0.assign(q_bg)  # 用背景场作为初始猜测
    
    # 5. 4D-Var 优化
    print("开始 4D-Var 优化...")
    
    # 使用 firedrake-adjoint
    J = objective_function(q0, observations, obs_times, obs_positions, q_bg)
    control = Control(q0)
    rf = ReducedFunctional(J, control)
    
    # 优化
    opt_q0 = minimize(rf, method="L-BFGS-B", options={"maxiter": 50})
    
    print("优化完成！")
    
    # 6. 评估结果
    print("评估结果...")
    
    # 计算初始条件误差
    q_true_init = true_solutions[0]
    init_error = sqrt(assemble((opt_q0 - q_true_init)**2 * dx))
    init_error_rel = init_error / sqrt(assemble(q_true_init**2 * dx))
    
    print(f"初始条件相对误差: {init_error_rel:.4f}")
    
    # 7. 简单可视化
    print("生成简单可视化...")
    
    # 在几个关键点比较
    test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("\n关键点对比:")
    print("位置\t真实值\t优化值\t背景值")
    for x_val in test_points:
        true_val = q_true_init.at(x_val)
        opt_val = opt_q0.at(x_val)
        bg_val = q_bg.at(x_val)
        print(f"{x_val:.1f}\t{true_val:.4f}\t{opt_val:.4f}\t{bg_val:.4f}")
    
    # 8. 保存结果
    print("保存结果...")
    File("optimized_initial_condition.pvd").write(opt_q0)
    File("true_initial_condition.pvd").write(q_true_init)
    File("background_field.pvd").write(q_bg)
    
    print("=== 完成 ===")
    return opt_q0, true_solutions, observations, obs_times, obs_positions

if __name__ == "__main__":
    opt_q0, true_solutions, observations, obs_times, obs_positions = main() 