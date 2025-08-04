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

# 时间积分函数（Forward Euler）
def integrate_forward(q_init, nt_steps):
    """从初始条件积分到指定时间步"""
    q = Function(V)
    q.assign(q_init)
    trajectory = [q.copy(deepcopy=True)]
    
    for n in range(nt_steps):
        v = TestFunction(V)
        F = (q - q_init)/dt*v*dx + u_vel*q.dx(0)*v*dx
        solve(F == 0, q)
        q_init.assign(q)
        trajectory.append(q.copy(deepcopy=True))
    
    return trajectory

# 生成背景场
def generate_background():
    """生成背景场 q_b"""
    q_b = Function(V)
    # 背景场：稍微偏移的高斯
    q_b.interpolate(0.8 * exp(-50 * (x[0] - 0.35)**2))
    return q_b

# 生成观测数据
def generate_observations(true_trajectory, obs_freq=5):
    """从真实轨迹生成观测数据"""
    obs_locations = [0.2, 0.5, 0.8]  # 观测位置
    
    observations = []
    obs_times = []
    obs_positions = []
    
    for n in range(0, len(true_trajectory), obs_freq):
        t = n * dt
        q_t = true_trajectory[n]
        
        for pos in obs_locations:
            # 添加观测噪声
            noise = 0.01 * np.random.randn()
            obs_value = q_t.at(pos) + noise
            
            observations.append(obs_value)
            obs_times.append(t)
            obs_positions.append(pos)
    
    return observations, obs_times, obs_positions

# 4D-Var 目标函数
def objective_function(q0, observations, obs_times, obs_positions, q_b):
    """4D-Var 目标函数: J = J_b + J_o"""
    # 背景误差项
    J_b = 0.5 * assemble((q0 - q_b)**2 * dx)
    
    # 观测误差项
    J_o = 0.0
    
    # 前向积分 q0 从 0 到 T
    q = Function(V)
    q.assign(q0)
    obs_idx = 0
    
    for n in range(nt + 1):
        t = n * dt
        
        # 检查当前时间步是否有观测
        while obs_idx < len(obs_times) and abs(obs_times[obs_idx] - t) < dt/2:
            pos = obs_positions[obs_idx]
            obs_val = observations[obs_idx]
            
            # 计算观测误差
            pred_val = q.at(pos)
            J_o += 0.5 * (pred_val - obs_val)**2
            
            obs_idx += 1
        
        # 时间推进（Forward Euler）
        if n < nt:
            v = TestFunction(V)
            F = (q - q0)/dt*v*dx + u_vel*q.dx(0)*v*dx
            solve(F == 0, q)
            q0.assign(q)
    
    return J_b + J_o

# 主程序
def main():
    print("=== 标准4D-Var流程：对流方程数据同化 ===")
    
    # 1. 生成真实解作为参考
    print("1. 生成真实解...")
    q_true_init = Function(V)
    q_true_init.interpolate(true_initial_condition(x))
    true_trajectory = integrate_forward(q_true_init, nt)
    print(f"   真实轨迹长度: {len(true_trajectory)}")
    
    # 2. 生成背景场 q_b
    print("2. 生成背景场 q_b...")
    q_b = generate_background()
    
    # 3. 积分背景场到T
    print("3. 积分背景场到T...")
    q_b_copy = q_b.copy(deepcopy=True)
    background_trajectory = integrate_forward(q_b_copy, nt)
    print(f"   背景轨迹长度: {len(background_trajectory)}")
    
    # 4. 生成观测数据
    print("4. 生成观测数据...")
    observations, obs_times, obs_positions = generate_observations(true_trajectory)
    print(f"   观测数量: {len(observations)}")
    print(f"   观测时间: {[f'{t:.3f}' for t in obs_times]}")
    print(f"   观测位置: {obs_positions}")
    
    # 5. 初始猜测 q0 = q_b + ε
    print("5. 生成初始猜测 q0 = q_b + ε...")
    q0 = Function(V)
    q0.assign(q_b)
    
    # 添加小扰动 ε
    epsilon = Function(V)
    epsilon.interpolate(0.1 * exp(-50 * (x[0] - 0.4)**2))  # 小的高斯扰动
    q0.assign(q0 + epsilon)
    
    print(f"   扰动大小: {sqrt(assemble(epsilon**2 * dx)):.6f}")
    
    # 6. 前向积分 q0 从 0 到 T
    print("6. 前向积分 q0 从 0 到 T...")
    q0_copy = q0.copy(deepcopy=True)
    initial_trajectory = integrate_forward(q0_copy, nt)
    print(f"   初始轨迹长度: {len(initial_trajectory)}")
    
    # 7. 构建4D-Var目标函数
    print("7. 构建4D-Var目标函数...")
    J = objective_function(q0, observations, obs_times, obs_positions, q_b)
    print(f"   初始目标函数值: {J:.6f}")
    
    # 8. 使用伴随方法最小化J
    print("8. 使用伴随方法最小化J...")
    control = Control(q0)
    rf = ReducedFunctional(J, control)
    
    # 优化
    opt_q0 = minimize(rf, method="L-BFGS-B", options={"maxiter": 100})
    
    print("   优化完成！")
    
    # 9. 评估结果
    print("9. 评估结果...")
    
    # 计算初始条件误差
    init_error = sqrt(assemble((opt_q0 - q_true_init)**2 * dx))
    init_error_rel = init_error / sqrt(assemble(q_true_init**2 * dx))
    
    bg_error = sqrt(assemble((q_b - q_true_init)**2 * dx))
    bg_error_rel = bg_error / sqrt(assemble(q_true_init**2 * dx))
    
    print(f"   背景场相对误差: {bg_error_rel:.4f}")
    print(f"   优化解相对误差: {init_error_rel:.4f}")
    print(f"   误差改善: {(bg_error_rel - init_error_rel)/bg_error_rel*100:.1f}%")
    
    # 10. 积分最优解到T
    print("10. 积分最优解到T...")
    opt_q0_copy = opt_q0.copy(deepcopy=True)
    optimal_trajectory = integrate_forward(opt_q0_copy, nt)
    
    # 11. 可视化结果
    print("11. 生成可视化...")
    
    # 在几个关键点比较
    test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("\n关键点对比:")
    print("位置\t真实值\t背景值\t优化值")
    for x_val in test_points:
        true_val = q_true_init.at(x_val)
        bg_val = q_b.at(x_val)
        opt_val = opt_q0.at(x_val)
        print(f"{x_val:.1f}\t{true_val:.4f}\t{bg_val:.4f}\t{opt_val:.4f}")
    
    # 12. 保存结果
    print("12. 保存结果...")
    File("optimal_initial_condition.pvd").write(opt_q0)
    File("true_initial_condition.pvd").write(q_true_init)
    File("background_field.pvd").write(q_b)
    File("initial_guess.pvd").write(q0)
    
    print("=== 标准4D-Var流程完成 ===")
    
    return {
        'opt_q0': opt_q0,
        'q_true_init': q_true_init,
        'q_b': q_b,
        'observations': observations,
        'obs_times': obs_times,
        'obs_positions': obs_positions,
        'true_trajectory': true_trajectory,
        'background_trajectory': background_trajectory,
        'optimal_trajectory': optimal_trajectory
    }

if __name__ == "__main__":
    results = main() 