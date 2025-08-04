from pyadjoint import *
from firedrake import *
import numpy as np

def efficient_4dvar():
    """高效的4D-Var实现"""
    
    # === 设置（与原代码相同）===
    mesh = UnitSquareMesh(40, 40, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", 1)
    
    # === 启用伴随记录 ===
    tape = get_working_tape()
    tape.clear_tape()
    
    # === 定义控制变量 ===
    q_control = Function(V, name="control")
    q_control.assign(q_background)
    control = Control(q_control)
    
    # === 前向求解器（带磁带记录）===
    def forward_solve_tape(q_init):
        q = Function(V)
        q.assign(q_init)  # 自动记录
        
        solutions = [q.copy(deepcopy=True)]
        
        # 时间积分（每步都记录）
        for step in range(num_steps):
            # Runge-Kutta步骤
            solve(mass_matrix == rhs1, dq1)  # 记录
            q1.assign(q + dq1)               # 记录
            
            solve(mass_matrix == rhs2, dq2)  # 记录  
            q2.assign(0.75*q + 0.25*(q1 + dq2))  # 记录
            
            solve(mass_matrix == rhs3, dq3)  # 记录
            q.assign((1./3.)*q + (2./3.)*(q2 + dq3))  # 记录
            
            if step % store_freq == 0:
                solutions.append(q.copy(deepcopy=True))  # 记录
        
        return solutions
    
    # === 运行前向模型 ===
    solutions = forward_solve_tape(q_control)
    
    # === 构建目标函数 ===
    J_background = 0.5 * alpha * assemble((q_control - q_background)**2 * dx)
    
    J_observation = 0
    for i, (obs_vector, time_idx) in enumerate(zip(observations, obs_times)):
        q_forecast = solutions[time_idx] 
        
        for obs_val, loc in zip(obs_vector, obs_locations):
            # 观测算子（插值）
            forecast_val = interpolate_at_point(q_forecast, loc)
            misfit = obs_val - forecast_val
            J_observation += 0.5 * misfit**2
    
    J_total = J_background + J_observation
    
    # === 创建约化泛函 ===
    reduced_functional = ReducedFunctional(J_total, control)
    
    # === 测试梯度正确性 ===
    print("验证梯度计算...")
    assert taylor_test(reduced_functional, q_control, 
                      Function(V).assign(1.0)) > 1.9
    print("梯度验证通过！")
    
    # === 高效优化 ===
    print("开始L-BFGS优化...")
    optimal_control = minimize(
        reduced_functional,
        method="L-BFGS-B",
        options={
            'disp': True,
            'maxiter': 100,
            'ftol': 1e-9,
            'gtol': 1e-6
        }
    )
    
    return optimal_control

def interpolate_at_point(func, point):
    """在点处插值的可微实现"""
    x, y = SpatialCoordinate(func.function_space().mesh())
    
    # 使用光滑核函数实现可微插值
    width = Constant(0.02)
    kernel = exp(-((x - Constant(point[0]))**2 + 
                  (y - Constant(point[1]))**2) / (2*width**2))
    
    numerator = assemble(func * kernel * dx)
    denominator = assemble(kernel * dx)
    
    return numerator / denominator