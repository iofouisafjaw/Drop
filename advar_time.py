from firedrake import *
from firedrake.__future__ import interpolate 
from firedrake.adjoint import *
from pyadjoint import ReducedFunctional
import numpy as np
import math
import matplotlib.pyplot as plt
from pyadjoint.tape import pause_annotation, continue_annotation
from firedrake.pyplot import FunctionPlotter, tripcolor

mesh = UnitSquareMesh(40, 40, quadrilateral=True)
V = FunctionSpace(mesh, "DQ", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)

velocity = as_vector((0.5 - y, x - 0.5))
u = Function(W).interpolate(velocity)

# initial setup for q_true, q0, qb

bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
             conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
               0.0, 1.0), 0.0)

q0 = Function(V, name="q0")
qb = Function(V, name="qb")
bell = 0.25 * (1 + cos(math.pi * min_value(sqrt(pow(x - 0.25, 2) + pow(y - 0.5, 2)) / 0.15, 1.0)))
q_true_init = Function(V, name="q_true_init").interpolate(1.0 + bell)
qb.assign(q_true_init + 0.1 * Function(V).interpolate(sin(2 * math.pi * x) * sin(2 * math.pi * y))) #assign changes qb, check qb still satisfy the initial condition
q0 = qb.copy(deepcopy=True)
q0.rename("q0")

# time steps and parameters
T = 2*math.pi
dt = T / 600.0
dtc = Constant(dt)
nt = int(T / dt)
output_freq = 20  
save_steps = nt // output_freq

phi = TestFunction(V)
dq_trial = TrialFunction(V)

n = FacetNormal(mesh)
un = 0.5 * (dot(u, n) + abs(dot(u, n)))
q_in = Constant(1.0)

def F(q):
    return (q * div(phi * u) * dx
          - conditional(dot(u, n) < 0, phi * dot(u, n) * q_in, 0.0) * ds
          - conditional(dot(u, n) > 0, phi * dot(u, n) * q, 0.0) * ds
          - (phi('+') - phi('-')) * (un('+') * q('+') - un('-') * q('-')) * dS)

a = phi * dq_trial * dx

def solve_rk(q0_init, return_series=False):
    q = q0_init.copy(deepcopy=True)

    q1, q2 = Function(V), Function(V)
    dq = Function(V)
    L1 = dtc * F(q)
    L2 = dtc * F(q1)
    L3 = dtc * F(q2)

    params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    
    solv1 = LinearVariationalSolver(LinearVariationalProblem(a, L1, dq), solver_parameters=params)
    solv2 = LinearVariationalSolver(LinearVariationalProblem(a, L2, dq), solver_parameters=params)
    solv3 = LinearVariationalSolver(LinearVariationalProblem(a, L3, dq), solver_parameters=params)

    qs = []
    t, step = 0.0, 0
    while t < T - 0.5 * float(dt):
        solv1.solve()
        q1.assign(q + dq)
        solv2.solve()
        q2.assign(0.75 * q + 0.25 * (q1 + dq))
        solv3.solve()
        q.assign((1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dq))

        step += 1
        t += float(dt)
        if step % output_freq == 0:
            qs.append(q.copy(deepcopy=True)) #remove annotate=False 

    return qs if return_series else q

# vertexonlymesh
observation_points = [
    [0.25, 0.5],  
    [0.5, 0.5],    
    [0.75, 0.5],   
    [0.5, 0.25],   
    [0.5, 0.75],
    [0.25, 0.25],  
    [0.75, 0.25],
    [0.25, 0.75],
    [0.75, 0.75]
]

observation_mesh = VertexOnlyMesh(mesh, observation_points)
vom = FunctionSpace(observation_mesh, "DG", 0)

#operator H
def H(x):
    return assemble(interpolate(x, vom))

q_obs_series = solve_rk(q_true_init, return_series=True)
q_obs_end = q_obs_series[-1]  

obs_times = [len(q_obs_series)//3, 2*len(q_obs_series)//3, -1]
y_obs = [H(q_obs_series[i]) for i in obs_times]

np.random.seed(42)

continue_annotation()

alpha = Constant(1e-3)
#background term
J = assemble(alpha * (q0 - qb)**2 * dx)
# Observation term 
q_series = solve_rk(q0, return_series=True)
for i, obs_idx in enumerate(obs_times):
    if obs_idx < len(q_series):
        J = J + assemble((H(q_series[obs_idx]) - y_obs[i])**2 * dx)
# Additional regularization 

rf = ReducedFunctional(J, Control(q0))

pause_annotation()
get_working_tape().progress_bar = ProgressBar

print("Derivative type:", type(rf.derivative()))
grad = rf.derivative()
print("‖∇J‖ =", norm(grad))

opt_q0 = minimize(rf, method="L-BFGS-B", options={"disp": True, "maxiter": 20}, derivative_options={'riesz_representation':'l2'})

err_prior = errornorm(qb, q_true_init)
err_opt = errornorm(opt_q0, q_true_init)
print("‖qb - q_true‖ₗ₂ =", err_prior)
print("‖opt_q0 - q_true‖ₗ₂ =", err_opt)
print(err_prior / err_opt)

y_true = H(q_obs_end)
y_prior = H(solve_rk(qb, return_series=False))
y_opt = H(solve_rk(opt_q0, return_series=False))

obs_error_prior = errornorm(y_prior, y_true)
obs_error_opt = errornorm(y_opt, y_true)

print(f"Observation error (prior): {obs_error_prior:.6e}")
print(f"Observation error (optimized): {obs_error_opt:.6e}")
print(f"Observation improvement ratio: {obs_error_prior/obs_error_opt:.6f}")

obs_x = [pt[0] for pt in observation_points]
obs_y = [pt[1] for pt in observation_points]


# 1. 原始的初始条件对比图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('4D-Var: Initial Condition Comparison', fontsize=16, fontweight='bold')

ax1 = axes[0]
tripcolor(q_true_init, axes=ax1, cmap='viridis')
ax1.set_title('True Initial Condition\n$q_{true}$', fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.scatter(obs_x, obs_y, c='red', s=50, marker='x', linewidth=2, label='Observation Points')
ax1.legend()

ax2 = axes[1]
tripcolor(qb, axes=ax2, cmap='viridis')
ax2.set_title('Background/Prior Initial Condition\n$q_b$', fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.scatter(obs_x, obs_y, c='red', s=50, marker='x', linewidth=2)

ax3 = axes[2]
tripcolor(opt_q0, axes=ax3, cmap='viridis')
ax3.set_title('4D-Var Optimized Initial Condition\n$q_{opt}$', fontweight='bold')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.scatter(obs_x, obs_y, c='red', s=50, marker='x', linewidth=2)

plt.tight_layout()
plt.savefig("4dvar_advection_time.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. 时间演化对比
print("Generating time evolution comparison...")
q_true_series = solve_rk(q_true_init, return_series=True)
q_prior_series = solve_rk(qb, return_series=True)
q_opt_series = solve_rk(opt_q0, return_series=True)

fig2, axes = plt.subplots(3, 4, figsize=(16, 12))
fig2.suptitle('Time Evolution Comparison', fontsize=16, fontweight='bold')

# 选择4个关键时刻
time_indices = [0, len(q_true_series)//4, len(q_true_series)//2, 3*len(q_true_series)//4]
time_labels = ['Early (t≈0)', 'Mid-Early (t≈π/2)', 'Mid-Late (t≈π)', 'Late (t≈3π/2)']

for j, (t_idx, t_label) in enumerate(zip(time_indices, time_labels)):
    if t_idx < len(q_true_series):
        # 真实解
        tripcolor(q_true_series[t_idx], axes=axes[0, j], cmap='viridis')
        axes[0, j].set_title(f'True - {t_label}')
        axes[0, j].scatter(obs_x, obs_y, c='red', s=20, marker='x')
        
        # 先验解
        tripcolor(q_prior_series[t_idx], axes=axes[1, j], cmap='viridis')  
        axes[1, j].set_title(f'Prior - {t_label}')
        axes[1, j].scatter(obs_x, obs_y, c='red', s=20, marker='x')
        
        # 4D-Var优化解
        tripcolor(q_opt_series[t_idx], axes=axes[2, j], cmap='viridis')
        axes[2, j].set_title(f'4D-Var - {t_label}')
        axes[2, j].scatter(obs_x, obs_y, c='red', s=20, marker='x')

# 添加行标签
axes[0, 0].set_ylabel('True Solution', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Prior Solution', fontweight='bold', fontsize=12)
axes[2, 0].set_ylabel('4D-Var Solution', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("4dvar_time_evolution.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. 观测点详细分析
print("Generating observation point analysis...")
fig3, axes = plt.subplots(3, 3, figsize=(18, 15))
fig3.suptitle('Observation Point Time Series Analysis', fontsize=16, fontweight='bold')

# 选择9个观测点进行分析
selected_obs_points = list(range(len(observation_points)))

for idx, obs_idx in enumerate(selected_obs_points):
    ax = axes[idx//3, idx%3]
    
    # 提取该观测点在所有时刻的值
    all_times = np.arange(len(q_true_series)) * output_freq * dt
    true_series = [H(q).dat.data[obs_idx] for q in q_true_series]
    prior_series = [H(q).dat.data[obs_idx] for q in q_prior_series]  
    opt_series = [H(q).dat.data[obs_idx] for q in q_opt_series]
    
    # 观测时刻和数据
    obs_time_vals = [obs_times[i] * output_freq * dt for i in range(len(obs_times))]
    obs_data_vals = [y_obs[i].dat.data[obs_idx] for i in range(len(y_obs))]
    
    # 绘制时间序列
    ax.plot(all_times, true_series, 'g-', linewidth=2, label='True', alpha=0.8)
    ax.plot(all_times, prior_series, 'r--', linewidth=2, label='Prior', alpha=0.8)
    ax.plot(all_times, opt_series, 'b-', linewidth=2, label='4D-Var', alpha=0.8)
    ax.scatter(obs_time_vals, obs_data_vals, c='orange', s=80, marker='*', 
              label='Observations', zorder=5, edgecolor='black', linewidth=1)
    
    # 标记观测时刻
    for obs_time in obs_time_vals:
        ax.axvline(x=obs_time, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # 设置标题和标签
    point_coord = observation_points[obs_idx]
    ax.set_title(f'Point {obs_idx+1}: ({point_coord[0]:.2f}, {point_coord[1]:.2f})', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 计算该点的改善效果
    final_true = H(q_true_series[-1]).dat.data[obs_idx]
    final_prior = H(q_prior_series[-1]).dat.data[obs_idx]
    final_opt = H(q_opt_series[-1]).dat.data[obs_idx]
    
    prior_error = abs(final_prior - final_true)
    opt_error = abs(final_opt - final_true)
    improvement = prior_error / opt_error if opt_error > 1e-12 else 999
    
    # 在图上添加改善信息
    ax.text(0.02, 0.98, f'Improvement: {improvement:.1f}x', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)

plt.tight_layout()
plt.savefig("4dvar_observation_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. 观测时刻详细对比
print("Generating observation time comparison...")
fig4, axes = plt.subplots(3, len(obs_times), figsize=(5*len(obs_times), 12))
fig4.suptitle('Solutions at Observation Times', fontsize=16, fontweight='bold')

for j, obs_time_idx in enumerate(obs_times):
    # 真实解在观测时刻
    tripcolor(q_true_series[obs_time_idx], axes=axes[0, j], cmap='viridis')
    axes[0, j].set_title(f'True at t={obs_time_idx * output_freq * dt:.2f}')
    axes[0, j].scatter(obs_x, obs_y, c='red', s=30, marker='x')
    
    # 先验解在观测时刻
    tripcolor(q_prior_series[obs_time_idx], axes=axes[1, j], cmap='viridis')
    axes[1, j].set_title(f'Prior at t={obs_time_idx * output_freq * dt:.2f}')
    axes[1, j].scatter(obs_x, obs_y, c='red', s=30, marker='x')
    
    # 4D-Var解在观测时刻
    tripcolor(q_opt_series[obs_time_idx], axes=axes[2, j], cmap='viridis')
    axes[2, j].set_title(f'4D-Var at t={obs_time_idx * output_freq * dt:.2f}')
    axes[2, j].scatter(obs_x, obs_y, c='red', s=30, marker='x')

# 添加行标签
axes[0, 0].set_ylabel('True Solution', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Prior Solution', fontweight='bold', fontsize=12) 
axes[2, 0].set_ylabel('4D-Var Solution', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("4dvar_observation_times.png", dpi=300, bbox_inches='tight')
plt.show()