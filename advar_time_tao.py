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
T = 1*math.pi
dt = T / 200.0
dtc = Constant(dt)
nt = int(T / dt)
output_freq = 10  
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
    
    solv1 = LinearVariationalSolver(LinearVariationalProblem(a, L1, dq, constant_jacobian=True), solver_parameters=params)
    solv2 = LinearVariationalSolver(LinearVariationalProblem(a, L2, dq, constant_jacobian=True), solver_parameters=params)
    solv3 = LinearVariationalSolver(LinearVariationalProblem(a, L3, dq, constant_jacobian=True), solver_parameters=params)

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

observation_configs = {
    "sparse": [  # 稀疏观测
        [0.5, 0.5], [0.25, 0.25], [0.75, 0.75]
    ],
    
    "dense": [  # 密集观测
        [0.2, 0.2], [0.2, 0.3], [0.2, 0.4], [0.2, 0.5], [0.2, 0.6], [0.2, 0.7], [0.2, 0.8],
        [0.3, 0.2], [0.3, 0.3], [0.3, 0.4], [0.3, 0.5], [0.3, 0.6], [0.3, 0.7], [0.3, 0.8],
        [0.4, 0.2], [0.4, 0.3], [0.4, 0.4], [0.4, 0.5], [0.4, 0.6], [0.4, 0.7], [0.4, 0.8],
        [0.5, 0.2], [0.5, 0.3], [0.5, 0.4], [0.5, 0.5], [0.5, 0.6], [0.5, 0.7], [0.5, 0.8],
        [0.6, 0.2], [0.6, 0.3], [0.6, 0.4], [0.6, 0.5], [0.6, 0.6], [0.6, 0.7], [0.6, 0.8],
        [0.7, 0.2], [0.7, 0.3], [0.7, 0.4], [0.7, 0.5], [0.7, 0.6], [0.7, 0.7], [0.7, 0.8],
        [0.8, 0.2], [0.8, 0.3], [0.8, 0.4], [0.8, 0.5], [0.8, 0.6], [0.8, 0.7], [0.8, 0.8]
    ],
    
    "clustered": [  # 聚集在钟形附近
        [0.2, 0.45], [0.2, 0.5], [0.2, 0.55],
        [0.25, 0.45], [0.25, 0.5], [0.25, 0.55],
        [0.3, 0.45], [0.3, 0.5], [0.3, 0.55]
    ],
    
    "boundary": [  # 边界观测
        [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
        [0.1, 0.5], [0.9, 0.5],
        [0.1, 0.9], [0.5, 0.9], [0.9, 0.9]
    ],
    
    "cross": [  # 十字形观测
        [0.5, 0.2], [0.5, 0.4], [0.5, 0.6], [0.5, 0.8],
        [0.2, 0.5], [0.4, 0.5], [0.6, 0.5], [0.8, 0.5]
    ]
}


observation_mesh = VertexOnlyMesh(mesh, observation_configs["dense"])  # 使用稀疏观测配置
vom = FunctionSpace(observation_mesh, "DG", 0)

#operator H
def H(x):
    return assemble(interpolate(x, vom))

q_obs_series = solve_rk(q_true_init, return_series=True)
q_obs_end = q_obs_series[-1]  

# obs_times = [0, len(q_obs_series)//4, 2*len(q_obs_series)//3, -1] #ob times initial time
obs_times = [0] + [j*len(q_obs_series)//10 for j in range(10)] + [-1]
y_obs = [H(q_obs_series[i]) for i in obs_times] #sample ob

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

from pyadjoint.optimization.optimization_problem import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver

tao_parameters = {
    'tao_monitor': None,
    'tao_converged_reason': None,
    'tao_gttol': 2e-1,
    'tao_grtol': 2e-1,
    'tao_type': 'lmvm',
}

min_problem = MinimizationProblem(rf)
tao_solver = TAOSolver(
    min_problem, parameters=tao_parameters,
    convert_options={'riesz_representation': 'l2'})
PETSc.Options()[tao_solver.tao.getOptionsPrefix()+"tao_view"] = ":tao_view.log"
opt_q0 = tao_solver.solve()

err_prior = errornorm(qb, q_true_init)
err_opt = errornorm(opt_q0, q_true_init)
print("‖qb - q_true‖ₗ₂ =", err_prior)
print("‖opt_q0 - q_true‖ₗ₂ =", err_opt)

y_true = H(q_obs_end)
y_prior = H(solve_rk(qb, return_series=False))
y_opt = H(solve_rk(opt_q0, return_series=False))

obs_norm = norm(y_true)
obs_error_prior = errornorm(y_prior, y_true)/obs_norm
obs_error_opt = errornorm(y_opt, y_true)/obs_norm

print(f"Observation error (prior): {obs_error_prior:.6e}")
print(f"Observation error (optimized): {obs_error_opt:.6e}")
print(f"Observation improvement ratio: {(obs_error_opt/obs_error_prior):.6f}")

obs_x = [pt[0] for pt in observation_points]
obs_y = [pt[1] for pt in observation_points]

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

