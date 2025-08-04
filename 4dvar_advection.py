from firedrake import *
from firedrake.adjoint import *
from pyadjoint import ReducedFunctional
from pyadjoint.tape import continue_annotation
import numpy as np
import math
import matplotlib.pyplot as plt

  # 启动 Pyadjoint tape 记录
from pyadjoint.tape import stop_annotating, continue_annotation
stop_annotating()
continue_annotation()
from firedrake.pyplot import FunctionPlotter, tripcolor
mesh = UnitSquareMesh(40, 40, quadrilateral=True)

# We set up a function space of discontinuous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = FunctionSpace(mesh, "DQ", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)

velocity = as_vector((0.5 - y, x - 0.5))
u = Function(W).interpolate(velocity)

#initial setup for q0 qb

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
qb.assign(q_true_init + 0.1 * Function(V).interpolate(sin(2 * math.pi * x) * sin(2 * math.pi * y)))
q0 = qb.copy(deepcopy=True)
q0.rename("q0")

#time step

T = 2 * math.pi
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
    L2 = replace(L1, {q: q1})
    L3 = replace(L1, {q: q2})

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
            qs.append(q.copy(deepcopy=True))

    return qs if return_series else q

#generate samples

np.random.seed(42)

q_obs_series = solve_rk(q_true_init, return_series=True)
q_obs_end = q_obs_series[-1]  # 只取最后一个时间步的观测

q_sim_end = solve_rk(q0, return_series=False)  # 只返回最终状态

alpha = Constant(1e-3)

# 构造目标函数
misfit = (q_sim_end - q_obs_end)**2 * dx
reg = alpha * (q0 - qb)**2 * dx
J = assemble(misfit + reg)

print('objective function generated')
rf = ReducedFunctional(J, Control(q0))
get_working_tape().progress_bar = ProgressBar

print("Derivative type:", type(rf.derivative()))
grad = rf.derivative()
print("‖∇J‖ =", norm(grad))


J_vals = []
def cb(qval):
    J_now = rf(qval)
    print("J =", J_now)
    J_vals.append(J_now)

opt_q0 = minimize(rf, method="L-BFGS-B", options={"disp": True, "maxiter": 20}, callback=cb)

from firedrake import assemble, sqrt, dx

# L2 误差
def l2_error(a, b):
    return sqrt(assemble((a - b)**2 * dx))

err_prior = l2_error(qb, q_true_init)
err_opt = l2_error(opt_q0, q_true_init)
print("‖qb - q_true‖ₗ₂ =", err_prior)
print("‖opt_q0 - q_true‖ₗ₂ =", err_opt)
print(err_opt / err_prior)

