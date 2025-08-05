from firedrake import *
from firedrake.adjoint import *
from pyadjoint import ReducedFunctional
import numpy as np
import math
import matplotlib.pyplot as plt
from pyadjoint.tape import pause_annotation, continue_annotation

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
qb.assign(q_true_init + 0.1 * Function(V).interpolate(sin(2 * math.pi * x) * sin(2 * math.pi * y))) #assign changes qb,# check qb still satisfy the initial condition
q0 = qb.copy(deepcopy=True)
q0.rename("q0")

#time step

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
            qs.append(q.copy(deepcopy=True, annotate=False))

    return qs if return_series else q

#generate samples

np.random.seed(42)

q_obs_series = solve_rk(q_true_init, return_series=True)
q_obs_end = q_obs_series[-1]  
continue_annotation()
q_sim_end = solve_rk(q0, return_series=False)  

alpha = Constant(1e-3)

misfit = (q_sim_end - q_obs_end)**2 * dx
reg = alpha * (q0 - qb)**2 * dx
J = assemble(misfit + reg)

print('objective function generated')
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
print(err_opt / err_prior)
 #tao 

fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(2, 3, 1)
c1 = tripcolor(q_true_init, axes=ax1)
plt.colorbar(c1)
ax1.set_title('True Initial Condition\n(q_true_init)', fontsize=12, fontweight='bold')
ax1.set_aspect('equal')

ax2 = plt.subplot(2, 3, 2)
c2 = tripcolor(qb, axes=ax2)
plt.colorbar(c2)
ax2.set_title('Prior Initial Guess\n(qb)', fontsize=12, fontweight='bold')
ax2.set_aspect('equal')

ax3 = plt.subplot(2, 3, 3)
c3 = tripcolor(opt_q0, axes=ax3)
plt.colorbar(c3)
ax3.set_title('Optimized Initial Condition\n(opt_q0)', fontsize=12, fontweight='bold')
ax3.set_aspect('equal')

plt.savefig("4dvar_advection_single.png", dpi=300, bbox_inches='tight')