from firedrake import *
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams['animation.ffmpeg_path'] = "/usr/local/bin/ffmpeg"

mesh = UnitSquareMesh(40, 40, quadrilateral=True)

V = FunctionSpace(mesh, "DQ", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)

velocity = as_vector((0.5 - y, x - 0.5))
u = Function(W).interpolate(velocity)

# Initial condition components
bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

bell = 0.25*(1+cos(math.pi*min_value(sqrt((x-bell_x0)**2 + (y-bell_y0)**2)/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt((x-cone_x0)**2 + (y-cone_y0)**2)/cyl_r0, 1.0)
slot_cyl = conditional(sqrt((x-cyl_x0)**2 + (y-cyl_y0)**2) < cyl_r0,
             conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
               0.0, 1.0), 0.0)

q = Function(V).interpolate(1.0 + bell + cone + slot_cyl)
q_init = Function(V).assign(q)
qs = []
T = 2*math.pi
dt = T/2000.0
dtc = Constant(dt)
q_in = Constant(1.0)

phi = TestFunction(V)
q_trial = TrialFunction(V)
q_new = Function(V)

n = FacetNormal(mesh)
un = 0.5 * (dot(u, n) + abs(dot(u, n)))

def F(q_1):
    return (q_1*div(phi*u)*dx
          - conditional(dot(u, n) < 0, phi*dot(u, n)*q_in, 0.0)*ds
          - conditional(dot(u, n) > 0, phi*dot(u, n)*q_1, 0.0)*ds
          - (phi('+') - phi('-'))*(un('+')*q_1('+') - un('-')*q_1('-'))*dS)

a = q_new*phi*dx-dtc*F(q_new)

L = q*phi*dx
    

params = {'ksp_type': 'preonly', 'pc_type': 'lu','snes_type':'ksponly'}
problem = NonlinearVariationalProblem(a-L, q_new)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)

# Time loop
t = 0.0
step = 0
output_freq = 20
while t < T - 0.5*dt:
    solver.solve()
    q.assign(q_new)

    step += 1
    t += dt

    if step % output_freq == 0:
        qs.append(q.copy(deepcopy=True))
        print("t =", t)

L2_err = sqrt(assemble((q - q_init)**2 * dx))
L2_init = sqrt(assemble(q_init**2 * dx))
print("Relative L2 error:", L2_err / L2_init)


nsp = 16
fn_plotter = FunctionPlotter(mesh, num_sample_points=nsp)
fig, axes = plt.subplots()
axes.set_aspect('equal')
colors = tripcolor(q_init, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
fig.colorbar(colors)

# Now we'll create a function to call in each frame. This function will use the
# helper object we created before. ::

def animate(q):
    colors.set_array(fn_plotter(q))

# The last step is to make the animation and save it to a file. ::

interval = 1e3 * output_freq * dt
animation = FuncAnimation(fig, animate, frames=qs, interval=interval)
try:
    animation.save("DG_advection_be.gif", writer="pillow")
except:
    print("Failed to write movie! Try installing `ffmpeg`.")