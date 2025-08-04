import numpy as np
from firedrake import *
from firedrake_adjoint import *
import math
import matplotlib.pyplot as plt

print("=== Step 1: Set up mesh and function spaces ===")
mesh = UnitSquareMesh(40, 40, quadrilateral=True)
V = FunctionSpace(mesh, "DQ", 1)  
W = VectorFunctionSpace(mesh, "CG", 1)  

print("=== Step 2: Set up velocity field (solid body rotation) ===")
x, y = SpatialCoordinate(mesh)
velocity = as_vector((0.5 - y, x - 0.5))
u = Function(W).interpolate(velocity)

print("=== Step 3: Time parameters ===")
T = 2*math.pi  
dt = T/600.0   
dtc = Constant(dt)
q_in = Constant(1.0)  

print("=== Step 4: Set up DG formulation ===")
phi = TestFunction(V)
dq_trial = TrialFunction(V)
a = phi * dq_trial * dx  

n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))  

def create_F(q_func):
    """Create DG flux form"""
    return (q_func*div(phi*u)*dx
          - conditional(dot(u, n) < 0, phi*dot(u, n)*q_in, 0.0)*ds
          - conditional(dot(u, n) > 0, phi*dot(u, n)*q_func, 0.0)*ds
          - (phi('+') - phi('-'))*(un('+')*q_func('+') - un('-')*q_func('-'))*dS)

dq = Function(V, name="dq")
q1 = Function(V, name="q1")
q2 = Function(V, name="q2")
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}

def create_true_q0():
    # get q_true
    bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
    cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25  
    cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
    slot_left = 0.475; slot_right = 0.525; slot_top = 0.85
    
    bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
    cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
    slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                 conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                   0.0, 1.0), 0.0)
    
    q_true = Function(V, name="q_true")
    q_true.interpolate(1.0 + bell + cone + slot_cyl)
    return q_true

q_true = create_true_q0()
print(f"Created q_true with range: [{q_true.dat.data.min():.3f}, {q_true.dat.data.max():.3f}]")

print("=== Step 6: Create background field q_b ===")
def create_background():
    # get q_b
    bell_r0 = 0.12; bell_x0 = 0.3; bell_y0 = 0.6
    cone_r0 = 0.18; cone_x0 = 0.4; cone_y0 = 0.3
    cyl_r0 = 0.13; cyl_x0 = 0.6; cyl_y0 = 0.7
    slot_left = 0.55; slot_right = 0.65; slot_top = 0.8
    
    bell = 0.2*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
    cone = 0.8 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
    slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                 conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                   0.0, 0.8), 0.0)
    
    q_b = Function(V, name="q_background")
    q_b.interpolate(1.0 + bell + cone + slot_cyl)
    return q_b

q_background = create_background()
print(f"Created q_background with range: [{q_background.dat.data.min():.3f}, {q_background.dat.data.max():.3f}]")

def solve_forward(q0, store_freq=20):
    q = Function(V, name="q")
    q.assign(q0)
    
    solutions = [q.copy(deepcopy=True)]
    
    # Create RK forms
    L1 = dtc * create_F(q)
    L2 = dtc * create_F(q1)
    L3 = dtc * create_F(q2)
    
    # Create solvers
    prob1 = LinearVariationalProblem(a, L1, dq)
    solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
    prob2 = LinearVariationalProblem(a, L2, dq)
    solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
    prob3 = LinearVariationalProblem(a, L3, dq)
    solv3 = LinearVariationalSolver(prob3, solver_parameters=params)
    
    t = 0.0
    step = 0
    
    while t < T - 0.5*dt:
        # Three-stage RK
        solv1.solve()
        q1.assign(q + dq)
        
        solv2.solve() 
        q2.assign(0.75*q + 0.25*(q1 + dq))
        
        solv3.solve()
        q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))
        
        step += 1
        t += dt
        
        if step % store_freq == 0:
            solutions.append(q.copy(deepcopy=True))
    
    return solutions

print("=== Step 8: Generate synthetic observations ===")
# Observation locations
obs_locations = [
    (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
    (0.5, 0.5), (0.3, 0.6), (0.7, 0.4), (0.4, 0.3), 
    (0.6, 0.6), (0.25, 0.75), (0.75, 0.25), (0.1, 0.5)
]

# Observation times (indices of stored solutions)
obs_times = [1, 3, 5, 7, 9]

print(f"Solving forward from q_true to generate observations...")
true_solutions = solve_forward(q_true, store_freq=20)
print(f"Got {len(true_solutions)} time snapshots")

# Extract observations
observations = []
noise_level = 0.02

for i, time_idx in enumerate(obs_times):
    if time_idx < len(true_solutions):
        q_true_t = true_solutions[time_idx]
        obs_values = []
        
        for loc in obs_locations:
            try:
                obs_val = q_true_t.at(loc, tolerance=1e-8)
                obs_values.append(float(obs_val))
            except:
                # Fallback: use interpolation
                width = 0.02
                weight = exp(-((x - Constant(loc[0]))**2 + (y - Constant(loc[1]))**2) / (2 * Constant(width)**2))
                total_weight = assemble(weight * dx)
                obs_val = assemble(q_true_t * weight * dx) / total_weight
                obs_values.append(float(obs_val))
        
        # Add noise
        obs_noisy = np.array(obs_values) + np.random.normal(0, noise_level, len(obs_values))
        observations.append(obs_noisy.tolist())

print(f"Generated {len(observations)} observation vectors with {noise_level*100}% noise")

# Regularization parameter
alpha = 1e-3

# Create observation operators (simplified for pyadjoint)
print("=== Setting up observation operators ===")
obs_operators = []
for loc in obs_locations:
    # Create a simple interpolation operator using L2 projection
    psi = TestFunction(V)
    obs_form = psi * exp(-50 * ((x - Constant(loc[0]))**2 + (y - Constant(loc[1]))**2)) * dx
    obs_operators.append(obs_form)

def forward_solve_simple(q_init):
    """Simplified forward solve for adjoint"""
    q = Function(V)
    q.assign(q_init)
    
    stored_states = []
    stored_states.append(q.copy(deepcopy=True))
    
    t = 0.0
    step = 0
    
    while t < T - 0.5*dt:
        # Simplified single-step scheme for adjoint compatibility
        F = create_F(q)
        solve(a == dtc * F, dq, solver_parameters=params)
        q.assign(q + dq)
        
        step += 1
        t += dt
        
        if step % 20 == 0:
            stored_states.append(q.copy(deepcopy=True))
    
    return stored_states

def cost_functional(q_init):
    """Simplified cost functional for pyadjoint"""
    # Background term
    diff = q_init - q_background
    J_bg = alpha * assemble(diff * diff * dx)
    
    # Forward solve
    states = forward_solve_simple(q_init)
    
    # Observation cost - simplified
    J_obs = 0.0
    for i, (obs_data, time_idx) in enumerate(zip(observations, obs_times)):
        if time_idx < len(states):
            state = states[time_idx]
            for j, (obs_val, loc) in enumerate(zip(obs_data, obs_locations)):
                # Simple L2 distance from observation point
                weight = exp(-50 * ((x - Constant(loc[0]))**2 + (y - Constant(loc[1]))**2))
                weight_norm = assemble(weight * dx)
                predicted = assemble(state * weight * dx) / weight_norm
                J_obs += 0.5 * (predicted - Constant(obs_val))**2
    
    return J_bg + J_obs

print("=== Step 10: 4D-Var optimization with pyadjoint ===")

# Initialize
q_opt = Function(V, name="control")
q_opt.assign(q_background)

# Create control
control = Control(q_opt)

print("Testing simplified cost functional...")
try:
    J = cost_functional(q_opt)
    print(f"Initial cost: {float(J):.6f}")
    
    # Compute gradient
    dJ = compute_gradient(J, control)
    grad_norm = sqrt(assemble(dJ * dJ * dx))
    print(f"Initial gradient norm: {float(grad_norm):.6f}")
    
    # Simple optimization loop
    costs = [float(J)]
    step_size = 0.01
    max_iter = 20
    
    print("Starting optimization...")
    for i in range(max_iter):
        # Take gradient step
        q_new = Function(V)
        q_new.assign(q_opt - Constant(step_size) * dJ)
        
        # Evaluate new cost
        J_new = cost_functional(q_new)
        cost_new = float(J_new)
        
        print(f"Iteration {i}: Cost = {cost_new:.6f}")
        
        # Accept if improvement
        if cost_new < costs[-1]:
            q_opt.assign(q_new)
            costs.append(cost_new)
            
            # Recompute gradient
            dJ = compute_gradient(J_new, control)
            grad_norm = sqrt(assemble(dJ * dJ * dx))
            
            if float(grad_norm) < 1e-5:
                print(f"Converged at iteration {i}")
                break
        else:
            step_size *= 0.8
            print(f"  Reducing step size to {step_size:.4f}")
            costs.append(costs[-1])
            
        if step_size < 1e-6:
            print("Step size too small, stopping")
            break
    
    q_best = q_opt.copy(deepcopy=True)
    
except Exception as e:
    print(f"Pyadjoint optimization failed: {e}")
    print("Falling back to finite difference optimization...")
    
    # Fallback to original method
    q_current = q_background.copy(deepcopy=True)
    q_best = q_current.copy(deepcopy=True)
    
    def cost_function_fd(q_analysis):
        try:
            q_diff = Function(V)
            q_diff.assign(q_analysis - q_background)
            J_background = alpha * float(assemble(dot(q_diff, q_diff) * dx))
            
            solutions = solve_forward(q_analysis, store_freq=20)
            J_observation = 0.0
            n_obs = 0
            
            for i, (obs_vector, time_idx) in enumerate(zip(observations, obs_times)):
                if time_idx < len(solutions):
                    q_forecast = solutions[time_idx]
                    
                    for j, (obs_val, loc) in enumerate(zip(obs_vector, obs_locations)):
                        try:
                            forecast_val = q_forecast.at(loc, tolerance=1e-8)
                            misfit = obs_val - float(forecast_val)
                            J_observation += 0.5 * (misfit**2)
                            n_obs += 1
                        except:
                            width = 0.02
                            weight = exp(-((x - Constant(loc[0]))**2 + (y - Constant(loc[1]))**2) / (2 * Constant(width)**2))
                            total_weight = assemble(weight * dx)
                            forecast_val = assemble(q_forecast * weight * dx) / total_weight
                            misfit = obs_val - float(forecast_val)
                            J_observation += 0.5 * (misfit**2)
                            n_obs += 1
            
            return J_background + J_observation
            
        except Exception as e:
            return 1e10
    
    best_cost = cost_function_fd(q_current)
    costs = [best_cost]
    step_size = 0.05
    max_iterations = 30

    for iteration in range(max_iterations):
        perturbation = Function(V)
        pert_expr = Constant(0.0)
        
        for k in range(6):
            kx = np.random.randint(1, 4)
            ky = np.random.randint(1, 4)
            amp = np.random.normal(0, 0.1)
            phase_x = np.random.uniform(0, 2*np.pi)
            phase_y = np.random.uniform(0, 2*np.pi)
            
            mode = Constant(amp) * sin(Constant(kx) * pi * x + Constant(phase_x)) * sin(Constant(ky) * pi * y + Constant(phase_y))
            pert_expr = pert_expr + mode
        
        perturbation.interpolate(pert_expr)
        
        q_test = Function(V)
        q_test.assign(q_current + Constant(step_size) * perturbation)
        
        test_cost = cost_function_fd(q_test)
        
        if test_cost < best_cost:
            improvement = best_cost - test_cost
            best_cost = test_cost
            q_best.assign(q_test)
            q_current.assign(q_test) 
            print(f"Iteration {iteration}: New best cost = {test_cost:.6f}, improvement = {improvement:.6f}")
        
        costs.append(best_cost)
        
        if iteration % 10 == 0 and iteration > 0:
            step_size *= 0.9
        
        if len(costs) > 15:
            recent_improvement = (costs[-15] - costs[-1]) / costs[0]
            if recent_improvement < 1e-5:
                print(f"Converged at iteration {iteration}")
                break

print("=== Step 12: Compute errors ===")
error_background = sqrt(assemble((q_background - q_true)**2 * dx))
error_analysis = sqrt(assemble((q_best - q_true)**2 * dx))

print(f"\nResults:")
print(f"Background L2 error: {float(error_background):.6f}")
print(f"Analysis L2 error:   {float(error_analysis):.6f}")
print(f"Error reduction:     {(float(error_background) - float(error_analysis)) / float(error_background) * 100:.2f}%")

#Plot the results
try:
    from firedrake.pyplot import tripcolor
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # True field
    c1 = tripcolor(q_true, axes=axes[0,0], vmin=1, vmax=2.5)
    axes[0,0].set_title('True Initial Condition')
    axes[0,0].set_aspect('equal')
    obs_x = [loc[0] for loc in obs_locations]
    obs_y = [loc[1] for loc in obs_locations]
    axes[0,0].scatter(obs_x, obs_y, c='red', s=20, marker='x', label='Observations')
    axes[0,0].legend()
    plt.colorbar(c1, ax=axes[0,0])
    
    # Background field
    c2 = tripcolor(q_background, axes=axes[0,1], vmin=1, vmax=2.5)
    axes[0,1].set_title('Background Field')
    axes[0,1].set_aspect('equal')
    plt.colorbar(c2, ax=axes[0,1])
    
    # Analysis result
    c3 = tripcolor(q_best, axes=axes[1,0], vmin=1, vmax=2.5)
    axes[1,0].set_title('4D-Var Analysis')
    axes[1,0].set_aspect('equal')
    plt.colorbar(c3, ax=axes[1,0])
    
    # Cost evolution
    if costs:
        axes[1,1].plot(costs, 'b-', linewidth=2)
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Cost Function J')
        axes[1,1].set_title('Cost Evolution')
        axes[1,1].grid(True)
        axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('4DVAR_Pyadjoint.png', dpi=150, bbox_inches='tight')
    plt.show()
    
except Exception as e:
    print(f"Plotting error: {e}")

print("Finished")