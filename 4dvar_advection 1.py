#!/usr/bin/env python3
"""
4D-Var Data Assimilation for DG Advection Problem using Firedrake

Based on the cosine-bell-cone-slotted-cylinder advection test case.
This implements variational data assimilation to estimate the initial condition
by minimizing the misfit between model predictions and observations.
"""

import numpy as np
from firedrake import *
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DGAdvectionSolver:
    """DG advection solver based on the provided reference implementation"""
    
    def __init__(self, mesh):
        self.mesh = mesh
        
        # Function spaces
        self.V = FunctionSpace(mesh, "DQ", 1)
        self.W = VectorFunctionSpace(mesh, "CG", 1)
        
        # Set up velocity field (solid body rotation)
        x, y = SpatialCoordinate(mesh)
        velocity = as_vector((0.5 - y, x - 0.5))
        self.u = Function(self.W).interpolate(velocity)
        
        # Time parameters
        self.T = 2*math.pi
        self.dt = self.T/600.0
        self.dtc = Constant(self.dt)
        self.q_in = Constant(1.0)
        
        # Set up DG formulation
        self.setup_dg_solver()
        
    def setup_dg_solver(self):
        """Set up the DG variational forms and solvers"""
        # Test and trial functions
        self.phi = TestFunction(self.V)
        dq_trial = TrialFunction(self.V)
        
        # Mass matrix (LHS)
        self.a = self.phi * dq_trial * dx
        
        # DG flux formulation - create the F functional
        n = FacetNormal(self.mesh)
        un = 0.5*(dot(self.u, n) + abs(dot(self.u, n)))
        
        def create_F(q_func):
            """Create the DG flux form for a given function"""
            return (q_func*div(self.phi*self.u)*dx
                  - conditional(dot(self.u, n) < 0, self.phi*dot(self.u, n)*self.q_in, 0.0)*ds
                  - conditional(dot(self.u, n) > 0, self.phi*dot(self.u, n)*q_func, 0.0)*ds
                  - (self.phi('+') - self.phi('-'))*(un('+')*q_func('+') - un('-')*q_func('-'))*dS)
        
        self.create_F = create_F
        
        # Working variables for Runge-Kutta
        self.dq = Function(self.V, name="dq")
        self.q1 = Function(self.V, name="q1") 
        self.q2 = Function(self.V, name="q2")
        
        # Store solver parameters
        self.params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    
    def solve_forward(self, q0, store_freq=20):
        """Solve forward advection problem and store solutions"""
        q = Function(self.V, name="q")
        q.assign(q0)
        
        solutions = [q.copy(deepcopy=True)]
        
        # Create the RHS forms for each RK stage with current functions
        L1 = self.dtc * self.create_F(q)
        L2 = self.dtc * self.create_F(self.q1) 
        L3 = self.dtc * self.create_F(self.q2)
        
        # Create linear variational problems for each stage
        prob1 = LinearVariationalProblem(self.a, L1, self.dq)
        solv1 = LinearVariationalSolver(prob1, solver_parameters=self.params)
        
        prob2 = LinearVariationalProblem(self.a, L2, self.dq)
        solv2 = LinearVariationalSolver(prob2, solver_parameters=self.params)
        
        prob3 = LinearVariationalProblem(self.a, L3, self.dq)
        solv3 = LinearVariationalSolver(prob3, solver_parameters=self.params)
        
        t = 0.0
        step = 0
        
        while t < self.T - 0.5*self.dt:
            # Three-stage Runge-Kutta following the original code exactly
            solv1.solve()
            self.q1.assign(q + self.dq)
            
            solv2.solve()
            self.q2.assign(0.75*q + 0.25*(self.q1 + self.dq))
            
            solv3.solve()
            q.assign((1.0/3.0)*q + (2.0/3.0)*(self.q2 + self.dq))
            
            step += 1
            t += self.dt
            
            if step % store_freq == 0:
                solutions.append(q.copy(deepcopy=True))
        
        return solutions

def create_true_initial_condition(V):
    """Create the standard cosine-bell-cone-slotted-cylinder initial condition"""
    x, y = SpatialCoordinate(V.mesh())
    
    # Parameters for the three objects
    bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
    cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
    cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
    slot_left = 0.475; slot_right = 0.525; slot_top = 0.85
    
    # Define the three components
    bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
    cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
    slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                 conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                   0.0, 1.0), 0.0)
    
    # Create and interpolate the function
    q_true = Function(V)
    q_true.interpolate(1.0 + bell + cone + slot_cyl)
    
    return q_true

def create_first_guess(V):
    """Create a first guess that's different from the true initial condition"""
    x, y = SpatialCoordinate(V.mesh())
    
    # Modified parameters (different positions and sizes)
    bell_r0 = 0.12; bell_x0 = 0.3; bell_y0 = 0.6
    cone_r0 = 0.18; cone_x0 = 0.4; cone_y0 = 0.3
    cyl_r0 = 0.13; cyl_x0 = 0.6; cyl_y0 = 0.7
    slot_left = 0.55; slot_right = 0.65; slot_top = 0.8
    
    # Define the modified components
    bell = 0.2*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
    cone = 0.8 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
    slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                 conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                   0.0, 0.8), 0.0)
    
    # Create and interpolate the function
    q_guess = Function(V)
    q_guess.interpolate(1.0 + bell + cone + slot_cyl)
    
    return q_guess

class FourDVarProblem:
    """4D-Var data assimilation problem"""
    
    def __init__(self, solver, obs_locations, obs_times, alpha=1e-4):
        self.solver = solver
        self.obs_locations = obs_locations
        self.obs_times = obs_times
        self.alpha = alpha  # Regularization parameter
        self.observations = []
        
    def create_synthetic_observations(self, true_q0, noise_level=0.05):
        """Generate synthetic observations from true solution"""
        print("Generating synthetic observations...")
        true_solutions = self.solver.solve_forward(true_q0, store_freq=20)
        
        observations = []
        for i, time_idx in enumerate(self.obs_times):
            if time_idx < len(true_solutions):
                q_true = true_solutions[time_idx]
                obs_values = []
                
                for loc in self.obs_locations:
                    try:
                        # Use point evaluation
                        obs_val = q_true.at(loc, tolerance=1e-8)
                        obs_values.append(float(obs_val))
                    except:
                        # Fallback to interpolation-based evaluation
                        x, y = SpatialCoordinate(self.solver.mesh)
                        width = 0.02
                        weight = exp(-((x - Constant(loc[0]))**2 + (y - Constant(loc[1]))**2) / (2 * Constant(width)**2))
                        total_weight = assemble(weight * dx)
                        if total_weight > 1e-10:
                            obs_val = assemble(q_true * weight * dx) / total_weight
                            obs_values.append(float(obs_val))
                        else:
                            obs_values.append(1.0)  # Background value
                
                # Add noise
                obs_noisy = np.array(obs_values) + np.random.normal(0, noise_level, len(obs_values))
                observations.append(obs_noisy.tolist())
        
        self.observations = observations
        return observations
    
    def cost_function(self, q0_control, q_guess):
        """Compute 4D-Var cost function"""
        # Solve forward problem
        try:
            solutions = self.solver.solve_forward(q0_control, store_freq=20)
        except:
            return 1e10  # Return large cost if forward solve fails
        
        # Data misfit term
        J_data = 0.0
        
        for i, (obs, time_idx) in enumerate(zip(self.observations, self.obs_times)):
            if time_idx < len(solutions):
                q_model = solutions[time_idx]
                
                # Compute misfit at observation locations
                for j, (obs_val, loc) in enumerate(zip(obs, self.obs_locations)):
                    try:
                        model_val = q_model.at(loc, tolerance=1e-8)
                        J_data += 0.5 * (float(model_val) - obs_val)**2
                    except:
                        # Fallback method
                        x, y = SpatialCoordinate(self.solver.mesh)
                        width = 0.02
                        weight = exp(-((x - Constant(loc[0]))**2 + (y - Constant(loc[1]))**2) / (2 * Constant(width)**2))
                        total_weight = assemble(weight * dx)
                        if total_weight > 1e-10:
                            model_val = assemble(q_model * weight * dx) / total_weight
                            J_data += 0.5 * (float(model_val) - obs_val)**2
        
        # Regularization term (background term)
        try:
            J_reg = 0.5 * self.alpha * float(assemble(dot(q0_control-q_guess,q0_control-q_guess) * dx))
        except:
            J_reg = 0.0
        
        return J_data + J_reg

def optimize_4dvar(fourDVar, first_guess, max_iterations=50):
    """Simple gradient-free optimization for 4D-Var"""
    print("Starting 4D-Var optimization...")
    
    current_q0 = first_guess.copy(deepcopy=True)
    best_q0 = current_q0.copy(deepcopy=True)
    
    # Initial cost
    best_cost = fourDVar.cost_function(current_q0, first_guess)
    print(f"Initial cost: {best_cost:.6f}")
    
    costs = [best_cost]
    step_size = 0.05
    
    for iteration in range(max_iterations):
        # Generate random perturbation
        perturbation = Function(fourDVar.solver.V)
        
        # Create smooth random perturbation using Fourier modes
        x, y = SpatialCoordinate(fourDVar.solver.mesh)
        pert_expr = Constant(0.0)
        
        # Add several random Fourier components
        for k in range(8):
            kx = np.random.randint(1, 5)
            ky = np.random.randint(1, 5)
            amp = np.random.normal(0, 0.1)
            phase_x = np.random.uniform(0, 2*np.pi)
            phase_y = np.random.uniform(0, 2*np.pi)
            
            mode = Constant(amp) * sin(Constant(kx) * pi * x + Constant(phase_x)) * sin(Constant(ky) * pi * y + Constant(phase_y))
            pert_expr = pert_expr + mode
        
        perturbation.interpolate(pert_expr)
        
        # Test perturbed solution
        test_q0 = Function(fourDVar.solver.V)
        test_q0.assign(current_q0 + Constant(step_size) * perturbation)
        
        # Evaluate cost
        test_cost = fourDVar.cost_function(test_q0, first_guess)
        
        # Accept if improvement
        if test_cost < best_cost:
            best_cost = test_cost
            best_q0.assign(test_q0)
            current_q0.assign(test_q0)
            print(f"Iteration {iteration}: New best cost = {test_cost:.6f}")
        
        costs.append(min(test_cost, best_cost))
        
        # Adaptive step size
        if iteration % 10 == 0 and iteration > 0:
            step_size *= 0.9
            
        # Early termination check
        if len(costs) > 10:
            recent_improvement = (costs[-10] - costs[-1]) / costs[-10]
            if recent_improvement < 1e-4:
                print(f"Converged at iteration {iteration}")
                break
    
    print(f"Final cost: {best_cost:.6f}")
    return best_q0, costs

def test_original_solver():
    """Test the original DG solver implementation to verify it works"""
    print("Testing original DG solver...")
    
    mesh = UnitSquareMesh(40, 40, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", 1)
    W = VectorFunctionSpace(mesh, "CG", 1)
    
    # Set up velocity field
    x, y = SpatialCoordinate(mesh)
    velocity = as_vector((0.5 - y, x - 0.5))
    u = Function(W).interpolate(velocity)
    
    # Create initial condition
    q = create_true_initial_condition(V)
    q_init = Function(V).assign(q)
    
    # Time parameters
    T = 2*math.pi
    dt = T/600.0
    dtc = Constant(dt)
    q_in = Constant(1.0)
    
    # Set up DG formulation exactly as in original
    dq_trial = TrialFunction(V)
    phi = TestFunction(V)
    a = phi*dq_trial*dx
    
    def F(q_func):
        n = FacetNormal(mesh)
        un = 0.5*(dot(u, n) + abs(dot(u, n)))
        return (q_func*div(phi*u)*dx
              - conditional(dot(u, n) < 0, phi*dot(u, n)*q_in, 0.0)*ds
              - conditional(dot(u, n) > 0, phi*dot(u, n)*q_func, 0.0)*ds
              - (phi('+') - phi('-'))*(un('+')*q_func('+') - un('-')*q_func('-'))*dS)
    
    L1 = dtc*F(q)
    q1 = Function(V)
    q2 = Function(V)
    L2 = replace(L1, {q: q1})
    L3 = replace(L1, {q: q2})
    
    dq = Function(V)
    params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    
    prob1 = LinearVariationalProblem(a, L1, dq)
    solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
    prob2 = LinearVariationalProblem(a, L2, dq)
    solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
    prob3 = LinearVariationalProblem(a, L3, dq)
    solv3 = LinearVariationalSolver(prob3, solver_parameters=params)
    
    # Time stepping
    t = 0.0
    step = 0
    output_freq = 20
    solutions = [q.copy(deepcopy=True)]
    
    max_steps = 100  # Just do a few steps for testing
    while t < T/60 and step < max_steps:  # Short test run
        solv1.solve()
        q1.assign(q + dq)
        
        solv2.solve()
        q2.assign(0.75*q + 0.25*(q1 + dq))
        
        solv3.solve()
        q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))
        
        step += 1
        t += dt
        
        if step % output_freq == 0:
            solutions.append(q.copy(deepcopy=True))
    
    print(f"Original solver test completed: {len(solutions)} solutions stored")
    return solutions

def run_4dvar_demo():
    """Main 4D-Var demonstration"""
    print("Setting up 4D-Var for DG advection problem...")
    
    # First test the original solver
    test_solutions = test_original_solver()
    
    # Create mesh and solver  
    mesh = UnitSquareMesh(40, 40, quadrilateral=True)
    solver = DGAdvectionSolver(mesh)
    
    # Create true initial condition and first guess
    true_q0 = create_true_initial_condition(solver.V)
    first_guess = create_first_guess(solver.V)
    
    # Define observation network
    obs_locations = [
        (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
        (0.5, 0.5), (0.3, 0.6), (0.7, 0.4), (0.4, 0.3), 
        (0.6, 0.6), (0.25, 0.75), (0.75, 0.25), (0.1, 0.5)
    ]
    
    # Observation times (correspond to stored solution indices)
    obs_times = [1, 3, 5, 7, 9]  # Every few stored time steps
    
    # Create 4D-Var problem
    fourDVar = FourDVarProblem(solver, obs_locations, obs_times, alpha=1e-5)
    
    # Generate synthetic observations
    observations = fourDVar.create_synthetic_observations(true_q0, noise_level=0.02)
    print(f"Created {len(observations)} observation sets with {len(obs_locations)} locations each")
    
    # Run optimization
    optimized_q0, costs = optimize_4dvar(fourDVar, first_guess, max_iterations=30)
    
    # Compute errors
    error_initial = sqrt(assemble((first_guess - true_q0)**2 * dx))
    error_final = sqrt(assemble((optimized_q0 - true_q0)**2 * dx))
    
    print(f"\nResults:")
    print(f"Initial L2 error: {float(error_initial):.6f}")
    print(f"Final L2 error: {float(error_final):.6f}")
    print(f"Error reduction: {(float(error_initial) - float(error_final)) / float(error_initial) * 100:.2f}%")
    
    # Plot results
    print("\nCreating plots...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # True initial condition
        c1 = tripcolor(true_q0, axes=axes[0,0], vmin=1, vmax=2)
        axes[0,0].set_title('True Initial Condition')
        axes[0,0].set_aspect('equal')
        # Add observation points
        obs_x = [loc[0] for loc in obs_locations]
        obs_y = [loc[1] for loc in obs_locations]
        axes[0,0].scatter(obs_x, obs_y, c='red', s=20, marker='x', label='Observations')
        axes[0,0].legend()
        plt.colorbar(c1, ax=axes[0,0])
        
        # First guess
        c2 = tripcolor(first_guess, axes=axes[0,1], vmin=1, vmax=2)
        axes[0,1].set_title('First Guess')
        axes[0,1].set_aspect('equal')
        plt.colorbar(c2, ax=axes[0,1])
        
        # 4D-Var result
        c3 = tripcolor(optimized_q0, axes=axes[1,0], vmin=1, vmax=2)
        axes[1,0].set_title('4D-Var Optimized')
        axes[1,0].set_aspect('equal')
        plt.colorbar(c3, ax=axes[1,0])
        
        # Cost function evolution
        axes[1,1].plot(costs, 'b-', linewidth=2)
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Cost Function')
        axes[1,1].set_title('Cost Function Evolution')
        axes[1,1].grid(True)
        axes[1,1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('4DVar.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Plotting error: {e}")
    
    print("4D-Var demonstration completed!")

if __name__ == "__main__":
    run_4dvar_demo()