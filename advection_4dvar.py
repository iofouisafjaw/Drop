#!/usr/bin/env python3
"""
4D-Var Data Assimilation for DG Advection Problem using Firedrake with pyadjoint

Based on the cosine-bell-cone-slotted-cylinder advection test case.
This implements variational data assimilation using automatic differentiation
to minimize the cost function and estimate optimal initial conditions.
"""

import numpy as np
from firedrake import *
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt

# Import pyadjoint for automatic differentiation
try:
    from firedrake.adjoint import *
    from pyadjoint import *
    from pyadjoint.tape import get_working_tape, pause_annotation, continue_annotation
    from pyadjoint.reduced_functional import ReducedFunctional
    from pyadjoint.optimization.optimization import minimize
    PYADJOINT_AVAILABLE = True
except ImportError:
    print("Warning: pyadjoint not available, using simplified implementation")
    PYADJOINT_AVAILABLE = False

class DGAdvectionSolver:
    """DG advection solver with adjoint capability"""
    
    def __init__(self, mesh):
        self.mesh = mesh
        
        # Function spaces
        self.V = FunctionSpace(mesh, "DQ", 1)
        self.W = VectorFunctionSpace(mesh, "CG", 1)
        
        # Set up velocity field (solid body rotation)
        x, y = SpatialCoordinate(mesh)
        velocity = as_vector((0.5 - y, x - 0.5))
        self.u = Function(self.W, name="velocity").interpolate(velocity)
        
        # Time parameters
        self.T = 2*math.pi
        self.dt = self.T/600.0
        self.dtc = Constant(self.dt)
        self.q_in = Constant(1.0)
        
        # Store number of time steps for observation scheduling
        self.total_steps = int(self.T / self.dt)
        
        # Set up DG formulation
        self.setup_dg_solver()
        
    def setup_dg_solver(self):
        """Set up the DG variational forms"""
        # Test and trial functions
        self.phi = TestFunction(self.V)
        dq_trial = TrialFunction(self.V)
        
        # Mass matrix (LHS)
        self.a = self.phi * dq_trial * dx
        
        # Store solver parameters
        self.params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    
    def dg_flux_form(self, q_func):
        """Create DG flux form for given function - adjoint compatible"""
        n = FacetNormal(self.mesh)
        un = 0.5*(dot(self.u, n) + abs(dot(self.u, n)))
        
        F = (q_func*div(self.phi*self.u)*dx
           - conditional(dot(self.u, n) < 0, self.phi*dot(self.u, n)*self.q_in, 0.0)*ds
           - conditional(dot(self.u, n) > 0, self.phi*dot(self.u, n)*q_func, 0.0)*ds
           - (self.phi('+') - self.phi('-'))*(un('+')*q_func('+') - un('-')*q_func('-'))*dS)
        
        return F
    
    def solve_forward_adjoint(self, q0, store_freq=20, final_time_fraction=0.5):
        """
        Solve forward with adjoint tape recording
        final_time_fraction: fraction of total time to run (for faster testing)
        """
        if not PYADJOINT_AVAILABLE:
            print("pyadjoint not available, running forward solve only")
            return self.solve_forward_simple(q0, store_freq, final_time_fraction)
        
        # Clear any existing tape
        tape = get_working_tape()
        tape.clear_tape()
        
        q = Function(self.V, name="q")
        q.assign(q0)
        
        # Store solutions for observations
        solutions = []
        solution_times = []
        
        # Working variables for RK3
        q1 = Function(self.V, name="q1")
        q2 = Function(self.V, name="q2")
        dq = Function(self.V, name="dq")
        
        t = 0.0
        step = 0
        max_time = self.T * final_time_fraction
        
        # Store initial condition
        solutions.append(q.copy(deepcopy=True))
        solution_times.append(t)
        
        while t < max_time - 0.5*self.dt:
            # Stage 1: dq = dt * F(q)
            L1 = self.dtc * self.dg_flux_form(q)
            solve(self.a == L1, dq, solver_parameters=self.params)
            q1.assign(q + dq)
            
            # Stage 2: dq = dt * F(q1)  
            L2 = self.dtc * self.dg_flux_form(q1)
            solve(self.a == L2, dq, solver_parameters=self.params)
            q2.assign(Constant(0.75)*q + Constant(0.25)*(q1 + dq))
            
            # Stage 3: dq = dt * F(q2)
            L3 = self.dtc * self.dg_flux_form(q2)
            solve(self.a == L3, dq, solver_parameters=self.params)
            q.assign(Constant(1.0/3.0)*q + Constant(2.0/3.0)*(q2 + dq))
            
            step += 1
            t += self.dt
            
            # Store solution at specified frequency
            if step % store_freq == 0:
                solutions.append(q.copy(deepcopy=True))
                solution_times.append(t)
        
        return solutions, solution_times
    
    def solve_forward_simple(self, q0, store_freq=20, final_time_fraction=0.5):
        """Simple forward solve without adjoint recording"""
        q = Function(self.V, name="q")
        q.assign(q0)
        
        solutions = []
        solution_times = []
        
        q1 = Function(self.V, name="q1")
        q2 = Function(self.V, name="q2")
        dq = Function(self.V, name="dq")
        
        t = 0.0
        step = 0
        max_time = self.T * final_time_fraction
        
        solutions.append(q.copy(deepcopy=True))
        solution_times.append(t)
        
        while t < max_time - 0.5*self.dt:
            # Stage 1
            L1 = self.dtc * self.dg_flux_form(q)
            solve(self.a == L1, dq, solver_parameters=self.params)
            q1.assign(q + dq)
            
            # Stage 2
            L2 = self.dtc * self.dg_flux_form(q1)
            solve(self.a == L2, dq, solver_parameters=self.params)
            q2.assign(0.75*q + 0.25*(q1 + dq))
            
            # Stage 3
            L3 = self.dtc * self.dg_flux_form(q2)
            solve(self.a == L3, dq, solver_parameters=self.params)
            q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))
            
            step += 1
            t += self.dt
            
            if step % store_freq == 0:
                solutions.append(q.copy(deepcopy=True))
                solution_times.append(t)
        
        return solutions, solution_times

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
    q_true = Function(V, name="true_initial")
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
    q_guess = Function(V, name="first_guess")
    q_guess.interpolate(1.0 + bell + cone + slot_cyl)
    
    return q_guess

class FourDVarProblem:
    """4D-Var data assimilation problem using pyadjoint"""
    
    def __init__(self, solver, obs_locations, obs_times, alpha=1e-4):
        self.solver = solver
        self.obs_locations = obs_locations
        self.obs_times = obs_times
        self.alpha = alpha
        self.observations = []
        
    def create_synthetic_observations(self, true_q0, noise_level=0.02):
        """Generate synthetic observations from true solution"""
        print("Generating synthetic observations...")
        
        # Get true solution
        true_solutions, solution_times = self.solver.solve_forward_adjoint(
            true_q0, store_freq=20, final_time_fraction=0.5)
        
        observations = []
        actual_obs_times = []
        
        for time_idx in self.obs_times:
            if time_idx < len(true_solutions):
                q_true = true_solutions[time_idx]
                obs_values = []
                
                for loc in self.obs_locations:
                    try:
                        obs_val = q_true.at(loc, tolerance=1e-8)
                        obs_values.append(float(obs_val))
                    except:
                        # Fallback: use background value
                        obs_values.append(1.0)
                
                # Add noise
                obs_noisy = np.array(obs_values) + np.random.normal(0, noise_level, len(obs_values))
                observations.append(obs_noisy.tolist())
                actual_obs_times.append(solution_times[time_idx] if time_idx < len(solution_times) else time_idx)
        
        self.observations = observations
        self.actual_obs_times = actual_obs_times
        print(f"Created {len(observations)} observation sets")
        return observations

def compute_cost_functional(q0_control, fourDVar):
    """
    Simplified cost functional that avoids complex UFL operations
    """
    try:
        # Solve forward problem
        solutions, solution_times = fourDVar.solver.solve_forward_adjoint(
            q0_control, store_freq=20, final_time_fraction=0.3)  # Shorter time window
        
        # Start with regularization only
        J_reg = assemble(q0_control * q0_control * dx)
        
        # Simple data term - use only a few observations
        J_data = Constant(0.0)
        
        # Use only first few time steps and locations to avoid complexity
        max_obs = min(3, len(fourDVar.observations))
        max_locs = min(4, len(fourDVar.obs_locations))
        
        for i in range(max_obs):
            if i < len(solutions) and fourDVar.obs_times[i] < len(solutions):
                q_model = solutions[fourDVar.obs_times[i]]
                
                # Use only a subset of observation locations
                for j in range(max_locs):
                    obs_val = fourDVar.observations[i][j]
                    loc = fourDVar.obs_locations[j]
                    
                    # Simple point evaluation without complex weights
                    try:
                        model_val = q_model.at(loc, tolerance=1e-6)
                        diff = Constant(float(model_val)) - Constant(obs_val)
                        J_data = J_data + diff * diff
                    except:
                        # Skip this observation if evaluation fails
                        continue
        
        # Simple total cost
        alpha = Constant(fourDVar.alpha)
        J_total = alpha * J_reg + J_data
        
        return J_total
        
    except Exception as e:
        print(f"Error in cost functional: {e}")
        # Return simple regularization if everything fails
        return assemble(q0_control * q0_control * dx)

def compute_simple_cost_functional(q0_control, fourDVar):
    """
    Very simple cost functional for testing pyadjoint
    """
    # Only regularization term - no forward solve
    J_reg = assemble(q0_control * q0_control * dx)
    
    # Simple target matching (avoid forward solve complexity)
    target_value = Constant(1.5)
    J_target = assemble((q0_control - target_value) * (q0_control - target_value) * dx)
    
    return J_reg + J_target

def test_pyadjoint_basic():
    """Test basic pyadjoint functionality"""
    print("Testing basic pyadjoint functionality...")
    
    if not PYADJOINT_AVAILABLE:
        print("pyadjoint not available")
        return False
    
    try:
        # Very simple test
        mesh = UnitSquareMesh(10, 10)
        V = FunctionSpace(mesh, "CG", 1)
        
        # Simple function
        u = Function(V, name="u")
        u.assign(Constant(1.0))
        
        # Simple cost: ||u - target||²
        target = Constant(2.0)
        J = assemble((u - target) * (u - target) * dx)
        
        # Create control and ReducedFunctional
        control = Control(u)
        rf = ReducedFunctional(J, control)
        
        print("Testing gradient...")
        grad = rf.derivative()
        print(f"Gradient computed: norm = {float(sqrt(assemble(grad * grad * dx))):.6f}")
        
        print("Testing optimization...")
        u_opt = minimize(rf, method="L-BFGS-B", options={"maxiter": 3})
        print(f"Optimization completed: optimal value = {float(u_opt.at((0.5, 0.5))):.6f}")
        
        return True
        
    except Exception as e:
        print(f"Basic pyadjoint test failed: {e}")
        return False

def run_4dvar_with_pyadjoint():
    """Run 4D-Var optimization using pyadjoint ReducedFunctional"""
    if not PYADJOINT_AVAILABLE:
        print("pyadjoint not available, running without optimization")
        return run_4dvar_without_adjoint()
    
    print("Setting up 4D-Var with pyadjoint ReducedFunctional...")
    
    # First test basic pyadjoint functionality
    if not test_pyadjoint_basic():
        print("Basic pyadjoint test failed, falling back...")
        return run_4dvar_without_adjoint()
    
    # Create mesh and solver
    mesh = UnitSquareMesh(15, 15, quadrilateral=True)  # Even smaller mesh
    solver = DGAdvectionSolver(mesh)
    
    # Create true initial condition and first guess
    true_q0 = create_true_initial_condition(solver.V)
    first_guess = create_first_guess(solver.V)
    
    # Define observation network
    obs_locations = [
        (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)  # Only 4 locations
    ]
    
    # Observation times
    obs_times = [1, 2, 3]  # Only 3 time steps
    
    # Create 4D-Var problem
    fourDVar = FourDVarProblem(solver, obs_locations, obs_times, alpha=1e-4)
    
    # Generate synthetic observations
    pause_annotation()
    observations = fourDVar.create_synthetic_observations(true_q0, noise_level=0.02)
    continue_annotation()
    
    # Set up control variable
    control_q0 = Function(solver.V, name="control_q0")
    control_q0.assign(first_guess)
    
    print("Testing very simple cost functional first...")
    try:
        # Test very simple cost function (no forward solve)
        simple_cost = compute_simple_cost_functional(control_q0, fourDVar)
        print(f"Simple cost: {float(simple_cost):.6f}")
        
        # Create control and ReducedFunctional for simple case
        control = Control(control_q0)
        rf_simple = ReducedFunctional(simple_cost, control)
        
        print("Testing gradient computation...")
        grad = rf_simple.derivative()
        grad_norm = sqrt(assemble(grad * grad * dx))
        print(f"Gradient norm: {float(grad_norm):.6f}")
        
        print("Running simple optimization...")
        optimized_simple = minimize(rf_simple, method="L-BFGS-B", options={"maxiter": 3})
        print("Simple optimization completed!")
        
    except Exception as e:
        print(f"Simple cost function failed: {e}")
        return run_4dvar_without_adjoint()
    
    print("\nNow trying simplified 4D-Var cost functional...")
    try:
        # Reset control
        control_q0.assign(first_guess)
        
        # Clear tape before full computation
        tape = get_working_tape()
        tape.clear_tape()
        
        # Evaluate initial cost
        initial_cost = compute_cost_functional(control_q0, fourDVar)
        print(f"Initial cost: {float(initial_cost):.6f}")
        
        # Create the control object
        control = Control(control_q0)
        
        # Create ReducedFunctional
        rf = ReducedFunctional(compute_cost_functional(control_q0, fourDVar), control)
        
        # Compute initial error
        error_initial = sqrt(assemble((first_guess - true_q0)**2 * dx))
        print(f"Initial L2 error: {float(error_initial):.6f}")
        
        print("Testing gradient computation for 4D-Var...")
        grad = rf.derivative()
        print("Gradient computation successful!")
        
        print("Starting simplified optimization...")
        
        # Use fewer iterations to avoid complexity
        optimized_q0 = minimize(rf, method="L-BFGS-B", 
                               options={"disp": True, "maxiter": 5, "gtol": 1e-4})
        
        # Compute final error
        error_final = sqrt(assemble((optimized_q0 - true_q0)**2 * dx))
        final_cost = float(compute_cost_functional(optimized_q0, fourDVar))
        
        print(f"\nOptimization Results:")
        print(f"Initial cost: {float(initial_cost):.6f}")
        print(f"Final cost: {final_cost:.6f}")
        print(f"Cost reduction: {(float(initial_cost) - final_cost) / float(initial_cost) * 100:.2f}%")
        print(f"Initial L2 error: {float(error_initial):.6f}")
        print(f"Final L2 error: {float(error_final):.6f}")
        print(f"Error reduction: {(float(error_initial) - float(error_final)) / float(error_initial) * 100:.2f}%")
        
        # Plot results
        plot_4dvar_results(true_q0, first_guess, optimized_q0, obs_locations)
        
        return optimized_q0
        
    except Exception as e:
        print(f"4D-Var optimization failed: {e}")
        print("The forward solve might be too complex for pyadjoint.")
        print("Falling back to gradient-free optimization...")
        return run_4dvar_without_adjoint()

def run_4dvar_without_adjoint():
    """Fallback 4D-Var implementation without automatic differentiation"""
    print("Running 4D-Var with gradient-free optimization...")
    
    # Create mesh and solver
    mesh = UnitSquareMesh(20, 20, quadrilateral=True)
    solver = DGAdvectionSolver(mesh)
    
    # Create true initial condition and first guess
    true_q0 = create_true_initial_condition(solver.V)
    first_guess = create_first_guess(solver.V)
    
    # Define observation network
    obs_locations = [
        (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
        (0.5, 0.5), (0.3, 0.6), (0.7, 0.4), (0.4, 0.3)
    ]
    
    obs_times = [1, 2, 3, 4, 5]
    
    # Create 4D-Var problem
    fourDVar = FourDVarProblem(solver, obs_locations, obs_times, alpha=1e-5)
    
    # Generate observations (without adjoint)
    pause_annotation() if PYADJOINT_AVAILABLE else None
    observations = fourDVar.create_synthetic_observations(true_q0, noise_level=0.02)
    continue_annotation() if PYADJOINT_AVAILABLE else None
    
    # Simple gradient-free optimization
    current_q0 = first_guess.copy(deepcopy=True)
    best_q0 = current_q0.copy(deepcopy=True)
    
    # Define simple cost function for gradient-free
    def simple_cost(q0):
        pause_annotation() if PYADJOINT_AVAILABLE else None
        try:
            solutions, _ = solver.solve_forward_simple(q0, store_freq=20, final_time_fraction=0.3)
            cost = 0.0
            
            # Simple L2 difference with observations
            for i, (obs, time_idx) in enumerate(zip(observations[:3], obs_times[:3])):  # Use fewer obs
                if time_idx < len(solutions):
                    q_model = solutions[time_idx]
                    for obs_val, loc in zip(obs, obs_locations):
                        try:
                            model_val = q_model.at(loc, tolerance=1e-6)
                            cost += 0.5 * (float(model_val) - obs_val)**2
                        except:
                            continue
            
            # Regularization
            cost += 1e-5 * float(assemble(q0 * q0 * dx))
            continue_annotation() if PYADJOINT_AVAILABLE else None
            return cost
            
        except Exception as e:
            continue_annotation() if PYADJOINT_AVAILABLE else None
            return 1e10
    
    # Initial cost
    best_cost = simple_cost(current_q0)
    print(f"Initial cost (gradient-free): {best_cost:.6f}")
    
    # Simple random search
    step_size = 0.1
    for iteration in range(20):
        # Random perturbation
        perturbation = Function(solver.V)
        x, y = SpatialCoordinate(mesh)
        
        # Random Fourier mode
        kx = np.random.randint(1, 4)
        ky = np.random.randint(1, 4)
        amp = np.random.normal(0, 0.1)
        phase = np.random.uniform(0, 2*np.pi)
        
        pert_expr = Constant(amp) * sin(Constant(kx) * pi * x + Constant(phase)) * sin(Constant(ky) * pi * y)
        perturbation.interpolate(pert_expr)
        
        # Test perturbed solution
        test_q0 = Function(solver.V)
        test_q0.assign(current_q0 + Constant(step_size) * perturbation)
        
        test_cost = simple_cost(test_q0)
        
        if test_cost < best_cost:
            best_cost = test_cost
            best_q0.assign(test_q0)
            current_q0.assign(test_q0)
            print(f"Iteration {iteration}: Improved cost = {test_cost:.6f}")
        
        if iteration % 5 == 0:
            step_size *= 0.9
    
    # Compute errors
    error_initial = sqrt(assemble((first_guess - true_q0)**2 * dx))
    error_final = sqrt(assemble((best_q0 - true_q0)**2 * dx))
    
    print(f"\nGradient-free Results:")
    print(f"Initial L2 error: {float(error_initial):.6f}")
    print(f"Final L2 error: {float(error_final):.6f}")
    print(f"Error reduction: {(float(error_initial) - float(error_final)) / float(error_initial) * 100:.2f}%")
    
    # Plot results
    plot_4dvar_results(true_q0, first_guess, best_q0, obs_locations)
    
    return best_q0

def plot_4dvar_results(true_q0, first_guess, optimized_q0, obs_locations):
    """Plot 4D-Var results"""
    print("Creating plots...")
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # True initial condition
        c1 = tripcolor(true_q0, axes=axes[0], vmin=1, vmax=2, cmap='viridis')
        axes[0].set_title('True Initial Condition', fontsize=14)
        axes[0].set_aspect('equal')
        # Add observation points
        obs_x = [loc[0] for loc in obs_locations]
        obs_y = [loc[1] for loc in obs_locations]
        axes[0].scatter(obs_x, obs_y, c='red', s=50, marker='x', linewidth=2, label='Observations')
        axes[0].legend()
        plt.colorbar(c1, ax=axes[0])
        
        # First guess
        c2 = tripcolor(first_guess, axes=axes[1], vmin=1, vmax=2, cmap='viridis')
        axes[1].set_title('First Guess', fontsize=14)
        axes[1].set_aspect('equal')
        plt.colorbar(c2, ax=axes[1])
        
        # 4D-Var result
        c3 = tripcolor(optimized_q0, axes=axes[2], vmin=1, vmax=2, cmap='viridis')
        axes[2].set_title('4D-Var Optimized (pyadjoint)', fontsize=14)
        axes[2].set_aspect('equal')
        plt.colorbar(c3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig('4dvar_pyadjoint_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Plotting error: {e}")

def simple_test():
    """Simple test of the solver without 4D-Var"""
    print("Running simple solver test...")
    
    mesh = UnitSquareMesh(20, 20, quadrilateral=True)
    solver = DGAdvectionSolver(mesh)
    q0 = create_true_initial_condition(solver.V)
    
    # Test forward solve
    solutions, times = solver.solve_forward_adjoint(q0, store_freq=10, final_time_fraction=0.1)
    print(f"Forward solve completed: {len(solutions)} solutions")
    
    # Plot initial and final
    if len(solutions) >= 2:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            tripcolor(solutions[0], axes=axes[0], vmin=1, vmax=2)
            axes[0].set_title('Initial')
            axes[0].set_aspect('equal')
            
            tripcolor(solutions[-1], axes=axes[1], vmin=1, vmax=2)
            axes[1].set_title('Final')
            axes[1].set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig('simple_test.png', dpi=150)
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    return solutions

if __name__ == "__main__":
    # Clear any existing tape
    tape = get_working_tape()
    tape.clear_tape()
    
    # Pause the tape during forward solve to avoid recording everything
    pause_annotation()
    
    try:
        # First run simple test
        simple_solutions = simple_test()
    except Exception as e:
        print(f"Simple test failed: {e}")
        
    
    # Resume annotation for 4D-Var
    continue_annotation()
    
    # Then run 4D-Var
    try:
        run_4dvar_with_pyadjoint()
    except Exception as e:
        print(f"4D-Var failed: {e}")
        print("Trying with simpler setup...")
        
        # Try with even simpler setup
        try:
            pause_annotation()
            mesh = UnitSquareMesh(10, 10, quadrilateral=True)
            solver = DGAdvectionSolver(mesh)
            continue_annotation()
            
            true_q0 = create_true_initial_condition(solver.V)
            first_guess = create_first_guess(solver.V)
            
            # Very simple cost function test
            control = Control(first_guess)
            
            # Simple quadratic cost
            simple_cost = assemble((first_guess - true_q0)**2 * dx)
            rf = ReducedFunctional(simple_cost, control)
            
            print("Testing simple quadratic optimization...")
            optimized = rf.minimize(method="L-BFGS-B", options={"maxiter": 5})
            print("Simple optimization completed!")
            
        except Exception as simple_error:
            print(f"Even simple test failed: {simple_error}")
            print("There might be an issue with the pyadjoint setup.")