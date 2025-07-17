from firedrake import *

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx
L = Constant(0.5) * v * dx
u_sol = Function(V)
solve(a == L, u_sol)

print("✅ Firedrake test passed, solution preview:", u_sol.dat.data[:5])
