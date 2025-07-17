from firedrake import *
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
L = inner(f, v) * dx
u = Function(V)
solve(a == L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})
VTKFile("helmholtz.pvd").write(u)

# This file can be visualised using `paraview <http://www.paraview.org/>`__.
#
# We could use the built-in plotting functions of firedrake by calling
# :func:`tripcolor <firedrake.pyplot.tripcolor>` to make a pseudo-color plot.
# Before that, matplotlib.pyplot should be installed and imported::

try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")

try:
  from firedrake.pyplot import tripcolor, tricontour
  fig, axes = plt.subplots()
  colors = tripcolor(u, axes=axes)
  fig.colorbar(colors)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)

# The plotting functions in Firedrake mimic those of matplotlib; to produce a
# contour plot instead of a pseudocolor plot, we can call
# :func:`tricontour <firedrake.pyplot.tricontour>` instead::

try:
  fig, axes = plt.subplots()
  contours = tricontour(u, axes=axes)
  fig.colorbar(contours)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)

# Don't forget to show the image::

try:
  plt.savefig("solution.png", dpi=150)
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)

# Alternatively, since we have an analytic solution, we can check the
# :math:`L_2` norm of the error in the solution::

f.interpolate(cos(x*pi*2)*cos(y*pi*2))
print(sqrt(assemble(dot(u - f, u - f) * dx)))

# A python script version of this demo can be found :demo:`here <helmholtz.py>`.
