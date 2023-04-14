from mpi4py import MPI
import gmsh
import numpy as np


import meshio
#msh = meshio.read("../mesh_jorgensd/mesh.msh")
msh = meshio.read("ee.msh")
cells = msh.get_cells_type("triangle")
cell_data  = msh.get_cell_data("gmsh:physical", "triangle")
#mesh_triangle = meshio.Mesh(points=msh.points, cells={"triangle":cells}, cell_data={"name_to_read":[cell_data]})
#msh.prune_z_0()
# PRUNE Z
mesh_triangle = meshio.Mesh(points=msh.points[:,:2], cells={"triangle":cells}, cell_data={"name_to_read":[cell_data]})
cells = msh.get_cells_type("line")
cell_data  = msh.get_cell_data("gmsh:physical", "line")
mesh_line = meshio.Mesh(points=msh.points, cells={"line":cells}, cell_data={"name_to_read":[cell_data]})
meshio.write("mesh.xdmf", mesh_triangle)
meshio.write("facets.xdmf", mesh_line)

import dolfinx.io
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
ft = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "facets.xdmf", "r").read_meshtags(mesh, name="Grid")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
from myplot import myplot_2d
myplot_2d(mesh, ax, savename="aa.png")

# domains tags: domain_p:8, domain_n:9
# BCs tags: bc_p:10, bc_n:11

#mesh=mesh
