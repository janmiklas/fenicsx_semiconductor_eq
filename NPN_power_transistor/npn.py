import gmsh
import numpy as np
#import pyvista
import matplotlib.pyplot as plt
#from myplot import myplot_2d pltsubdomains
import myplot
import os
from petsc4py import PETSc

from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                                 form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                         dx, grad, inner, div,
                         FiniteElement, VectorElement, MixedElement, TestFunctions, TrialFunctions)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

import ufl

from params import *

EPS = np.spacing(1)

############
### MESH ###

import dolfinx.io
meshpath = "mesh_gmsh"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, meshpath+"/mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
ft = dolfinx.io.XDMFFile(MPI.COMM_WORLD, meshpath+"/facets.xdmf", "r").read_meshtags(mesh, name="Grid") # boundaries

### z generovania meshu:
# dif:
tag_domain_E = 24
tag_domain_B = 25
tag_domain_drift = 26
tag_domain_C = 27
tag_bc_E = 28
tag_bc_B = 29
tag_bc_C = 30

#######################
### Function Spaces ###
#######################

elem_poiss = FiniteElement("CG", mesh.ufl_cell(), 1)
elem_cont = FiniteElement("CG", mesh.ufl_cell(), 1)
elem_mix = MixedElement([elem_poiss, elem_cont, elem_cont])
V = FunctionSpace(mesh, elem_mix)
#
(v0, v1, v2) = TestFunctions(V)
#q = Function(V)
#u, y = split(q)
#alebo? TrialFunctions(

u = Function(V)
#(u0, u1, u2) = (Psi, p, n) = (Psi, p, n) = u.split() ### TOTO by sa mi pozdavalo viac, ale neskonvergoval Newton solver!
(u0, u1, u2) = (Psi, p, n) = (Psi, p, n) =  ufl.split(u) # po tomto v pohode skonvergoval
#u = TrialFunction(V)
#u0 = Psi = u[0]
#u1i = p = u[1]
#u2 = n = u[2]
#(u0, u1, u2) = (Psi, p, n) = TrialFunctions(V)

###########################
### Discontinuous kappa ###
###########################


Q = FunctionSpace(mesh, ("DG", 0))
#kappa = Function(Q)
cells_E = ct.indices[ct.values==tag_domain_E]
cells_B = ct.indices[ct.values==tag_domain_B]
cells_drift = ct.indices[ct.values==tag_domain_drift]
cells_C = ct.indices[ct.values==tag_domain_C]
#kappa.x.array[bottom_cells] = np.full(len(bottom_cells), 5)

def materialy(hodnota1, hodnota2, hodnota3, hodnota4):
     kappa = Function(Q)
     kappa.x.array[cells_E] = np.full(len(cells_E), hodnota1)
     kappa.x.array[cells_B] = np.full(len(cells_B), hodnota2)
     kappa.x.array[cells_drift] = np.full(len(cells_drift), hodnota3)
     kappa.x.array[cells_C] = np.full(len(cells_C), hodnota4)
     return kappa

#kappa = materialy(1, .1+EPS)
Nd = materialy(Nd1, Nd2, Nd3, Nd4)
Na = materialy(Na1, Na2, Na3, Na4)
Ndoping = materialy(Nd1-Na1, Nd2-Na2, Nd3-Na3, Nd4-Na4)
Nsubdomains = materialy(1, 2, 3, 4)

###########################
### Boundary Conditions ###
###########################
tag=tag_bc_E
facets = facets_bc_E = ft.indices[ft.values==tag]
dofs0 = dofs0_bc_E = locate_dofs_topological(V, mesh.topology.dim-1, facets)
dofs1 = dofs1_bc_E = locate_dofs_topological(V.sub(1), mesh.topology.dim-1, facets)
dofs2 = locate_dofs_topological(V.sub(2), mesh.topology.dim-1, facets)
bc_psi_E = dirichletbc(ScalarType(Psi1), dofs0, V.sub(0))
bc_p_E = dirichletbc(ScalarType(0), dofs1, V.sub(1))
bc_n_E = dirichletbc(ScalarType(Nd1-Na1), dofs2, V.sub(2))

tag=tag_bc_B
facets = facets_bc_B = ft.indices[ft.values==tag]
dofs0 = dofs0_bc_B = locate_dofs_topological(V, mesh.topology.dim-1, facets)
dofs1 = dofs1_bc_B = locate_dofs_topological(V.sub(1), mesh.topology.dim-1, facets)
dofs2 = locate_dofs_topological(V.sub(2), mesh.topology.dim-1, facets)
bc_psi_B = dirichletbc(ScalarType(Psi2), dofs0, V.sub(0))
bc_p_B = dirichletbc(ScalarType(Na2-Nd2), dofs1, V.sub(1))
bc_n_B = dirichletbc(ScalarType(0), dofs2, V.sub(2))

tag=tag_bc_C
facets = facets_bc_C = ft.indices[ft.values==tag]
dofs0 = dofs0_bc_C = locate_dofs_topological(V, mesh.topology.dim-1, facets)
dofs1 = locate_dofs_topological(V.sub(1), mesh.topology.dim-1, facets)
dofs2 = locate_dofs_topological(V.sub(2), mesh.topology.dim-1, facets)
bc_psi_C = dirichletbc(ScalarType(Psi3), dofs0, V.sub(0))
bc_p_C = dirichletbc(ScalarType(0), dofs1, V.sub(1))
bc_n_C = dirichletbc(ScalarType(Nd4-Na4), dofs2, V.sub(2))


##bc_p_n = dirichletbc(ScalarType(Na2), dofs1, V.sub(1))
#bc_p_n = dirichletbc(ScalarType(ni**2/Nd2), dofs1, V.sub(1))
#bc_n_n = dirichletbc(ScalarType(Nd2), dofs2, V.sub(2))
#bc_p_p = dirichletbc(ScalarType(Na1), dofs1, V.sub(1))
##bc_n_p = dirichletbc(ScalarType(Nd1), dofs2, V.sub(2))
#bc_n_p = dirichletbc(ScalarType(ni**2/Na1), dofs2, V.sub(2))

ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)  
dS = ufl.Measure('dS', domain=mesh, subdomain_data=ft)  
dxx = ufl.Measure('dx', domain=mesh, subdomain_data = ct)

###########
## initial:
###########
#n_i1.interpolate(Nd)
#p_i1.interpolate(Na)
#space0, map0 = V.sub(0).collapse(collapsed_dofs=True)
#space1, map1 = V.sub(1).collapse(collapsed_dofs=True)
#space2, map2 = V.sub(2).collapse(collapsed_dofs=True)

#p_.interpolate(Na)
#n_.interpolate(Nd)
#resp. rovno:
u.sub(1).interpolate(Na)
u.sub(2).interpolate(Nd)
# bez pociatocneho  oodhadu sa prechod roztahoval na hocijaku dlzku struktury a poissonova rovnica vyriesila tomu prisluchajuce aj obrovske napatie

###############
### POISSON ###
###############

#a = inner(-kappa*grad(u), grad(v)) * dx
a0 = inner(grad(Psi), grad(v0)) * dx ### POZOR tuto je to minus nejake pofiderne, ale s nim to funguje. mozno zalezi od znamienka elementarneho naboja
#L = Constant(mesh, ScalarType(10)) * v * dx
L0 = 1/lmda0 * (Nd-Na-n+p + EPS) * v0 * dx # EPS aby tam nevznikla nula
#L0 = -1* (Nd-Na-n+p + EPS) * v0 * dx # EPS aby tam nevznikla nula

a0 = inner(grad(Psi), grad(v0)) * dx - 1/lmda0 * (Nd-Na-n+p + EPS) * v0 * dx 
L0 = 0

nn=ufl.FacetNormal(mesh)
### Neumannova BC
# pre emitor:
J = 0e-10
#g = -(J+grad(p)[0])/p/mob_p
#g = -(Dp*inner(grad(p), nn) - inner(J, nn))/(p*mob_p) # ked J by bol vektor
g = (-Dp*inner(grad(p), nn) - J)/(p*mob_p) # ked J je uz jedno cislo
L0g = - g * v0 * ds(tag_bc_C)

####################
## n, p Continuity
####################
znamienka = 1
if znamienka:
    #a1 = p*v1*dx + dt*D_p*inner(grad(p), grad(v1))*dx
    #L1 = p_*v1*dx - dt*mob_p*p_*inner(grad(Psi_), grad(v1))*dx
    #a1 = -D_p * inner(grad(p), grad(v1)) * dx
    #L1 = mob_p * p * inner(grad(Psi), grad(v1))*dx
    a1 = -D_p * inner(grad(p), grad(v1)) * dx - mob_p * p * inner(grad(Psi), grad(v1))*dx
    L1=0

###################
if znamienka:
    #a2 = n*v2*dx + dt*D_n*inner(grad(n), grad(v2))*dx
    #L2 = n_*v2*dx + dt*mob_n*n_*inner(grad(Psi_), grad(v2))*dx
    #a2 = -D_n * inner(grad(n), grad(v2)) * dx
    #L2 = -mob_n * n * inner(grad(Psi), grad(v2)) * dx
    a2 = -D_n * inner(grad(n), grad(v2)) * dx + mob_n * n * inner(grad(Psi), grad(v2)) * dx
    L2 = 0

###################
print("dalej")
# equillibrium:
print("equillibrium:")
bcs = [bc_psi_E, bc_n_E, bc_p_B, bc_p_C, bc_n_C]
bcs = [bc_psi_E, bc_n_E, bc_p_E, bc_p_B, bc_n_B, bc_p_C, bc_n_C]
#problem = linearproblem(a, l, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
###
F=a0+a1+a2-L0-L1-L2
problem = dolfinx.fem.petsc.NonlinearProblem(F, u=u, bcs = bcs)
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
aa=solver.solve(u)
#print(aa)

### pokus o rozbeh :
Na2_list = np.arange(1, .2, -.2)*Na2*3
#Na = materialy(Na1, Na2_list[0], Na3, Na4)
#u.sub(1).interpolate(Na)
from myplot import myplot2Dmoj, pltsubdomains
from myploteeict import plotting, plotting_Psipn, plotting_J, plotting1D
#for Na2_ in Na2_list:
if False:
    print(Na2_, Na2)
    Na = materialy(Na1, Na2_, Na3, Na4)
    bc_p_B = dirichletbc(ScalarType(Na2_-Nd2), dofs1_bc_B, V.sub(1))
    bc_n_E = dirichletbc(ScalarType(Nd1-Na2_), dofs1_bc_E, V.sub(2))
    bcs = [bc_psi_E, bc_n_E, bc_p_E, bc_p_B, bc_n_B, bc_p_C, bc_n_C]
    F=a0+a1+a2-L0-L1-L2
    problem = dolfinx.fem.petsc.NonlinearProblem(F, u=u, bcs = bcs)
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    aa=solver.solve(u)
    print(aa)
    pltret = plotting(u, Nd, Na, savename=f'aa')
###

#quickplot()
pltret = plotting(u, Nd, Na, savename=f'aa_equillibrium')
#myplot2Dmoj(u.sub(0), savename=f'plt2D_aa_equillibrium')
#myplot2Dmoj(u.sub(1), savename=f'plt2D_p_equillibrium')
#myplot2Dmoj(u.sub(2), savename=f'plt2D_n_equillibrium')
#plotting_J(u, savename=f"plt2D_J_equillibrium")
#pltsubdomains(Ndoping, levels=[Nd1-Na1, Nd2-Na2, Nd3-Na3, Nd4-Na4], savename=f'plt2D_doping')
pltsubdomains(Nsubdomains, levels=[1, 2, 3, 4], savename=f'subdomains')

Vbi = pltret["uu"][-1]
print(f"Vbi = {Vbi:.1f} = {Psi_norma*Vbi:.2f}V")


####################
## RECOMBINATION ##
####################
u_eq = Function(V)
psi_eq, p_eq, n_eq = u_eq.split()
p_eq.interpolate(u.sub(1))
n_eq.interpolate(u.sub(2))

RG = (ni**2-n*p) / (Tau_p*(n+n_eq) + Tau_n*(p+p_eq))
#RGp = -(ni**2-nRG*pRG) / (Taup*(nRG+n_eq) + Taun*(pRG+p_eq)) *v1*dx # typicky model
RGp = -RG*v1*dx
RGn =  -RG*v2*dx

FRG=a0+a1+a2-L0-L1-L2-RGp-RGn
problem = dolfinx.fem.petsc.NonlinearProblem(FRG, u=u, bcs = bcs)
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
aa=solver.solve(u)
print(aa)
pltret = plotting(u, Nd, Na, savename=f'aa_equillibrium_RG')
#plotting_J(u, savename=f"plt2D_J_equillibrium_RG")

####################


####################
elem_vect = VectorElement("CG", mesh.ufl_cell(), 1)
VV = FunctionSpace(mesh, elem_vect)
Jp = -u.sub(1)*mob_p*ufl.grad(u.sub(0)*Vth) - Vth*mob_p*ufl.grad(u.sub(1))
Jn = -u.sub(2)*mob_n*ufl.grad(u.sub(0)*Vth) + Vth*mob_n*ufl.grad(u.sub(2))
Jp_proj = myplot.project(Jp, VV)
Jn_proj = myplot.project(Jn, VV)

fig5 = plt.figure()
ax50 = fig5.add_subplot(211)
ax51 = fig5.add_subplot(212)
#myplot.mystreamplot(Jp_proj, ax=ax50)
#myplot.mystreamplot(Jn_proj, ax=ax51)
#fig5.savefig("bb")
####################
        
# Integral:
In_E = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jn_proj.sub(1) * ds(tag_bc_E)))
Ip_E = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jp_proj.sub(1) * ds(tag_bc_E)))
In_C = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jn_proj.sub(1) * ds(tag_bc_C)))
Ip_C = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jp_proj.sub(1) * ds(tag_bc_C)))
print(f"In_E = {In_E}")
print(f"Ip_E = {Ip_E}")
print(f"In_C = {In_C}")
print(f"Ip_C = {Ip_C}")


#######
#plotting(savename="aa.png")
#Psi1_eq = pltret["uu"][0]
Psi2_list = np.arange(0, 5, 1)*1e1
#Psi3_list = np.arange(-9, 43, 3)
#Psi3_list = np.array([0,1, 3,6,10,15,20,25,35,45,60, 80, 120, 150, 180, 210, 250, 300, 350])*1
Psi3_list = np.array([0,1, 3,6,10,15,20,25,35,45,60, 80, 120, 150])*1
J_list = []
V_list = []
In_E_list = []
In_B_list = []
In_C_list = []
Ip_E_list = []
Ip_B_list = []
Ip_C_list = []
JRG_list = []
naboj_Ib = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
naboj_Ic = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
naboj_n_B = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
naboj_n_drift = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
naboj_p_B = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
naboj_p_drift = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
pltret_list = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
IV_I_list = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
IV_V_list = np.empty(shape=(len(Psi2_list), len(Psi3_list)), dtype='object')
figiv, (axiv1, axiv2, axiv3) = plt.subplots(1, 3)

from data_polsko import Ic as naboj_Ic_polsko
from data_polsko import QQb as naboj_B_polsko
from data_polsko import QQc as naboj_C_polsko
from data_polsko import QQ as naboj_tot_polsko

if not os.path.isdir("results"):
    os.system("mkdir results")
#if False:
ii=0
jjj=-1
for Psi2 in Psi2_list:
#if False:
    jj=list(Psi2_list).index(Psi2)
    jjj+=1
    bc_psi_B = dirichletbc(ScalarType(Psi2), dofs0_bc_B, V.sub(0))
    for iii in range(len(Psi3_list)):
        if True:
            if jjj%2 == 0:
                ii=iii
            else:
                ii = len(Psi3_list)-iii-1
        Psi3 = Psi3_list[ii]

        print(f"bias: Psi2={Psi2}, Psi3 = {Psi3}")
        bc_psi_C = dirichletbc(ScalarType(Psi3), dofs0_bc_C, V.sub(0))
        #bcs = [bc_psi_E, bc_psi_B, bc_psi_C, bc_n_E, bc_p_B, bc_n_B, bc_p_C, bc_n_C]
        bcs = [bc_psi_E, bc_psi_B, bc_psi_C, bc_p_B, bc_n_B, bc_p_C, bc_n_C, bc_n_E]
        problem = dolfinx.fem.petsc.NonlinearProblem(F, u=u, bcs = bcs)
        solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
        aa=solver.solve(u)
        print(aa)
        
        print(f"plotting {ii}, iii:{iii}, ub{jj:02d}_uc{ii:02d}")
        #pltret = plotting(u, Nd, Na, IV=[V_list,J_list], savename=f'results/aa__ub{jj:02d}_uc{iii:02d}')
        pltret = plotting1D(u, Nd, Na, savename=f'results/bb__ub{jj:02d}_uc{iii:02d}')


        # ii: Psi3, jj: Psi2
        #if jj==4 and ii==2:
        #if jj==4 or ii==2:
        if False:
            #myplot2Dmoj(u.sub(0), savename=f'results/plt2D_Psi_ub{jj:02d}_uc{ii:02d}')
            #myplot2Dmoj(u.sub(1), savename=f'results/plt2D_p_ub{jj:02d}_uc{ii:02d}')
            #myplot2Dmoj(u.sub(2), savename=f'results/plt2D_n_ub{jj:02d}_uc{ii:02d}')
            plotting_J(u, subdomain_levels=[1, 2, 3, 4], subdomain_obj=Nsubdomains, savename=f"results/plt2D_J_ub{jj:02d}_uc{ii:02d}")

        #if ii%4 == 0:
        #if False:
        if ii == 4:
        #if jj==2 or ii==4:
        #if ii == 4 or ii==1:
            plotting_Psipn(u, subdomains_levels=[1, 2, 3, 4], subdomains_obj=Nsubdomains, savename=f"results/plt2D_Psipn_ub{jj:02d}_uc{ii:02d}")
            plotting_J(u, subdomains_levels=[1, 2, 3, 4], subdomains_obj=Nsubdomains, savename=f"results/plt2D_J_ub{jj:02d}_uc{ii:02d}")

        V_list.append(Psi3)
        Jp = -u.sub(1)*mob_p*ufl.grad(u.sub(0)*Vth) - Vth*mob_p*ufl.grad(u.sub(1))
        Jn = -u.sub(2)*mob_n*ufl.grad(u.sub(0)*Vth) + Vth*mob_n*ufl.grad(u.sub(2))
        Jp_proj = myplot.project(Jp, VV)
        Jn_proj = myplot.project(Jn, VV)
        # Integral:
        Ip_E = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jp_proj.sub(1) * ds(tag_bc_E)))
        Ip_B = -dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jp_proj.sub(1) * ds(tag_bc_B)))
        Ip_C = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jp_proj.sub(1) * ds(tag_bc_C)))
        Ip_C = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jp_proj.sub(1) * ds(tag_bc_C)))
        #Ip_Cnn = dolfinx.fem.assemble_scalar(dolfinx.fem.form(inner(Jp_proj,nn) * ds(tag_bc_C)))
        Ip_E_list.append(Ip_E)
        Ip_B_list.append(Ip_B)
        Ip_C_list.append(Ip_C)
        #Ip_Cnn_list.append(Ip_Cnn)
        In_E = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jn_proj.sub(1) * ds(tag_bc_E)))
        In_B = -dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jn_proj.sub(1) * ds(tag_bc_B)))
        In_C = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jn_proj.sub(1) * ds(tag_bc_C)))
        #In_Cnn = dolfinx.fem.assemble_scalar(dolfinx.fem.form(inner(Jp_proj,nn) * ds(tag_bc_C)))
        In_E_list.append(In_E)
        In_B_list.append(In_B)
        In_C_list.append(In_C)
        #print(f"In_E = {In_E}, In_B = {In_B}, In_C = {In_C}")
        #print(f'{ii}. Psi1:{Psi1_list[ii]:.2f}, {ii}. J:{J_list[ii]:.3f}')

        # Naboje
        from dolfinx.fem import assemble_scalar, form
        naboj_Ib[jj][ii] = Ip_B+In_B
        naboj_Ic[jj][ii] = Ip_C+In_C
        naboj_n_B[jj][ii] = assemble_scalar(form(u.sub(2) * dxx(tag_domain_B)))
        naboj_p_drift[jj][ii] = assemble_scalar(form(u.sub(1) * dxx(tag_domain_drift)))

        pltret_list[jj][ii] = pltret

        plt.clf()

        figiv, (axiv1, axiv2, axiv3) = plt.subplots(1, 3)
        #Psi_norma=1
        #ax4.plot(V_list, J_list, ".-")
        #ax4.grid()
        #fig2.savefig("IV")
        #axiv1.cla()
        axiv1.plot(V_list, Ip_E_list, ".-g")
        axiv1.plot(V_list, In_E_list, ".-b")
        axiv1.plot(V_list, np.array(Ip_E_list)+np.array(In_E_list), ".-r")
        #axiv2.cla()
        axiv2.plot(V_list, Ip_B_list, ".-g")
        axiv2.plot(V_list, In_B_list, ".-b")
        axiv2.plot(V_list, np.array(Ip_B_list)+np.array(In_B_list), ".-k")
        #axiv3.cla()
        axiv3.plot(V_list, Ip_C_list, ".-g")
        axiv3.plot(V_list, In_C_list, ".-b")
        axiv3.plot(V_list, np.array(Ip_C_list)+np.array(In_C_list), ".-k")
        axiv1.grid()
        axiv2.grid()
        axiv3.grid()
        figiv.savefig("IV_3")
        #figiv.clf()
        ####
        fig, ax = plt.subplots(1, 1, figsize = (6,4))
        #ax.plot(V_list, In_C_list, ".-b")
        #ax.plot(V_list, Ip_C_list, ".-g")
        ax.plot(Psi_norma*np.array(V_list), np.array(Ip_C_list)+np.array(In_C_list), ".-k")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Collector - emitter Voltage (V)")
        ax.set_ylabel("Collector current (A)")
        ax.grid()
        plt.tight_layout()
        fig.savefig("IV_Ic")
        '''
        fig, ax = plt.subplots(1, 1)
        ax.plot(V_list, In_B_list, ".-b")
        ax.plot(V_list, Ip_B_list, ".-g")
        ax.plot(V_list, np.array(Ip_B_list)+np.array(In_B_list), ".-k")
        fig.savefig("IV_Ib")
        fig, ax = plt.subplots(1, 1)
        ax.plot(V_list, In_E_list, ".-b")
        ax.plot(V_list, Ip_E_list, ".-g")
        ax.plot(V_list, np.array(Ip_E_list)+np.array(In_E_list), ".-k")
        fig.savefig("IV_Ie")
        '''

        figq, (axq1, axq2) = plt.subplots(1,2,figsize=(6,4))
        axq1.plot(naboj_Ic[jj], naboj_n_B[jj], '-o', label="Base")
        axq1.plot(naboj_Ic[jj], naboj_p_drift[jj], '-og', label="Collector")
        axq2.plot(naboj_Ic_polsko[4:], naboj_B_polsko[4:], '-o', label="Base")
        axq2.plot(naboj_Ic_polsko[4:], naboj_C_polsko[4:], '-og', label="Collector")
        axq1.grid()
        axq2.grid()
        axq2.set_ylim(top=1.2)
        axq1.set_xlim(left=0)
        #axq1.legend()
        axq2.legend()
        axq1.set_title("Simulation")
        axq2.set_title("Measurement")
        axq1.set_xlabel("Collector current (A)")
        axq2.set_xlabel("Collector current (A)")
        axq1.set_ylabel("Excess charge ($\mu$C)")
        plt.tight_layout(h_pad=.2)
        figq.savefig(f'QvsIc_ub{jj}')
    #Psi3_list = Psi3_list[::-1] # odzadu
        
        fig1d, (ax1d1, ax1d2) = plt.subplots(1,2,figsize=(6,4))
        axq1.plot(naboj_Ic[jj], naboj_n_B[jj], '-o')
        axq1.plot(naboj_Ic[jj], naboj_p_drift[jj], '-o')
        axq1.grid()
        #figq.savefig(f'QvsIc_ub{jj}')

from data_polsko import Ib as naboj_Ib_polsko
from data_polsko import Qb as naboj_B_polsko
from data_polsko import Qc as naboj_C_polsko
from data_polsko import Q as naboj_tot_polsko
figq, (axq1, axq2) = plt.subplots(1,2,figsize=(6,3))
for ii in range(len(naboj_Ib[0])):
    axq1.plot(naboj_Ib[:,ii], naboj_n_B[:,ii], '-o')
    axq1.plot(naboj_Ib[:,ii], naboj_p_drift[:,ii], '-o')
    axq1.plot(naboj_Ib[:,ii], naboj_p_drift[:,ii]+naboj_n_B[:,ii], '-o')
    axq2.plot(naboj_Ib_polsko, naboj_B_polsko, '--o')
    axq2.plot(naboj_Ib_polsko, naboj_C_polsko, '--o')
    axq2.plot(naboj_Ib_polsko, naboj_tot_polsko, '--o')
    axq1.grid()
figq.savefig(f'QvsIb_uc')



if False:
    plt.clf()
    #Psi_norma=1
    plt.plot(Psi1_list*Psi_norma, J_list, ".-")
    plt.plot((Psi1_list-Psi1_eq)*Psi_norma, J_list, ".-b")
    if False:
        plt.plot(Psi1_list*Psi_norma, JRG_list, ".-r")
        plt.plot((Psi1_list-Psi1_eq)*Psi_norma, JRG_list, ".-r")
    plt.vlines([-Psi1_eq*Psi_norma], 0, max(J_list), colors='k')
    plt.grid()
    plt.savefig("IV")
    #ax=myplot.9yplot2Dmoj(u.sub(1), shading="flat")
    #ax.set_title("diery")
    #plt.savefig("aa_2D.png")
