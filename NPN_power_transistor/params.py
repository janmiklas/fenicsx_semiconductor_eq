
#################################3
### EDIT
#################################

#Psi1=+1e-4 # forward
#Psi1=0 # forward
#Psi1_orig = -.075 #
#Psi1 = -6.5

#########
Psi1_orig = 0
Psi2_orig = -.3 #reverse
Psi2_orig = .2 #
Psi3_orig = .03
#########
Psi1_orig = 0
Psi2_orig = 5 #reverse: - 
Psi3_orig = 20
Psi3_orig = 0
#########
X_norma = .005 * 1e-6 * 100 # mikron -> m -> cm
#########
#drift
Nd3_orig = 1e15 # N type
Na3_orig = 0e17 # 
#C
#Nd4_orig = 5*Nd3_orig # N type
Nd4_orig = 5e18 # N type
Na4_orig = 0e17 # 
#B
Na2_orig =  5e17 # P type
Nd2_orig = Nd3_orig 
#E
Nd1_orig = 5e18 # N type
Na1_orig = Na2_orig # 



ni_orig = 1e10

###########

q =  1.602e-19
#eps0 = 8.854e-12 * 1/m2cm # [F / m] * 1/m2cm
eps0 =  8.854e-14 # [F/cm]
epsr_Si = 11.68
eps_Si = eps0 * epsr_Si

Vth = 0.0259 # k.T/q

mob_n_orig = 1*1400 # [cm2 / V / s]
mob_p_orig = 450
D_p_orig = 1*Vth*mob_p_orig
D_n_orig = 1*Vth*mob_n_orig
#D_n = .6# 36 # [cm2 / s]
#D_p = .6# 12
Tau_p_orig = .001e-9 # s
Tau_n_orig = .002e-9 # s
Tau_p_orig = .005e-9 # s
Tau_n_orig = .010e-9 # s

#########

#X_norma = .3e-5 * 100 # m -> cm
Psi_norma = 0.0259 # k.T/q
N_norma = max(Na1_orig, Nd1_orig, Na2_orig, Nd2_orig)
#N_norma = ni_orig
#N_norma = N_norma*1e2
mob_norma = max(mob_p_orig, mob_n_orig)
D_norma = Psi_norma * mob_norma

t_norma = X_norma**2/(Psi_norma*mob_norma)

Na1 =  Na1_orig / N_norma
Na2 =  Na2_orig / N_norma
Na3 =  Na3_orig / N_norma
Na4 =  Na4_orig / N_norma
Nd1 =  Nd1_orig / N_norma
Nd2 =  Nd2_orig / N_norma
Nd3 =  Nd3_orig / N_norma
Nd4 =  Nd4_orig / N_norma
ni = ni_orig / N_norma

#Psi1 =  Psi1_orig / Psi_norma
#Psi2 =  Psi2_orig / Psi_norma
#Psi3 =  Psi3_orig / Psi_norma
Psi1 =  Psi1_orig / 1
Psi2 =  Psi2_orig / 1
Psi3 =  Psi3_orig / 1
mob_p = mob_p_orig / mob_norma
mob_n = mob_n_orig / mob_norma
D_p = Dp = D_p_orig / D_norma
D_n = Dn = D_n_orig / D_norma
Tau_n = Tau_n_orig/t_norma
Tau_p = Tau_p_orig/t_norma
#########

lmda0 = eps_Si * Psi_norma / (X_norma**2 * q * N_norma)
#lmda1 = N_norma * Psi_norma / X_norma**2 * mob_p
#lmda2 = N_norma * Psi_norma / X_norma**2 * mob_n
lmda1 = N_norma * D_norma / X_norma**2
lmda2 = N_norma * D_norma / X_norma**2

#########
