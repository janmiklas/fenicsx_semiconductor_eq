import numpy as np

Xa = 45
Xb = 319
Ya = 216.16
Yb = 15.9
xscale = 1
yscale = 2.5

xx = np.array([71.6, 97.5, 119, 122, 130, 151, 181, 220, 250, 282, 300])
#yy = np.array([209, 207, 218.5, 221.6, 197.6, 173, 139.5, 104.1, 79.8, 54.6, 40.8])
yy = np.array([208.9, 206.6, 205, 203.5, 198, 172.7, 139.6, 103.6, 80.3, 54.6, 40.7])
yyc = np.array([215.5, 215, 214.4, 213.5, 208.9, 188.8, 157.2, 122.3, 100.3, 75, 61.3])
yyb = np.array([209.4, 207.7, 206.5, 205.8, 204.5, 200, 198.2, 196.4, 195.9, 195.5, 195.1])

Ib = (xx-Xa)/(Xb-Xa)*xscale
Q = (yy-Ya)/(Yb-Ya)*yscale
Qc = (yyc-Ya)/(Yb-Ya)*yscale
Qb = (yyb-Ya)/(Yb-Ya)*yscale

import matplotlib.pyplot as plt
plt.plot(Ib, Q, '-o')
plt.plot(Ib, Qc, '-o')
plt.plot(Ib, Qb, '-o')
plt.ion()
plt.grid()
plt.show()

XXa = 44.95
XXb = 320.5
YYa = 216.4
YYb = 15.9
xxscale = 14
yyscale = 3.5

xxx = np.array([69.7, 95.8, 116.3, 135.5, 155, 178.1, 197.3, 216.9, 234.2, 258.3, 268.6, 276.7, 285.6])
yyy = np.array([32.9, 63.9, 98.1, 127.4, 149.2, 169, 181.9, 192.5, 199.6, 205.4, 206.8, 207.3, 207.7])
yyyc = np.array([39.4, 72, 105.6, 136.15, 159.2, 180.1, 193.7, 203.4, 209.4, 213.2, 214.7, 215.1, 215.5])
yyyb = np.array([210.1, 208.5, 209.4, 207.4, 206.7, 205.3, 204.1, 205.5, 206.4, 208.6, 209.6, 209.5, 209.7])

Ic = (xxx-XXa)/(XXb-XXa)*xxscale
QQ = (yyy-YYa)/(YYb-YYa)*yyscale
QQc = (yyyc-YYa)/(YYb-YYa)*yyscale
QQb = (yyyb-YYa)/(YYb-YYa)*yyscale

import matplotlib.pyplot as plt
plt.plot(Ic, QQ, '-o')
plt.plot(Ic, QQc, '-o')
plt.plot(Ic, QQb, '-o')
plt.ion()
plt.grid()
plt.show()

