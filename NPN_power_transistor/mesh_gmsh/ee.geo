ylenC = 6;
ylendrift = 15;
ylenB = 6;
ylenE = 4;
xlenE = 50;
xlenB = 8;
xlen_medzera = 1;

yCd = 0+ylenC;
yBC = yCd+ylendrift;
yBE = yBC+ylenB;
ylen = yBE+ylenE;

xB_medzera1 = xlenB;
xBE1 = xB_medzera1+xlen_medzera;
xBE2 = xBE1 + xlenE;
xB_medzera2 = xBE2+xlen_medzera;
xlen = xB_medzera2+xlenB;

x_arc1 = xBE1 + ylenE;
x_arc2 = xBE2 - ylenE;

xB1_mid = xlenB/2;
xB2_mid = xlen-(xlenB/2);


drift_mid = yCd+0.5*ylendrift;

meshsize = 0.2*ylen;

//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {xlen, -0, 0, 1.0};
//+
Point(3) = {xlen, ylen, 0, 1.0};
//+
Point(4) = {0, ylen, 0, 1.0};
//+
Point(5) = {0, drift_mid, 0, 1.0};
//+
Point(6) = {xlen, drift_mid, 0, 1.0};
//+
Point(7) = {0, yCd, 0, 1.0};
//+
Point(8) = {xlen, yCd, 0, 1.0};
//+
Point(9) = {xBE1, ylen, 0, 1.0};
//+
Point(10) = {xBE2, ylen, 0, 1.0};
//+
Point(11) = {xBE2, yBE, 0, 1.0};
//+
Point(12) = {xBE1, yBE, 0, 1.0};
//+
Point(13) = {xBE1+ylenE, yBE, 0, 1.0};
//+
Point(14) = {xBE2-ylenE, yBE, 0, 1.0};
//+
Point(15) = {xBE1+ylenE, ylen, 0, 1.0};
//+
Point(16) = {xBE2-ylenE, ylen, 0, 1.0};
//+
Point(17) = {0, yBC, 0, 1.0};
//+
Point(18) = {xlen, yBC, 0, 1.0};
//+
Point(19) = {xB1_mid, ylen, 0, 1.0};
//+
Point(20) = {xB_medzera1, ylen, 0, 1.0};
//+
Point(21) = {xB_medzera2, ylen, 0, 1.0};
//+
Point(22) = {xB2_mid, ylen, 0, 1.0};
//+
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 8};
//+
Line(3) = {8, 6};
//+
Line(4) = {6, 18};
//+
Line(5) = {18, 3};
//+
Line(6) = {3, 22};
//+
Line(7) = {22, 21};
//+
Line(8) = {21, 10};
//+
Line(9) = {10, 16};
//+
Line(10) = {16, 15};
//+
Line(11) = {15, 9};
//+
Line(12) = {9, 20};
//+
Line(13) = {20, 19};
//+
Line(14) = {19, 4};
//+
Line(15) = {4, 17};
//+
Line(16) = {17, 5};
//+
Line(17) = {5, 7};
//+
Line(18) = {7, 1};
//+
Line(19) = {13, 14};
//+
Circle(20) = {9, 15, 13};
//+
Circle(21) = {14, 16, 10};
//+
Line(22) = {17, 18};
//+
Line(23) = {7, 8};
//+
Curve Loop(1) = {10, 11, 20, 19, 21, 9};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {12, 13, 14, 15, 22, 5, 6, 7, 8, -21, -19, -20};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {22, -4, -3, -23, -17, -16};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {23, -2, -1, -18};
//+
Plane Surface(4) = {4};
//+
Physical Surface("domain_E", 24) = {1};
//+
Physical Surface("domain_B", 25) = {2};
//+
Physical Surface("domain_drift", 26) = {3};
//+
Physical Surface("domain_C", 27) = {4};
//+
Physical Curve("bc_E", 28) = {11, 10, 9};
//+
Physical Curve("bc_B", 29) = {14, 13, 12, 8, 7, 6};
//Physical Curve("bc_B", 29) = {14, 13, 7, 6};
//+
Physical Curve("bc_C", 30) = {1};
//+
Field[1] = Attractor;
//+
Field[1].CurvesList = {19, 20, 21, 22, 14, 6, 23, 1};
//+
Field[2] = Threshold;
//+
Field[2].DistMax = 0.8*meshsize;
//+
Field[2].DistMin = 0.5*meshsize;
//+
Field[2].InField = 1;
//+
Field[2].SizeMax = 0.5*meshsize;
//+
Field[2].SizeMin = 0.1*meshsize;
//+
Background Field = 2;
