ylenC = 2;
ylendrift = 10;
ylenB = 10;
ylenE = 2;
xlen = .5;

yCd = 0+ylenC;
yBC = yCd+ylendrift;
yBE = yBC+ylenB;
ylen = yBE+ylenE;

drift_mid = yCd+0.5*ylendrift;

//meshsize = 0.2*ylen;
meshsize = 1;

//+
Point(1) = {0, 0, 0, meshsize};
//+
Point(2) = {xlen, 0, 0, meshsize};
//+
Point(3) = {xlen, ylen, 0, meshsize};
//+
Point(4) = {0, ylen, 0, meshsize};
//+
Point(5) = {0, yCd, 0, meshsize};
//+
Point(6) = {xlen, yCd, 0, meshsize};
//+
Point(7) = {0, drift_mid, 0, meshsize};
//+
Point(8) = {xlen, drift_mid, 0, meshsize};
//+
Point(9) = {0, yBC, 0, meshsize};
//+
Point(10) = {xlen, yBC, 0, meshsize};
//+
Point(11) = {0, yBE, 0, meshsize};
//+
Point(12) = {xlen, yBE, 0, meshsize};
//+
Point(13) = {0, yBE-0.2*ylenB, 0, meshsize};
//+
Point(14) = {xlen, yBE-0.2*ylenB, 0, meshsize};
//+
Point(15) = {0, yBE-0.8*ylenB, 0, meshsize};
//+
Point(16) = {xlen, yBE-0.8*ylenB, 0, meshsize};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 6};
//+
Line(3) = {6, 8};
//+
Line(4) = {8, 10};
//+
Line(5) = {10, 16};
//+
Line(6) = {16, 14};
//+
Line(7) = {14, 12};
//+
Line(8) = {12, 3};
//+
Line(9) = {3, 4};
//+
Line(10) = {4, 11};
//+
Line(11) = {11, 13};
//+
Line(12) = {13, 15};
//+
Line(13) = {15, 9};
//+
Line(14) = {9, 7};
//+
Line(15) = {7, 5};
//+
Line(16) = {5, 1};
//+
Line(17) = {5, 6};
//+
Line(18) = {7, 8};
//+
Line(19) = {9, 10};
//+
Line(20) = {11, 12};
//+
Curve Loop(1) = {1, 2, -17, 16};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {15, 17, 3, 4, -19, 14};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {19, 5, 6, 7, -20, 11, 12, 13};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {10, 20, 8, 9};
//+
Plane Surface(4) = {4};
//+
Physical Surface("domain_E", 24) = {4};
//+
Physical Surface("domain_B", 25) = {3};
//+
Physical Surface("domain_drift", 26) = {2};
//+
Physical Surface("domain_C", 27) = {1};
//+
Physical Curve("bc_E", 28) = {9};
//+
Physical Curve("bc_B", 29) = {5, 6, 7, 11, 12, 13};
//+
Physical Curve("bc_C", 30) = {1};
//+

Field[1] = Attractor;
//+
//Field[1].CurvesList = {20, 19, 17, 6, 12,1, 9, 18};
Field[1].CurvesList = {19};
//+
Field[2] = Threshold;
//+
Field[2].DistMax = 3.5*meshsize;
//+
Field[2].DistMin = 3*meshsize;
//+
Field[2].InField = 1;
//+
Field[2].SizeMax = 0.5*meshsize;
//+
Field[2].SizeMin = 0.1*meshsize;
//+
Background Field = 2;
