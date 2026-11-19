// Matching fluid cylinder generated from the same parameters as the shell mesh
SetFactory("OpenCASCADE");

R = 5.0000000000000001e-03;
x0 = 0.0000000000000000e+00;
L = 5.9999999999999998e-02;
lc = 1.0000000000000000e-03;

// Cylinder axis is the x-direction, centered at (y,z)=(0,0).
Cylinder(1) = {x0, 0, 0, L, 0, 0, R, 2*Pi};

// Mesh size control.
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

// Physical names are useful if you later export boundary markers.
// Your current FEniCS code marks boundaries geometrically, so these are optional.
Physical Volume("fluid", 10) = {1};

// After gmsh creates the cylinder, the boundary surface tags are determined by OCC.
// The Python/FEniCS code currently detects inlet/outlet/wall by x_min/x_max, so
// explicit physical surface tags are not required for fluid.xml.
