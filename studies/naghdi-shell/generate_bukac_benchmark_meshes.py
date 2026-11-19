"""Generate meshes for the straight-tube pressure-pulse benchmark.

1. A 2D rectangular shell mesh on (x, y_arc):
       0 <= x <= width,
       0 <= y_arc <= 2*pi*Radius.
   This is always saved as legacy ``stent.xml``.  If h5py is installed, the
   script also writes ``stent.xdmf`` + ``stent.h5``.

2. A warped 3D shell surface mesh using
       (x, y_arc) -> (x, Radius*cos(y_arc/Radius), Radius*sin(y_arc/Radius)).
   This is saved as shell_surface_3d.vtk for checking in ParaView.

3. A conforming tetrahedral mesh of the 3D fluid cylinder, saved directly as
   legacy ``fluid.xml``.  By default, its lateral boundary has the same axial
   stations, circumferential vertices, and triangle connectivity as the shell
   mesh, modulo the shell's duplicated periodic seam.  A Gmsh ``.geo`` file is
   also written as an optional nonconforming alternative for mesh-refinement
   studies.

Notes:
- The structure mesh is a 2D parameter-domain mesh. The FSI code warps it into
  3D using initial_shape.
- The direct fluid mesh is obtained by triangulating a circular cross section,
  extruding it in the axial direction, and subdividing every triangular prism
  into three consistently oriented tetrahedra.
- The default direct meshes have a conforming fluid--structure interface.  The
  shell still contains two copies of the periodic seam in parameter space;
  those copies map to the same fluid-interface vertices in three dimensions.
- Passing ``--nonmatching-interface`` restores the earlier independent fluid
  boundary and shell-surface discretizations.
"""

import math
import os
import argparse
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.spatial import Delaunay
except ImportError:
    Delaunay = None


def write_structure_dolfin_xml(filename, points_2d, triangles):
    """Write a two-dimensional triangle mesh in legacy DOLFIN XML."""
    points_2d = np.asarray(points_2d, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    with open(filename, "w") as stream:
        stream.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        stream.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
        stream.write('  <mesh celltype="triangle" dim="2">\n')
        stream.write('    <vertices size="{}">\n'.format(len(points_2d)))
        for index, point in enumerate(points_2d):
            stream.write(
                '      <vertex index="{}" x="{:.16e}" y="{:.16e}"/>\n'.format(
                    index, point[0], point[1]
                )
            )
        stream.write('    </vertices>\n')
        stream.write('    <cells size="{}">\n'.format(len(triangles)))
        for index, triangle in enumerate(triangles):
            stream.write(
                '      <triangle index="{}" v0="{}" v1="{}" v2="{}"/>\n'.format(
                    index, triangle[0], triangle[1], triangle[2]
                )
            )
        stream.write('    </cells>\n')
        stream.write('  </mesh>\n')
        stream.write('</dolfin>\n')


def write_fluid_dolfin_xml(filename, points_3d, tetrahedra):
    """Write a three-dimensional tetrahedral mesh in legacy DOLFIN XML."""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    with open(filename, "w") as stream:
        stream.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        stream.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
        stream.write('  <mesh celltype="tetrahedron" dim="3">\n')
        stream.write('    <vertices size="{}">\n'.format(len(points_3d)))
        for index, point in enumerate(points_3d):
            stream.write(
                '      <vertex index="{}" x="{:.16e}" y="{:.16e}" '
                'z="{:.16e}"/>\n'.format(
                    index, point[0], point[1], point[2]
                )
            )
        stream.write('    </vertices>\n')
        stream.write('    <cells size="{}">\n'.format(len(tetrahedra)))
        for index, tetrahedron in enumerate(tetrahedra):
            stream.write(
                '      <tetrahedron index="{}" v0="{}" v1="{}" '
                'v2="{}" v3="{}"/>\n'.format(
                    index,
                    tetrahedron[0],
                    tetrahedron[1],
                    tetrahedron[2],
                    tetrahedron[3],
                )
            )
        stream.write('    </cells>\n')
        stream.write('  </mesh>\n')
        stream.write('</dolfin>\n')


def generate_disk_mesh(
    radius,
    target_size,
    boundary_angular_points=None,
):
    """Triangulate a polygonal disk with approximately uniform spacing."""
    if Delaunay is None:
        raise RuntimeError(
            "SciPy is required for direct fluid.xml generation."
        )

    number_of_radial_layers = max(2, int(math.ceil(radius / target_size)))
    points = [[0.0, 0.0]]
    for radial_index in range(1, number_of_radial_layers + 1):
        radial_coordinate = radius * radial_index / number_of_radial_layers
        if (
            radial_index == number_of_radial_layers
            and boundary_angular_points is not None
        ):
            number_of_angular_points = int(boundary_angular_points)
        else:
            number_of_angular_points = max(
                6,
                int(
                    math.ceil(
                        2.0
                        * math.pi
                        * radial_coordinate
                        / target_size
                    )
                ),
            )
        angular_increment = 2.0 * math.pi / number_of_angular_points
        # The outer ring must start at theta=0 so that it coincides with the
        # shell vertices.  Interior rings remain staggered to improve the
        # cross-sectional triangle quality.
        if radial_index == number_of_radial_layers:
            phase = 0.0
        else:
            phase = 0.5 * angular_increment if radial_index % 2 else 0.0
        for angular_index in range(number_of_angular_points):
            angle = phase + angular_index * angular_increment
            points.append(
                [
                    radial_coordinate * math.cos(angle),
                    radial_coordinate * math.sin(angle),
                ]
            )

    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(Delaunay(points).simplices, dtype=np.int64)

    # Orient every cross-sectional triangle counterclockwise in (y,z).
    for triangle in triangles:
        a, b, c = points[triangle]
        signed_twice_area = (
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )
        if signed_twice_area < 0.0:
            triangle[1], triangle[2] = triangle[2], triangle[1]
    outer_boundary_indices = np.where(
        np.isclose(
            np.linalg.norm(points, axis=1),
            radius,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
    )[0]
    outer_angles = np.mod(
        np.arctan2(
            points[outer_boundary_indices, 1],
            points[outer_boundary_indices, 0],
        ),
        2.0 * math.pi,
    )
    outer_boundary_indices = outer_boundary_indices[np.argsort(outer_angles)]
    if (
        boundary_angular_points is not None
        and len(outer_boundary_indices) != boundary_angular_points
    ):
        raise RuntimeError(
            "The fluid cross section does not contain the requested number "
            "of outer-boundary vertices."
        )
    return points, triangles, outer_boundary_indices


def signed_tetrahedron_jacobians(points_3d, tetrahedra):
    """Return six times the oriented volume of every tetrahedron."""
    coordinates = np.asarray(points_3d, dtype=np.float64)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    x0 = coordinates[cells[:, 0]]
    x1 = coordinates[cells[:, 1]]
    x2 = coordinates[cells[:, 2]]
    x3 = coordinates[cells[:, 3]]
    return np.einsum(
        "ij,ij->i",
        x1 - x0,
        np.cross(x2 - x0, x3 - x0),
    )


def extrude_disk_mesh(points_yz, triangles, x0, width, axial_cells):
    """Extrude a disk triangulation and split each prism into three tets.

    A common ordering of cross-sectional vertex numbers determines every
    diagonal on a shared quadrilateral prism face.  Neighboring prisms
    therefore induce exactly the same surface triangulation.
    """
    number_of_cross_section_points = len(points_yz)
    axial_coordinates = np.linspace(x0, x0 + width, axial_cells + 1)
    points_3d = np.empty(
        ((axial_cells + 1) * number_of_cross_section_points, 3),
        dtype=np.float64,
    )
    for axial_index, axial_coordinate in enumerate(axial_coordinates):
        start = axial_index * number_of_cross_section_points
        stop = start + number_of_cross_section_points
        points_3d[start:stop, 0] = axial_coordinate
        points_3d[start:stop, 1:] = points_yz

    tetrahedra = []
    for axial_index in range(axial_cells):
        lower_offset = axial_index * number_of_cross_section_points
        upper_offset = (axial_index + 1) * number_of_cross_section_points
        for triangle in triangles:
            a, b, c = sorted(int(value) for value in triangle)
            a0, b0, c0 = (
                lower_offset + a,
                lower_offset + b,
                lower_offset + c,
            )
            a1, b1, c1 = (
                upper_offset + a,
                upper_offset + b,
                upper_offset + c,
            )
            tetrahedra.extend(
                (
                    [a0, b0, c0, a1],
                    [b0, c0, a1, b1],
                    [c0, a1, b1, c1],
                )
            )

    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    signed_jacobians = signed_tetrahedron_jacobians(points_3d, tetrahedra)
    negative_cells = signed_jacobians < 0.0
    if np.any(negative_cells):
        temporary = tetrahedra[negative_cells, 2].copy()
        tetrahedra[negative_cells, 2] = tetrahedra[negative_cells, 3]
        tetrahedra[negative_cells, 3] = temporary

    signed_jacobians = signed_tetrahedron_jacobians(points_3d, tetrahedra)
    length_scale = float(np.ptp(points_3d, axis=0).max())
    volume_tolerance = 1.0e-14 * length_scale**3
    if np.any(signed_jacobians <= volume_tolerance):
        raise RuntimeError("Direct fluid meshing produced a degenerate tetrahedron.")
    return points_3d, tetrahedra


def write_structure_xdmf_h5(filename_base, points_2d, triangles):
    """Write a legacy-compatible XDMF/HDF5 mesh for FEniCS."""
    if h5py is None:
        raise RuntimeError("h5py is required for XDMF/HDF5 output.")
    h5_name = filename_base + ".h5"
    xdmf_name = filename_base + ".xdmf"

    # FEniCS XDMF wants geometry as N x 2 for GeometryType XY
    points_2d = np.asarray(points_2d, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)

    with h5py.File(h5_name, "w") as h5:
        h5.create_dataset("/data0", data=points_2d)
        h5.create_dataset("/data1", data=triangles)

    xdmf = f'''<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="Triangle" NumberOfElements="{len(triangles)}">
        <DataItem Format="HDF" DataType="Int" Dimensions="{len(triangles)} 3">{h5_name}:/data1</DataItem>
      </Topology>
      <Geometry GeometryType="XY">
        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{len(points_2d)} 2">{h5_name}:/data0</DataItem>
      </Geometry>
    </Grid>
  </Domain>
</Xdmf>
'''
    with open(xdmf_name, "w") as f:
        f.write(xdmf)


def write_vtk_surface(filename, points_3d, triangles):
    """Write a simple ASCII VTK PolyData surface for ParaView inspection."""
    points_3d = np.asarray(points_3d, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Warped cylindrical shell surface\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points_3d)} float\n")
        for p in points_3d:
            f.write(f"{p[0]:.16e} {p[1]:.16e} {p[2]:.16e}\n")
        f.write(f"POLYGONS {len(triangles)} {4*len(triangles)}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def generate_rectangular_shell_mesh(
    width,
    radius,
    nx,
    ntheta,
    match_extruded_fluid_boundary=True,
):
    """Generate 2D triangular mesh on [0,width] x [0,2*pi*R].

    The top and bottom circumferential edges are left as separate boundaries;
    periodicity is imposed later in FEniCS by constrained_domain=pbc.
    """
    length_arc = 2.0 * math.pi * radius
    xs = np.linspace(0.0, width, nx + 1)
    ys = np.linspace(0.0, length_arc, ntheta + 1)

    points = []
    for i in range(nx + 1):
        for j in range(ntheta + 1):
            points.append([xs[i], ys[j]])
    points = np.array(points, dtype=np.float64)

    def vid(i, j):
        return i * (ntheta + 1) + j

    triangles = []
    for i in range(nx):
        for j in range(ntheta):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            # For consecutive angular fluid-boundary vertices, the prism
            # subdivision in extrude_disk_mesh() uses the v10--v01 diagonal.
            # Across the periodic wrap, it uses v00--v11.  Reproducing those
            # choices here makes the embedded shell triangles identical to
            # the lateral boundary faces of the tetrahedral fluid mesh.
            if match_extruded_fluid_boundary and j == ntheta - 1:
                triangles.append([v00, v10, v11])
                triangles.append([v00, v11, v01])
            elif match_extruded_fluid_boundary:
                triangles.append([v00, v10, v01])
                triangles.append([v10, v11, v01])
            # Retain the former alternating pattern for explicitly requested
            # nonmatching meshes.
            elif (i + j) % 2 == 0:
                triangles.append([v00, v10, v11])
                triangles.append([v00, v11, v01])
            else:
                triangles.append([v00, v10, v01])
                triangles.append([v10, v11, v01])
    return points, np.array(triangles, dtype=np.int64)


def tetrahedral_boundary_faces(tetrahedra):
    """Return the unoriented triangular faces on a tetrahedral-mesh boundary."""
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    faces = np.concatenate(
        (
            tetrahedra[:, [0, 1, 2]],
            tetrahedra[:, [0, 1, 3]],
            tetrahedra[:, [0, 2, 3]],
            tetrahedra[:, [1, 2, 3]],
        ),
        axis=0,
    )
    faces.sort(axis=1)
    unique_faces, counts = np.unique(faces, axis=0, return_counts=True)
    return unique_faces[counts == 1]


def verify_matching_interface(
    shell_points_2d,
    shell_triangles,
    fluid_points,
    fluid_tetrahedra,
    fluid_cross_section_size,
    fluid_outer_boundary_indices,
    radius,
    nx,
    ntheta,
):
    """Verify interface coordinates and triangle connectivity exactly."""
    shell_points_3d = warp_shell_to_cylinder(shell_points_2d, radius)

    shell_to_fluid_vertex = np.empty(len(shell_points_2d), dtype=np.int64)
    for axial_index in range(nx + 1):
        fluid_axial_offset = axial_index * fluid_cross_section_size
        for angular_index in range(ntheta + 1):
            shell_vertex = axial_index * (ntheta + 1) + angular_index
            fluid_outer_vertex = fluid_outer_boundary_indices[
                angular_index % ntheta
            ]
            shell_to_fluid_vertex[shell_vertex] = (
                fluid_axial_offset + fluid_outer_vertex
            )

    coordinate_error = np.linalg.norm(
        shell_points_3d - fluid_points[shell_to_fluid_vertex],
        axis=1,
    )
    maximum_coordinate_error = float(coordinate_error.max())

    mapped_shell_faces = shell_to_fluid_vertex[shell_triangles]
    mapped_shell_faces.sort(axis=1)
    mapped_shell_face_set = {
        tuple(face) for face in mapped_shell_faces.tolist()
    }

    boundary_faces = tetrahedral_boundary_faces(fluid_tetrahedra)
    boundary_coordinates = fluid_points[boundary_faces]
    boundary_radii = np.linalg.norm(boundary_coordinates[:, :, 1:3], axis=2)
    axial_span = np.ptp(boundary_coordinates[:, :, 0], axis=1)
    lateral_mask = (
        np.all(
            np.isclose(
                boundary_radii,
                radius,
                rtol=1.0e-12,
                atol=1.0e-14,
            ),
            axis=1,
        )
        & (axial_span > 1.0e-14)
    )
    lateral_fluid_faces = boundary_faces[lateral_mask]
    lateral_fluid_face_set = {
        tuple(face) for face in lateral_fluid_faces.tolist()
    }

    if maximum_coordinate_error > 1.0e-12 * max(1.0, radius):
        raise RuntimeError(
            "The shell and fluid interface vertices do not coincide."
        )
    if len(mapped_shell_face_set) != len(shell_triangles):
        raise RuntimeError(
            "The periodic shell mapping produced duplicate interface triangles."
        )
    if mapped_shell_face_set != lateral_fluid_face_set:
        missing_from_fluid = len(
            mapped_shell_face_set - lateral_fluid_face_set
        )
        missing_from_shell = len(
            lateral_fluid_face_set - mapped_shell_face_set
        )
        raise RuntimeError(
            "The interface triangulations do not match: {} shell faces are "
            "absent from the fluid boundary and {} fluid faces are absent "
            "from the shell.".format(missing_from_fluid, missing_from_shell)
        )

    return shell_to_fluid_vertex, maximum_coordinate_error


def write_interface_node_map(filename, shell_to_fluid_vertex):
    """Write the shell-vertex to fluid-interface-vertex correspondence."""
    with open(filename, "w") as stream:
        stream.write("shell_vertex,fluid_vertex\n")
        for shell_vertex, fluid_vertex in enumerate(shell_to_fluid_vertex):
            stream.write("{},{}\n".format(shell_vertex, fluid_vertex))


def warp_shell_to_cylinder(points_2d, radius):
    """Apply initial_shape: (x,y_arc)->(x,R cos(y/R), R sin(y/R))."""
    points_3d = np.zeros((len(points_2d), 3), dtype=np.float64)
    points_3d[:, 0] = points_2d[:, 0]
    theta = points_2d[:, 1] / radius
    points_3d[:, 1] = radius * np.cos(theta)
    points_3d[:, 2] = radius * np.sin(theta)
    return points_3d


def write_fluid_geo(path, width, radius, x0, lc):
    """Write a Gmsh .geo file for a matching cylinder volume.

    We use OpenCASCADE Cylinder because it creates a clean closed volume for
    tetrahedralization. Its lateral surface is exactly the same surface as the
    warped shell: (x, R cos(theta), R sin(theta)).
    """
    geo = f'''// Matching fluid cylinder generated from the same parameters as the shell mesh
SetFactory("OpenCASCADE");

R = {radius:.16e};
x0 = {x0:.16e};
L = {width:.16e};
lc = {lc:.16e};

// Cylinder axis is the x-direction, centered at (y,z)=(0,0).
Cylinder(1) = {{x0, 0, 0, L, 0, 0, R, 2*Pi}};

// Mesh size control.
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

// Physical names are useful if you later export boundary markers.
// Your current FEniCS code marks boundaries geometrically, so these are optional.
Physical Volume("fluid", 10) = {{1}};

// After gmsh creates the cylinder, the boundary surface tags are determined by OCC.
// The Python/FEniCS code currently detects inlet/outlet/wall by x_min/x_max, so
// explicit physical surface tags are not required for fluid.xml.
'''
    with open(path, "w") as f:
        f.write(geo)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the straight-tube FSI benchmark meshes."
    )
    parser.add_argument("--radius", type=float, default=0.005, help="Cylinder radius [m].")
    parser.add_argument("--width", type=float, default=0.060, help="Axial length [m].")
    parser.add_argument("--x0", type=float, default=0.0, help="Fluid cylinder inlet x-coordinate [m].")
    parser.add_argument("--nx", type=int, default=60, help="Number of shell cells in axial direction.")
    parser.add_argument("--ntheta", type=int, default=32, help="Number of shell cells circumferentially.")
    parser.add_argument(
        "--lc",
        type=float,
        default=1.0e-3,
        help="Target fluid-mesh spacing [m].",
    )
    parser.add_argument(
        "--fluid-axial-cells",
        type=int,
        default=0,
        help=(
            "Number of fluid axial layers. For the default matching interface, "
            "zero selects nx and a positive value must equal nx. With "
            "--nonmatching-interface, zero selects ceil(width/lc)."
        ),
    )
    parser.add_argument(
        "--nonmatching-interface",
        action="store_true",
        help=(
            "Generate the fluid boundary independently of the shell. The "
            "default is a conforming fluid--shell interface."
        ),
    )
    parser.add_argument("--outdir", type=str, default=".", help="Output directory.")
    args = parser.parse_args()

    if args.radius <= 0.0 or args.width <= 0.0 or args.lc <= 0.0:
        raise ValueError("radius, width, and lc must be positive.")
    if args.nx < 2 or args.ntheta < 4:
        raise ValueError("Use at least nx=2 and ntheta=4.")
    if not args.nonmatching_interface and args.ntheta < 5:
        raise ValueError(
            "The matching polygonal interface requires ntheta>=5."
        )
    if args.fluid_axial_cells < 0:
        raise ValueError("fluid-axial-cells cannot be negative.")
    if (
        not args.nonmatching_interface
        and args.fluid_axial_cells not in (0, args.nx)
    ):
        raise ValueError(
            "For a matching interface, --fluid-axial-cells must equal --nx "
            "or be left at zero."
        )
    if not math.isclose(args.x0, 0.0, abs_tol=1.0e-14):
        raise ValueError(
            "The benchmark solver requires the fluid inlet at x0=0."
        )

    os.makedirs(args.outdir, exist_ok=True)

    matching_interface = not args.nonmatching_interface
    points_2d, triangles = generate_rectangular_shell_mesh(
        args.width,
        args.radius,
        args.nx,
        args.ntheta,
        match_extruded_fluid_boundary=matching_interface,
    )
    points_3d = warp_shell_to_cylinder(points_2d, args.radius)
    fluid_points_yz, fluid_triangles, fluid_outer_boundary_indices = (
        generate_disk_mesh(
            args.radius,
            args.lc,
            boundary_angular_points=(
                args.ntheta if matching_interface else None
            ),
        )
    )
    if matching_interface:
        fluid_axial_cells = args.nx
    else:
        fluid_axial_cells = (
            args.fluid_axial_cells
            if args.fluid_axial_cells > 0
            else int(math.ceil(args.width / args.lc))
        )
    fluid_points, fluid_tetrahedra = extrude_disk_mesh(
        fluid_points_yz,
        fluid_triangles,
        args.x0,
        args.width,
        fluid_axial_cells,
    )

    shell_to_fluid_vertex = None
    maximum_interface_coordinate_error = None
    if matching_interface:
        (
            shell_to_fluid_vertex,
            maximum_interface_coordinate_error,
        ) = verify_matching_interface(
            points_2d,
            triangles,
            fluid_points,
            fluid_tetrahedra,
            len(fluid_points_yz),
            fluid_outer_boundary_indices,
            args.radius,
            args.nx,
            args.ntheta,
        )

    old_cwd = os.getcwd()
    os.chdir(args.outdir)
    try:
        write_structure_dolfin_xml("stent.xml", points_2d, triangles)
        if h5py is not None:
            write_structure_xdmf_h5("stent", points_2d, triangles)
        write_vtk_surface("shell_surface_3d.vtk", points_3d, triangles)
        write_fluid_dolfin_xml(
            "fluid.xml", fluid_points, fluid_tetrahedra
        )
        if matching_interface:
            write_interface_node_map(
                "interface_node_map.csv",
                shell_to_fluid_vertex,
            )
        write_fluid_geo("fluid_from_shell_surface.geo", args.width, args.radius, args.x0, args.lc)
    finally:
        os.chdir(old_cwd)

    print("Generated:")
    print(os.path.join(args.outdir, "stent.xml"))
    if h5py is not None:
        print(os.path.join(args.outdir, "stent.xdmf"))
        print(os.path.join(args.outdir, "stent.h5"))
    else:
        print("h5py was not found; skipped optional stent.xdmf/stent.h5 output.")
    print(os.path.join(args.outdir, "shell_surface_3d.vtk"))
    print(os.path.join(args.outdir, "fluid.xml"))
    if matching_interface:
        print(os.path.join(args.outdir, "interface_node_map.csv"))
    print(os.path.join(args.outdir, "fluid_from_shell_surface.geo"))
    fluid_volume = signed_tetrahedron_jacobians(
        fluid_points, fluid_tetrahedra
    ).sum() / 6.0
    print("\nDirect fluid mesh statistics:")
    print("  vertices: {}".format(len(fluid_points)))
    print("  tetrahedra: {}".format(len(fluid_tetrahedra)))
    print("  polygonal-cylinder volume: {:.16e} m^3".format(fluid_volume))
    if matching_interface:
        print("\nFluid--shell interface verification:")
        print("  status: matching")
        print("  shell triangles: {}".format(len(triangles)))
        print(
            "  maximum corresponding-vertex error: {:.3e} m".format(
                maximum_interface_coordinate_error
            )
        )
    print("\nOptional Gmsh replacement:")
    print("  cd {}".format(args.outdir))
    print("  gmsh -3 fluid_from_shell_surface.geo -format msh2 -o fluid.msh")
    print("  dolfin-convert fluid.msh fluid.xml")


if __name__ == "__main__":
    main()
