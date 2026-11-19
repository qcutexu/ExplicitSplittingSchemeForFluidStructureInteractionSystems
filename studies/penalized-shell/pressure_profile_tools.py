"""Geometry utilities for pressure profiles on mapped material sections.

The fluid mesh used by the pressure-pulse benchmark is obtained by extruding
one triangulated disk through a sequence of reference axial stations.  This
module reconstructs those disk sections from the tetrahedral connectivity.
At any later ALE configuration, it then integrates a continuous P1 pressure
field over the mapped triangular section exactly.

Only NumPy is required, so the geometric calculations can be tested without a
FEniCS installation.
"""

from itertools import combinations

import numpy as np


def _cluster_sorted_values(values, tolerance):
    """Cluster sorted scalar values that differ by at most ``tolerance``."""
    sorted_values = np.sort(np.asarray(values, dtype=float))
    if sorted_values.size == 0:
        return np.empty(0, dtype=float)

    clusters = [[float(sorted_values[0])]]
    for value in sorted_values[1:]:
        value = float(value)
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return np.asarray([np.mean(cluster) for cluster in clusters], dtype=float)


def build_material_cross_sections(
    reference_coordinates,
    tetrahedra,
    axial_tolerance=None,
):
    """Recover the triangulated material cross-sections of an extruded mesh.

    Parameters
    ----------
    reference_coordinates : array_like, shape (number_of_vertices, 3)
        Coordinates of the fixed reference fluid mesh.
    tetrahedra : array_like, shape (number_of_cells, 4)
        Tetrahedral vertex connectivity.
    axial_tolerance : float, optional
        Tolerance used to identify vertices on the same reference axial plane.

    Returns
    -------
    axial_stations : numpy.ndarray
        Sorted reference axial coordinates.
    section_triangles : list[numpy.ndarray]
        For each station, an array of unique triangular faces represented by
        global mesh-vertex indices.

    Notes
    -----
    A material section is stored by its reference vertex indices.  The same
    indices therefore identify its deformed image after ``ALE.move`` changes
    the current mesh coordinates.
    """
    coordinates = np.asarray(reference_coordinates, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int64)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("reference_coordinates must have shape (N, 3).")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (M, 4).")
    if cells.size == 0 or coordinates.size == 0:
        raise ValueError("The reference mesh must be nonempty.")
    if cells.min() < 0 or cells.max() >= len(coordinates):
        raise ValueError("A tetrahedron contains an invalid vertex index.")

    axial_extent = float(np.ptp(coordinates[:, 0]))
    if axial_tolerance is None:
        axial_tolerance = max(1.0e-12 * max(1.0, axial_extent), 1.0e-14)
    if axial_tolerance <= 0.0:
        raise ValueError("axial_tolerance must be positive.")

    unique_axial_values = np.unique(coordinates[:, 0])
    axial_stations = _cluster_sorted_values(
        unique_axial_values,
        axial_tolerance,
    )
    if len(axial_stations) < 2:
        raise ValueError("At least two axial stations are required.")

    vertex_station = np.empty(len(coordinates), dtype=np.int64)
    for vertex, axial_coordinate in enumerate(coordinates[:, 0]):
        insertion = int(np.searchsorted(axial_stations, axial_coordinate))
        candidates = []
        if insertion < len(axial_stations):
            candidates.append(insertion)
        if insertion > 0:
            candidates.append(insertion - 1)
        station = min(
            candidates,
            key=lambda index: abs(axial_stations[index] - axial_coordinate),
        )
        error = abs(axial_stations[station] - axial_coordinate)
        if error > axial_tolerance:
            raise RuntimeError(
                "A reference vertex could not be assigned to an axial station."
            )
        vertex_station[vertex] = station

    face_sets = [set() for _ in axial_stations]
    for cell in cells:
        for face in combinations((int(value) for value in cell), 3):
            stations = vertex_station[np.asarray(face, dtype=np.int64)]
            if stations[0] == stations[1] == stations[2]:
                face_sets[int(stations[0])].add(tuple(sorted(face)))

    section_triangles = []
    for station, faces in enumerate(face_sets):
        if not faces:
            raise RuntimeError(
                "No cross-sectional triangles were found at reference x={:.16e}."
                .format(axial_stations[station])
            )
        section_triangles.append(
            np.asarray(sorted(faces), dtype=np.int64).reshape((-1, 3))
        )

    return axial_stations, section_triangles


def compute_mapped_pressure_profile(
    current_coordinates,
    pressure_at_vertices,
    axial_stations,
    section_triangles,
):
    """Integrate pressure over each ALE-mapped material cross-section.

    For a continuous piecewise-linear pressure, the integral over one mapped
    triangle is its current area times the arithmetic mean of its three nodal
    values.  The calculation is therefore exact for the discrete pressure
    trace on every cross-sectional triangle.

    Returns tuples containing

    ``(reference_x, current_area, current_centroid_x, mean_pressure)``.
    """
    coordinates = np.asarray(current_coordinates, dtype=float)
    pressure = np.asarray(pressure_at_vertices, dtype=float).reshape(-1)
    stations = np.asarray(axial_stations, dtype=float).reshape(-1)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("current_coordinates must have shape (N, 3).")
    if len(pressure) != len(coordinates):
        raise ValueError(
            "There must be one scalar pressure value per fluid-mesh vertex."
        )
    if len(stations) != len(section_triangles):
        raise ValueError(
            "axial_stations and section_triangles must have the same length."
        )
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(pressure)):
        raise ValueError("Coordinates and pressure values must be finite.")

    profile = []
    for reference_x, faces in zip(stations, section_triangles):
        faces = np.asarray(faces, dtype=np.int64)
        first = coordinates[faces[:, 0]]
        second = coordinates[faces[:, 1]]
        third = coordinates[faces[:, 2]]
        triangle_areas = 0.5 * np.linalg.norm(
            np.cross(second - first, third - first),
            axis=1,
        )
        current_area = float(np.sum(triangle_areas))
        if not np.isfinite(current_area) or current_area <= 0.0:
            raise RuntimeError(
                "A mapped material cross-section has nonpositive area at "
                "reference x={:.16e}.".format(reference_x)
            )

        mean_triangle_pressure = (
            pressure[faces[:, 0]]
            + pressure[faces[:, 1]]
            + pressure[faces[:, 2]]
        ) / 3.0
        mean_triangle_x = (first[:, 0] + second[:, 0] + third[:, 0]) / 3.0

        mean_pressure = float(
            np.dot(triangle_areas, mean_triangle_pressure) / current_area
        )
        current_centroid_x = float(
            np.dot(triangle_areas, mean_triangle_x) / current_area
        )
        profile.append(
            (
                float(reference_x),
                current_area,
                current_centroid_x,
                mean_pressure,
            )
        )

    return profile
