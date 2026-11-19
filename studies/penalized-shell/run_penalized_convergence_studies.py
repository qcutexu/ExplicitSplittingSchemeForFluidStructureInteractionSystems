#!/usr/bin/env python3
"""Generate and run the penalized pressure-profile convergence studies.

The default study contains three uniformly refined matching fluid--shell
meshes and three time steps.  The medium-mesh run with dt=25 microseconds is
shared by the spatial and temporal studies, so only five distinct simulations
are required.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


SNAPSHOT_TIMES_MS = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
FINAL_TIME_S = 0.012
SPATIAL_TIME_STEP_S = 2.5e-5
PROFILE_BINS = 30
DEFAULT_MEMBRANE_SHEAR_SCALE = 1.0e-2
DEFAULT_BENDING_SCALE = 1.0e-3
DEFAULT_AXIAL_PENALTY_SCALE = 10.0
DEFAULT_SYMMETRY_PENALTY_SCALE = 10.0

MESH_LEVELS = (
    {
        "name": "coarse",
        "nx": 24,
        "ntheta": 16,
        "lc_m": 2.0e-3,
        "h_m": 2.0e-3,
    },
    {
        "name": "medium",
        "nx": 40,
        "ntheta": 24,
        "lc_m": 1.0e-3,
        "h_m": 1.0e-3,
    },
    {
        "name": "fine",
        "nx": 60,
        "ntheta": 32,
        "lc_m": 0.5e-3,
        "h_m": 0.5e-3,
    },
)

TEMPORAL_LEVELS = (
    {"name": "dt_100us", "dt_s": 1.0e-4},
    {"name": "dt_50us", "dt_s": 5.0e-5},
    {"name": "dt_25us", "dt_s": 2.5e-5},
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate meshes, run the spatial and temporal convergence cases, "
            "and create pressure-profile comparisons."
        )
    )
    parser.add_argument(
        "--study-dir",
        default="penalized_convergence_results",
        help="Directory containing meshes, case results, and comparisons.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help=(
            "Python executable with legacy FEniCS, NumPy, SciPy, and "
            "Matplotlib available."
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Fluid--structure splitting parameter passed to every case.",
    )
    parser.add_argument(
        "--pressure-stabilization",
        type=float,
        default=1.0e-2,
        help="Equal-order pressure-stabilization coefficient.",
    )
    parser.add_argument(
        "--membrane-shear-scale",
        type=float,
        default=DEFAULT_MEMBRANE_SHEAR_SCALE,
        help="Multiplier applied to the transverse-shear constitutive coefficient.",
    )
    parser.add_argument(
        "--bending-scale",
        type=float,
        default=DEFAULT_BENDING_SCALE,
        help="Multiplier applied to the complete shell bending form.",
    )
    parser.add_argument(
        "--axial-penalty-scale",
        type=float,
        default=DEFAULT_AXIAL_PENALTY_SCALE,
        help="Dimensionless axial-displacement penalty scale.",
    )
    parser.add_argument(
        "--symmetry-penalty-scale",
        type=float,
        default=DEFAULT_SYMMETRY_PENALTY_SCALE,
        help="Dimensionless tangential-symmetry penalty scale.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed simulation cases.",
    )
    parser.add_argument(
        "--regenerate-meshes",
        action="store_true",
        help="Regenerate mesh files even when they already exist.",
    )
    parser.add_argument(
        "--write-field-output",
        action="store_true",
        help="Write the large PVD field collections for every convergence run.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Run the simulations without invoking the comparison postprocessor.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write the manifest without executing them.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.beta <= 1.0:
        parser.error("--beta must lie in [0,1].")
    if args.pressure_stabilization < 0.0:
        parser.error("--pressure-stabilization cannot be negative.")
    for option_name, option_value in (
        ("--membrane-shear-scale", args.membrane_shear_scale),
        ("--bending-scale", args.bending_scale),
        ("--axial-penalty-scale", args.axial_penalty_scale),
        ("--symmetry-penalty-scale", args.symmetry_penalty_scale),
    ):
        if option_value < 0.0:
            parser.error("{} cannot be negative.".format(option_name))
    return args


def command_text(command):
    """Return a readable command without relying on a shell for execution."""
    try:
        import shlex

        return " ".join(shlex.quote(str(value)) for value in command)
    except ImportError:
        return " ".join(str(value) for value in command)


def run_command(command, log_path, dry_run=False, environment=None):
    print("\n$ {}".format(command_text(command)))
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            [str(value) for value in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=environment,
        )
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        process.stdout.close()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def mesh_is_available(mesh_directory):
    return all(
        (mesh_directory / filename).is_file()
        for filename in ("fluid.xml", "stent.xml")
    )


def completed_case_matches(
    output_directory,
    time_step,
    beta,
    pressure_stabilization,
    membrane_shear_scale,
    bending_scale,
    axial_penalty_scale,
    symmetry_penalty_scale,
    require_field_output,
):
    completion_path = output_directory / "run_complete.json"
    metadata_path = output_directory / "run_metadata.json"
    axial_profile_path = output_directory / "axial_profiles.csv"
    cross_section_profile_path = (
        output_directory / "cross_section_pressure_profiles.csv"
    )
    if not (
        completion_path.is_file()
        and metadata_path.is_file()
        and axial_profile_path.is_file()
        and cross_section_profile_path.is_file()
    ):
        return False
    try:
        with completion_path.open() as handle:
            completion = json.load(handle)
        with metadata_path.open() as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return False

    stored_snapshots = completion.get("snapshot_times_s", [])
    expected_snapshots = [1.0e-3 * value for value in SNAPSHOT_TIMES_MS]
    snapshots_match = (
        len(stored_snapshots) == len(expected_snapshots)
        and all(
            math.isclose(
                float(stored),
                expected,
                rel_tol=1.0e-12,
                abs_tol=1.0e-14,
            )
            for stored, expected in zip(stored_snapshots, expected_snapshots)
        )
    )

    return (
        bool(completion.get("completed", False))
        and math.isclose(
            float(completion.get("final_time_s", -1.0)),
            FINAL_TIME_S,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
        and math.isclose(
            float(metadata.get("time_step_s", -1.0)),
            time_step,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
        and math.isclose(
            float(metadata.get("beta", -1.0)),
            beta,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
        and math.isclose(
            float(metadata.get("pressure_stabilization", -1.0)),
            pressure_stabilization,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
        and math.isclose(
            float(metadata.get("membrane_shear_scale", -1.0)),
            membrane_shear_scale,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
        and math.isclose(
            float(metadata.get("bending_scale", -1.0)),
            bending_scale,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
        and math.isclose(
            float(metadata.get("axial_penalty_scale", -1.0)),
            axial_penalty_scale,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
        and math.isclose(
            float(metadata.get("symmetry_penalty_scale", -1.0)),
            symmetry_penalty_scale,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
        and snapshots_match
        and (
            not require_field_output
            or bool(metadata.get("field_output_written", False))
        )
    )


def relative_to_study(path, study_directory):
    return os.path.relpath(str(path), str(study_directory))


def build_manifest(
    study_directory,
    beta,
    pressure_stabilization,
    membrane_shear_scale,
    bending_scale,
    axial_penalty_scale,
    symmetry_penalty_scale,
    field_output_written,
):
    mesh_entries = []
    for mesh in MESH_LEVELS:
        entry = dict(mesh)
        entry["directory"] = relative_to_study(
            study_directory / "meshes" / mesh["name"],
            study_directory,
        )
        mesh_entries.append(entry)

    spatial_cases = []
    for mesh in MESH_LEVELS:
        spatial_cases.append(
            {
                "name": mesh["name"],
                "mesh": mesh["name"],
                "h_m": mesh["h_m"],
                "dt_s": SPATIAL_TIME_STEP_S,
                "output_directory": relative_to_study(
                    study_directory / "spatial" / mesh["name"],
                    study_directory,
                ),
            }
        )

    temporal_cases = []
    for temporal in TEMPORAL_LEVELS:
        if math.isclose(
            temporal["dt_s"],
            SPATIAL_TIME_STEP_S,
            rel_tol=0.0,
            abs_tol=1.0e-16,
        ):
            output_directory = study_directory / "spatial" / "medium"
        else:
            output_directory = study_directory / "temporal" / temporal["name"]
        temporal_cases.append(
            {
                "name": temporal["name"],
                "mesh": "medium",
                "h_m": 1.0e-3,
                "dt_s": temporal["dt_s"],
                "output_directory": relative_to_study(
                    output_directory,
                    study_directory,
                ),
            }
        )

    return {
        "schema_version": 1,
        "description": (
            "Spatial and temporal convergence study for the three-dimensional "
            "fluid--Naghdi-shell penalized pressure-pulse benchmark."
        ),
        "final_time_s": FINAL_TIME_S,
        "snapshot_times_ms": list(SNAPSHOT_TIMES_MS),
        "profile_coordinate": "reference_x_m",
        "profile_quantity": (
            "axial_slab_mean_pressure_Pa_and_mean_diameter_m"
        ),
        "additional_profile_quantity": "mapped_cross_section_mean_pressure_Pa",
        "profile_bins": PROFILE_BINS,
        "solver_parameters": {
            "beta": beta,
            "pressure_stabilization": pressure_stabilization,
            "membrane_shear_scale": membrane_shear_scale,
            "bending_scale": bending_scale,
            "axial_penalty_scale": axial_penalty_scale,
            "symmetry_penalty_scale": symmetry_penalty_scale,
            "field_output_written": field_output_written,
        },
        "mesh_levels": mesh_entries,
        "spatial_study": {
            "fixed_dt_s": SPATIAL_TIME_STEP_S,
            "cases": spatial_cases,
        },
        "temporal_study": {
            "fixed_mesh": "medium",
            "cases": temporal_cases,
        },
    }


def main():
    args = parse_arguments()
    script_directory = Path(__file__).resolve().parent
    study_directory = Path(args.study_dir).expanduser().resolve()
    mesh_generator = script_directory / "generate_bukac_benchmark_meshes.py"
    solver = (
        script_directory / "fsi_naghdi_pressure_pulse_penalized_convergence.py"
    )
    analyzer = script_directory / "analyze_convergence.py"

    for required_script in (mesh_generator, solver, analyzer):
        if not required_script.is_file():
            raise FileNotFoundError("Missing required script: {}".format(required_script))

    study_directory.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        study_directory,
        args.beta,
        args.pressure_stabilization,
        args.membrane_shear_scale,
        args.bending_scale,
        args.axial_penalty_scale,
        args.symmetry_penalty_scale,
        args.write_field_output,
    )
    manifest_path = study_directory / "study_manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    mesh_by_name = {mesh["name"]: mesh for mesh in MESH_LEVELS}
    for mesh in MESH_LEVELS:
        mesh_directory = study_directory / "meshes" / mesh["name"]
        mesh_directory.mkdir(parents=True, exist_ok=True)
        if args.regenerate_meshes or not mesh_is_available(mesh_directory):
            mesh_command = [
                args.python_executable,
                str(mesh_generator),
                "--radius",
                "0.005",
                "--width",
                "0.060",
                "--x0",
                "0.0",
                "--nx",
                str(mesh["nx"]),
                "--ntheta",
                str(mesh["ntheta"]),
                "--lc",
                "{:.16g}".format(mesh["lc_m"]),
                "--outdir",
                str(mesh_directory),
            ]
            run_command(
                mesh_command,
                mesh_directory / "mesh_generation.log",
                dry_run=args.dry_run,
            )
        else:
            print("Reusing mesh: {}".format(mesh_directory))

    unique_cases = []
    seen_outputs = set()
    all_manifest_cases = (
        manifest["spatial_study"]["cases"]
        + manifest["temporal_study"]["cases"]
    )
    for case in all_manifest_cases:
        output_directory = study_directory / case["output_directory"]
        output_key = str(output_directory.resolve())
        if output_key not in seen_outputs:
            unique_cases.append(case)
            seen_outputs.add(output_key)

    snapshot_argument = ",".join(
        "{:g}".format(value) for value in SNAPSHOT_TIMES_MS
    )
    solver_environment = os.environ.copy()
    if sys.platform == "darwin" and "FI_PROVIDER" not in solver_environment:
        # Recent MPICH/libfabric builds can fail in MPI_Finalize on macOS when
        # the default network provider is selected, even for one process.  The
        # MPICH-recommended workaround is to select the TCP provider explicitly.
        solver_environment["FI_PROVIDER"] = "tcp"
        print("Using FI_PROVIDER=tcp for macOS FEniCS solver processes.")

    for case in unique_cases:
        mesh = mesh_by_name[case["mesh"]]
        mesh_directory = study_directory / "meshes" / mesh["name"]
        output_directory = study_directory / case["output_directory"]
        output_directory.mkdir(parents=True, exist_ok=True)

        if (
            not args.force
            and not args.regenerate_meshes
            and completed_case_matches(
                output_directory,
                case["dt_s"],
                args.beta,
                args.pressure_stabilization,
                args.membrane_shear_scale,
                args.bending_scale,
                args.axial_penalty_scale,
                args.symmetry_penalty_scale,
                args.write_field_output,
            )
        ):
            print("Reusing completed case: {}".format(case["name"]))
            continue

        solver_command = [
            args.python_executable,
            str(solver),
            "--mesh-dir",
            str(mesh_directory),
            "--output-dir",
            str(output_directory),
            "--dt",
            "{:.16g}".format(case["dt_s"]),
            "--final-time",
            "{:.16g}".format(FINAL_TIME_S),
            "--beta",
            "{:.16g}".format(args.beta),
            "--pressure-stabilization",
            "{:.16g}".format(args.pressure_stabilization),
            "--membrane-shear-scale",
            "{:.16g}".format(args.membrane_shear_scale),
            "--bending-scale",
            "{:.16g}".format(args.bending_scale),
            "--axial-penalty-scale",
            "{:.16g}".format(args.axial_penalty_scale),
            "--symmetry-penalty-scale",
            "{:.16g}".format(args.symmetry_penalty_scale),
            "--profile-bins",
            str(PROFILE_BINS),
            "--snapshot-times-ms",
            snapshot_argument,
            "--case-label",
            case["name"],
        ]
        if not args.write_field_output:
            solver_command.append("--skip-field-output")
        try:
            run_command(
                solver_command,
                output_directory / "solver.log",
                dry_run=args.dry_run,
                environment=solver_environment,
            )
        except subprocess.CalledProcessError as error:
            # The solver writes run_complete.json only after the last time step,
            # all requested profiles, and all CSV files have been completed and
            # closed.  Some macOS MPICH builds can nevertheless return a nonzero
            # status afterward from MPI_Finalize.  Accept only this fully
            # validated post-completion situation; every genuine incomplete run
            # still stops the study immediately.
            if completed_case_matches(
                output_directory,
                case["dt_s"],
                args.beta,
                args.pressure_stabilization,
                args.membrane_shear_scale,
                args.bending_scale,
                args.axial_penalty_scale,
                args.symmetry_penalty_scale,
                args.write_field_output,
            ):
                print(
                    "WARNING: case '{}' returned status {} after writing a valid "
                    "completion marker; accepting the completed case and continuing."
                    .format(case["name"], error.returncode)
                )
            else:
                raise

    if not args.skip_analysis:
        analysis_command = [
            args.python_executable,
            str(analyzer),
            "--study-dir",
            str(study_directory),
        ]
        run_command(
            analysis_command,
            study_directory / "comparison" / "analysis.log",
            dry_run=args.dry_run,
        )

    print("\nConvergence study directory: {}".format(study_directory))
    if args.dry_run:
        print("Dry run complete; no mesh, simulation, or analysis command was executed.")


if __name__ == "__main__":
    main()
