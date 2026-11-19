#!/usr/bin/env python3
"""Compare mapped cross-sectional pressure profiles from convergence runs."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROFILE_FILENAME = "cross_section_pressure_profiles.csv"
COMPARISON_POINTS = 1201


def trapezoidal_integral(values, coordinates):
    """Use the current NumPy name while retaining compatibility with NumPy 1.x."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, coordinates)
    return np.trapz(values, coordinates)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create waveform overlays, profile-difference norms, and observed "
            "orders from a convergence-study manifest."
        )
    )
    parser.add_argument(
        "--study-dir",
        default="convergence_results",
        help="Directory containing study_manifest.json and completed cases.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of the PNG comparison figures.",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    return args


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def read_profile_file(path, requested_times_ms):
    """Read and validate profiles at all requested snapshot times."""
    grouped_rows = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "time_s",
            "reference_x_m",
            "current_cross_section_area_m2",
            "current_cross_section_centroid_x_m",
            "mean_pressure_Pa",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "{} is missing columns: {}".format(
                    path,
                    ", ".join(sorted(missing)),
                )
            )
        for row in reader:
            time_s = float(row["time_s"])
            grouped_rows.setdefault(time_s, []).append(
                {
                    "time_s": time_s,
                    "reference_x_m": float(row["reference_x_m"]),
                    "current_cross_section_area_m2": float(
                        row["current_cross_section_area_m2"]
                    ),
                    "current_cross_section_centroid_x_m": float(
                        row["current_cross_section_centroid_x_m"]
                    ),
                    "mean_pressure_Pa": float(row["mean_pressure_Pa"]),
                }
            )

    profiles = {}
    available_times = np.asarray(sorted(grouped_rows), dtype=float)
    if available_times.size == 0:
        raise ValueError("No pressure profiles were found in {}.".format(path))

    for requested_time_ms in requested_times_ms:
        requested_time_s = 1.0e-3 * float(requested_time_ms)
        nearest_index = int(np.argmin(np.abs(available_times - requested_time_s)))
        stored_time = float(available_times[nearest_index])
        if not math.isclose(
            stored_time,
            requested_time_s,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "{} does not contain the requested profile at {} ms."
                .format(path, requested_time_ms)
            )

        rows = sorted(
            grouped_rows[stored_time],
            key=lambda item: item["reference_x_m"],
        )
        x = np.asarray([row["reference_x_m"] for row in rows], dtype=float)
        pressure = np.asarray([row["mean_pressure_Pa"] for row in rows], dtype=float)
        area = np.asarray(
            [row["current_cross_section_area_m2"] for row in rows],
            dtype=float,
        )
        current_centroid_x = np.asarray(
            [row["current_cross_section_centroid_x_m"] for row in rows],
            dtype=float,
        )
        if len(x) < 2 or np.any(np.diff(x) <= 0.0):
            raise ValueError(
                "Reference axial stations must be strictly increasing in {}."
                .format(path)
            )
        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(pressure))
            and np.all(np.isfinite(area))
            and np.all(np.isfinite(current_centroid_x))
        ):
            raise ValueError("A profile contains a non-finite value in {}.".format(path))
        if np.any(area <= 0.0):
            raise ValueError("A profile contains a nonpositive area in {}.".format(path))

        profiles[float(requested_time_ms)] = {
            "x_m": x,
            "pressure_Pa": pressure,
            "area_m2": area,
            "current_centroid_x_m": current_centroid_x,
        }
    return profiles


def load_cases(study_directory, cases, requested_times_ms):
    loaded = []
    for case in cases:
        case_copy = dict(case)
        case_directory = study_directory / case["output_directory"]
        profile_path = case_directory / PROFILE_FILENAME
        completion_path = case_directory / "run_complete.json"
        if not profile_path.is_file():
            raise FileNotFoundError(
                "Missing profile file for case '{}': {}".format(
                    case["name"],
                    profile_path,
                )
            )
        if not completion_path.is_file():
            raise RuntimeError(
                "Case '{}' has no completion marker: {}".format(
                    case["name"],
                    completion_path,
                )
            )
        completion = load_json(completion_path)
        if not completion.get("completed", False):
            raise RuntimeError("Case '{}' is incomplete.".format(case["name"]))
        case_copy["directory"] = case_directory
        case_copy["profiles"] = read_profile_file(
            profile_path,
            requested_times_ms,
        )
        loaded.append(case_copy)
    return loaded


def spatial_case_label(case):
    return "{} ($h={:.1f}$ mm)".format(
        case["name"].capitalize(),
        1.0e3 * float(case["h_m"]),
    )


def temporal_case_label(case):
    return "$\\Delta t={:.3f}$ ms".format(1.0e3 * float(case["dt_s"]))


def plot_profiles(cases, requested_times_ms, output_base, study_kind, dpi):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.0, 7.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()
    colors = ("#0072B2", "#E69F00", "#009E73")
    line_styles = ("--", "-.", "-")

    all_pressure_kpa = []
    for axis, time_ms in zip(axes, requested_times_ms):
        for index, case in enumerate(cases):
            profile = case["profiles"][float(time_ms)]
            label = (
                spatial_case_label(case)
                if study_kind == "spatial"
                else temporal_case_label(case)
            )
            pressure_kpa = 1.0e-3 * profile["pressure_Pa"]
            all_pressure_kpa.extend(pressure_kpa.tolist())
            axis.plot(
                1.0e3 * profile["x_m"],
                pressure_kpa,
                color=colors[index % len(colors)],
                linestyle=line_styles[index % len(line_styles)],
                linewidth=1.8,
                label=label,
            )
        axis.set_title("$t={:g}$ ms".format(time_ms))
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.set_xlim(0.0, 60.0)

    if all_pressure_kpa:
        lower = min(all_pressure_kpa)
        upper = max(all_pressure_kpa)
        margin = 0.04 * max(upper - lower, 1.0e-6)
        for axis in axes:
            axis.set_ylim(lower - margin, upper + margin)

    for axis in axes[3:]:
        axis.set_xlabel("Reference axial coordinate $x$ (mm)")
    for axis in axes[::3]:
        axis.set_ylabel("Cross-sectional mean pressure (kPa)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=len(cases),
        frameon=False,
    )
    title = (
        "Spatial convergence of pressure-wave profiles"
        if study_kind == "spatial"
        else "Temporal convergence of pressure-wave profiles"
    )
    fig.suptitle(title, y=1.08, fontsize=14)
    fig.savefig(str(output_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def interpolate_pair(first_profile, second_profile):
    lower = max(first_profile["x_m"][0], second_profile["x_m"][0])
    upper = min(first_profile["x_m"][-1], second_profile["x_m"][-1])
    if upper <= lower:
        raise ValueError("Two profiles do not share a nonempty axial interval.")
    x = np.linspace(lower, upper, COMPARISON_POINTS)
    first = np.interp(x, first_profile["x_m"], first_profile["pressure_Pa"])
    second = np.interp(x, second_profile["x_m"], second_profile["pressure_Pa"])
    return x, first, second


def profile_difference(first_profile, second_profile):
    x, first, second = interpolate_pair(first_profile, second_profile)
    difference = first - second
    length = x[-1] - x[0]
    l2 = math.sqrt(float(trapezoidal_integral(difference**2, x) / length))
    linf = float(np.max(np.abs(difference)))
    reference_l2 = math.sqrt(float(trapezoidal_integral(second**2, x) / length))
    reference_linf = float(np.max(np.abs(second)))
    return {
        "l2_difference_Pa": l2,
        "linf_difference_Pa": linf,
        "relative_l2_difference": (
            l2 / reference_l2 if reference_l2 > 0.0 else float("nan")
        ),
        "relative_linf_difference": (
            linf / reference_linf if reference_linf > 0.0 else float("nan")
        ),
    }


def observed_order(first_error, second_error, refinement_ratio):
    if first_error <= 0.0 or second_error <= 0.0 or refinement_ratio <= 1.0:
        return float("nan")
    return math.log(first_error / second_error) / math.log(refinement_ratio)


def write_case_profiles(cases, requested_times_ms, path, study_kind):
    fieldnames = [
        "study",
        "case",
        "h_m",
        "dt_s",
        "time_s",
        "reference_x_m",
        "current_cross_section_area_m2",
        "current_cross_section_centroid_x_m",
        "mean_pressure_Pa",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            for time_ms in requested_times_ms:
                profile = case["profiles"][float(time_ms)]
                for index in range(len(profile["x_m"])):
                    writer.writerow(
                        {
                            "study": study_kind,
                            "case": case["name"],
                            "h_m": case["h_m"],
                            "dt_s": case["dt_s"],
                            "time_s": 1.0e-3 * float(time_ms),
                            "reference_x_m": profile["x_m"][index],
                            "current_cross_section_area_m2": profile["area_m2"][index],
                            "current_cross_section_centroid_x_m": profile[
                                "current_centroid_x_m"
                            ][index],
                            "mean_pressure_Pa": profile["pressure_Pa"][index],
                        }
                    )


def calculate_study_differences(cases, requested_times_ms, resolution_key):
    if len(cases) != 3:
        raise ValueError("The observed-order calculation requires exactly three cases.")

    pair_rows = []
    order_rows = []
    pair_errors_by_time = {}
    for time_ms in requested_times_ms:
        pair_errors = []
        for first_case, second_case in zip(cases[:-1], cases[1:]):
            metrics = profile_difference(
                first_case["profiles"][float(time_ms)],
                second_case["profiles"][float(time_ms)],
            )
            pair_errors.append(metrics)
            row = {
                "time_ms": float(time_ms),
                "coarser_case": first_case["name"],
                "finer_case": second_case["name"],
                "coarser_resolution": float(first_case[resolution_key]),
                "finer_resolution": float(second_case[resolution_key]),
            }
            row.update(metrics)
            pair_rows.append(row)

        first_ratio = (
            float(cases[0][resolution_key]) / float(cases[1][resolution_key])
        )
        second_ratio = (
            float(cases[1][resolution_key]) / float(cases[2][resolution_key])
        )
        if not math.isclose(first_ratio, second_ratio, rel_tol=1.0e-10):
            raise ValueError("Successive refinement ratios must agree.")
        order_rows.append(
            {
                "time_ms": float(time_ms),
                "refinement_ratio": first_ratio,
                "observed_order_l2": observed_order(
                    pair_errors[0]["l2_difference_Pa"],
                    pair_errors[1]["l2_difference_Pa"],
                    first_ratio,
                ),
                "observed_order_linf": observed_order(
                    pair_errors[0]["linf_difference_Pa"],
                    pair_errors[1]["linf_difference_Pa"],
                    first_ratio,
                ),
            }
        )
        pair_errors_by_time[float(time_ms)] = pair_errors

    global_errors = []
    for pair_index in range(2):
        l2_values = np.asarray(
            [
                pair_errors_by_time[float(time_ms)][pair_index][
                    "l2_difference_Pa"
                ]
                for time_ms in requested_times_ms
            ]
        )
        linf_values = np.asarray(
            [
                pair_errors_by_time[float(time_ms)][pair_index][
                    "linf_difference_Pa"
                ]
                for time_ms in requested_times_ms
            ]
        )
        global_errors.append(
            {
                "l2_difference_Pa": float(np.sqrt(np.mean(l2_values**2))),
                "linf_difference_Pa": float(np.max(linf_values)),
            }
        )

    refinement_ratio = (
        float(cases[0][resolution_key]) / float(cases[1][resolution_key])
    )
    global_summary = {
        "coarse_to_medium_l2_difference_Pa": global_errors[0][
            "l2_difference_Pa"
        ],
        "medium_to_fine_l2_difference_Pa": global_errors[1][
            "l2_difference_Pa"
        ],
        "observed_order_l2": observed_order(
            global_errors[0]["l2_difference_Pa"],
            global_errors[1]["l2_difference_Pa"],
            refinement_ratio,
        ),
        "coarse_to_medium_linf_difference_Pa": global_errors[0][
            "linf_difference_Pa"
        ],
        "medium_to_fine_linf_difference_Pa": global_errors[1][
            "linf_difference_Pa"
        ],
        "observed_order_linf": observed_order(
            global_errors[0]["linf_difference_Pa"],
            global_errors[1]["linf_difference_Pa"],
            refinement_ratio,
        ),
    }
    return pair_rows, order_rows, global_summary


def write_rows(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_order(value):
    return "{:.3f}".format(value) if math.isfinite(value) else "not available"


def write_summary(path, spatial_summary, temporal_summary, requested_times_ms):
    lines = [
        "# Pressure-profile convergence summary",
        "",
        "Profiles were compared after piecewise-linear interpolation to a common ",
        "reference-coordinate grid. The reported values are differences between ",
        "successive discrete solutions, not errors with respect to an exact solution.",
        "",
        "The sampled times were {} ms.".format(
            ", ".join("{:g}".format(value) for value in requested_times_ms)
        ),
        "",
        "| Study | Combined root-mean-square difference, coarse--medium (Pa) | "
        "Combined root-mean-square difference, medium--fine (Pa) | "
        "Observed order | Maximum difference, coarse--medium (Pa) | "
        "Maximum difference, medium--fine (Pa) | Observed order |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in (
        ("Spatial", spatial_summary),
        ("Temporal", temporal_summary),
    ):
        lines.append(
            "| {} | {:.6e} | {:.6e} | {} | {:.6e} | {:.6e} | {} |".format(
                name,
                summary["coarse_to_medium_l2_difference_Pa"],
                summary["medium_to_fine_l2_difference_Pa"],
                format_order(summary["observed_order_l2"]),
                summary["coarse_to_medium_linf_difference_Pa"],
                summary["medium_to_fine_linf_difference_Pa"],
                format_order(summary["observed_order_linf"]),
            )
        )
    lines.extend(
        [
            "",
            "An observed order should be interpreted only after confirming that the ",
            "three solutions lie in an asymptotic refinement regime.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    args = parse_arguments()
    study_directory = Path(args.study_dir).expanduser().resolve()
    manifest_path = study_directory / "study_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing study manifest: {}".format(manifest_path))
    manifest = load_json(manifest_path)
    requested_times_ms = [float(value) for value in manifest["snapshot_times_ms"]]

    spatial_cases = load_cases(
        study_directory,
        manifest["spatial_study"]["cases"],
        requested_times_ms,
    )
    temporal_cases = load_cases(
        study_directory,
        manifest["temporal_study"]["cases"],
        requested_times_ms,
    )
    if len(spatial_cases) != 3 or len(temporal_cases) != 3:
        raise ValueError("Both studies must contain exactly three ordered levels.")

    comparison_directory = study_directory / "comparison"
    comparison_directory.mkdir(parents=True, exist_ok=True)

    plot_profiles(
        spatial_cases,
        requested_times_ms,
        comparison_directory / "spatial_pressure_profiles",
        "spatial",
        args.dpi,
    )
    plot_profiles(
        temporal_cases,
        requested_times_ms,
        comparison_directory / "temporal_pressure_profiles",
        "temporal",
        args.dpi,
    )
    write_case_profiles(
        spatial_cases,
        requested_times_ms,
        comparison_directory / "spatial_profiles_combined.csv",
        "spatial",
    )
    write_case_profiles(
        temporal_cases,
        requested_times_ms,
        comparison_directory / "temporal_profiles_combined.csv",
        "temporal",
    )

    spatial_pairs, spatial_orders, spatial_summary = calculate_study_differences(
        spatial_cases,
        requested_times_ms,
        "h_m",
    )
    temporal_pairs, temporal_orders, temporal_summary = calculate_study_differences(
        temporal_cases,
        requested_times_ms,
        "dt_s",
    )

    pair_fields = [
        "time_ms",
        "coarser_case",
        "finer_case",
        "coarser_resolution",
        "finer_resolution",
        "l2_difference_Pa",
        "linf_difference_Pa",
        "relative_l2_difference",
        "relative_linf_difference",
    ]
    order_fields = [
        "time_ms",
        "refinement_ratio",
        "observed_order_l2",
        "observed_order_linf",
    ]
    write_rows(
        comparison_directory / "spatial_profile_differences.csv",
        spatial_pairs,
        pair_fields,
    )
    write_rows(
        comparison_directory / "spatial_observed_orders.csv",
        spatial_orders,
        order_fields,
    )
    write_rows(
        comparison_directory / "temporal_profile_differences.csv",
        temporal_pairs,
        pair_fields,
    )
    write_rows(
        comparison_directory / "temporal_observed_orders.csv",
        temporal_orders,
        order_fields,
    )

    write_summary(
        comparison_directory / "convergence_summary.md",
        spatial_summary,
        temporal_summary,
        requested_times_ms,
    )
    with (comparison_directory / "convergence_summary.json").open("w") as handle:
        json.dump(
            {
                "spatial": spatial_summary,
                "temporal": temporal_summary,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print("Comparison results written to {}.".format(comparison_directory))


if __name__ == "__main__":
    main()
