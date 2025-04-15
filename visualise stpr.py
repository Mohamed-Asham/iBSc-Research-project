import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import logging
import re
import itertools
from scipy.stats import linregress
import json

matplotlib.set_loglevel("warning")

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

# output_dir = r"C:\Users\mamoh\PycharmProjects\GitHub\iBSc-Research-project\Mohamed Asham - Code\Output folder for stpr"

stpr_at_lag_0_dict = {}
stpr_data_dict = {}

def plot_stpr_for_one_vs_population(stpr_data_dict):
    for stpr_key, (lags, stpr_values) in stpr_data_dict.items():
        neuron_id, bin_width, animal_id, comparison_type, movement_type, extracted_date, stimulus_type = stpr_key

        mean_stpr = np.mean(stpr_values)
        std_stpr = np.std(stpr_values)

        plt.figure(figsize=(8, 5))
        plt.plot(lags, stpr_values, marker="o", linestyle="-", color="blue", label=f"Neuron {neuron_id}")
        plt.axvline(0, color="black", linestyle="--", linewidth=1)

        single_neuron_area, population_area = comparison_type.split("-")

        plt.xlabel("Time Lag (s)")
        plt.ylabel("Correlation (STPR)")
        plt.title(f"STPR for Neuron {neuron_id} {single_neuron_area} vs {population_area} Population "
                  f"(Bin Width: {bin_width}s) Animal: {animal_id}_{extracted_date}_{movement_type}_{stimulus_type}")

        stats_text = f"Mean: {mean_stpr:.3f}\nSD: {std_stpr:.3f}"
        plt.text(0.02, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        # output_file = os.path.join(output_dir, f"_{animal_id}_STPR_oneVSpopulation_{bin_width}s_{movement_type}.png")
        # plt.savefig(output_file, bbox_inches="tight", dpi=600)
        # plt.close()
        plt.show()


def plot_stpr_overlayed(stpr_data_dict):
    unique_sessions = sorted(set((key[2], key[5], key[6]) for key in stpr_data_dict.keys()))  # Include stimulus_type
    comparison_types = ["TH-TH", "VCX-VCX", "HCX-HCX", "Undefined-Undefined", "PFC-PFC",
                        "TH-PFC", "VCX-PFC", "HCX-PFC", "Undefined-PFC",
                        "PFC-TH", "PFC-VCX", "PFC-HCX", "PFC-Undefined"]
    unique_movements = sorted(set(key[4] for key in stpr_data_dict.keys()))

    for animal_id, extracted_date, stimulus_type in unique_sessions:
        for movement_type in unique_movements:
            for comparison in comparison_types:
                plt.figure(figsize=(8, 5))
                all_stpr_values = []
                lags_list = []

                for key, (lags, stpr_values) in stpr_data_dict.items():
                    neuron_id, _, animal_key, comparison_key, movement_key, date_key, stim_key = key
                    if (animal_key, date_key, stim_key) == (animal_id, extracted_date, stimulus_type) and \
                            comparison_key == comparison and movement_key == movement_type:
                        plt.plot(lags, stpr_values, color='blue', alpha=0.2)
                        all_stpr_values.append(stpr_values)
                        lags_list.append(lags)

                if all_stpr_values:
                    mean_stpr = np.nanmean(all_stpr_values, axis=0)
                    plt.plot(lags_list[0], mean_stpr, color='black', linewidth=2, label="Mean STPR")
                    overall_mean_stpr = np.nanmean(mean_stpr)
                    plt.text(0.03, 0.95, f"Mean STPR: {overall_mean_stpr:.3f}", transform=plt.gca().transAxes,
                             fontsize=8, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'),
                             verticalalignment='top', horizontalalignment='left')

                plt.axvline(0, color="black", linestyle="--", linewidth=1)
                plt.xlabel("Time Lag (s)")
                plt.ylabel("Correlation (STPR)")
                plt.title(
                    f"STPR Curves for {comparison} in Animal {animal_id}_{extracted_date} ({movement_type}, {stimulus_type})")
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.show()
                
                # output_file = os.path.join(output_dir,f"_{animal_id}_STPR_overlayed_{bin_width}s_{movement_type}.png")
                # plt.savefig(output_file, bbox_inches="tight", dpi=600)
                # plt.close()


def plot_scatter_and_boxplot_stpr_for_mean_of_all_comparisonTypes(stpr_data_dict):
    if not stpr_data_dict:
        logging.warning("stpr_data_dict is empty. No plots will be generated.")
        return

    comparison_types = ["TH-TH", "VCX-VCX", "HCX-HCX", "Undefined-Undefined", "PFC-PFC",
                        "TH-PFC", "VCX-PFC", "HCX-PFC", "Undefined-PFC",
                        "PFC-TH", "PFC-VCX", "PFC-HCX", "PFC-Undefined"]

    colors = ["blue", "red", "green", "purple", "orange", "cyan", "magenta", "yellow", "brown", "pink", "gray", "lime",
              "teal"]

    unique_animals = sorted(set(key[2] for key in stpr_data_dict.keys()))
    unique_movements = sorted(set(key[4] for key in stpr_data_dict.keys()))
    unique_stimulus_types = sorted(set(key[6] for key in stpr_data_dict.keys()))

    for animal_id in unique_animals:
        for movement_type in unique_movements:
            for stimulus_type in unique_stimulus_types:
                # --- First Plot: Original Scatter Plot with Jittered Dots ---
                plt.figure(figsize=(8, 5))
                mean_of_mean_stpr_values = []
                std_stpr_values = []
                stats_text = f"Animal {animal_id} ({movement_type}, {stimulus_type})\n"
                box_plot_data = []  

                for i, comparison in enumerate(comparison_types):
                    mean_stpr_values = []

                    for key, (lags, stpr_curve) in stpr_data_dict.items():
                        _, _, animal_key, comparison_key, movement_key, _, stim_key = key
                        if animal_key == animal_id and comparison_key == comparison and \
                                movement_key == movement_type and stim_key == stimulus_type:
                            mean_stpr = np.mean(stpr_curve)  
                            mean_stpr_values.append(mean_stpr)

                    box_plot_data.append(mean_stpr_values)

                    jitter = np.random.uniform(-0.1, 0.1, size=len(mean_stpr_values))
                    x_positions = np.full(len(mean_stpr_values), i) + jitter
                    plt.scatter(x_positions, mean_stpr_values, color=colors[i], alpha=0.2,
                                label=f"{comparison} Individual Neurons" if i == 0 else "")

                    if mean_stpr_values:
                        mean_stpr_final = np.mean(mean_stpr_values)
                        std_stpr_final = np.std(mean_stpr_values)
                        mean_of_mean_stpr_values.append(mean_stpr_final)
                        std_stpr_values.append(std_stpr_final)
                        stats_text += f"{comparison}: Mean = {mean_stpr_final:.3f}, SD = {std_stpr_final:.3f}\n"
                    else:
                        mean_of_mean_stpr_values.append(np.nan)
                        std_stpr_values.append(np.nan)

                plt.errorbar(comparison_types, mean_of_mean_stpr_values, yerr=std_stpr_values, fmt='o',
                             color="black", capsize=5, label="Mean ± Std")

                plt.gcf().text(0.92, 0.87, stats_text, fontsize=10, bbox=dict(facecolor='white',
                                                                              alpha=0.5, edgecolor='black'),
                               verticalalignment='top', horizontalalignment='left')

                plt.xlabel("Comparison Type")
                plt.ylabel("Mean STPR Value")
                plt.title(f"Mean STPR Values for Animal {animal_id} ({movement_type}, {stimulus_type}) - Scatter")
                plt.xticks(rotation=45)
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.show()

                # output_file = os.path.join(output_dir, f"_{animal_id}_STPR_scatter_{stimulus_type}.png")
                # plt.savefig(output_file, bbox_inches="tight", dpi=600)
                # plt.close()

                # --- Second Plot: Scatter-Like with Box Plots for Means ---
                plt.figure(figsize=(8, 5))

                plt.scatter(range(len(comparison_types)), mean_of_mean_stpr_values, color="black", s=50, alpha=0,
                            label="Mean of Means")

                plt.boxplot(box_plot_data, positions=range(len(comparison_types)), widths=0.3,
                            patch_artist=True, boxprops=dict(facecolor="lightgray", alpha=0.5),
                            medianprops=dict(color="red"), whiskerprops=dict(color="black"),
                            capprops=dict(color="black"), showmeans=False, showfliers=False)

                plt.xticks(range(len(comparison_types)), comparison_types, rotation=45)

                plt.gcf().text(0.92, 0.87, stats_text, fontsize=10, bbox=dict(facecolor='white',
                                                                              alpha=0.5, edgecolor='black'),
                               verticalalignment='top', horizontalalignment='left')

                plt.xlabel("Comparison Type")
                plt.ylabel("Mean STPR Value")
                plt.title(f"Mean STPR Values for Animal {animal_id} ({movement_type}, {stimulus_type}) - Box Plot")
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.show()

                # output_file = os.path.join(output_dir, f"_{animal_id}_Box_plot_{stimulus_type}.png")
                # plt.savefig(output_file, bbox_inches="tight", dpi=600)
                # plt.close()

                # --- Third Plot: Box Plots without whiskers ---

                plt.figure(figsize=(8, 5))

                plt.scatter(range(len(comparison_types)), mean_of_mean_stpr_values, color="black", s=50, alpha=0,
                            label="Mean of Means")

                plt.boxplot(box_plot_data, positions=range(len(comparison_types)), widths=0.3,
                            patch_artist=True, boxprops=dict(facecolor="lightgray", alpha=0.5),
                            medianprops=dict(color="red", linewidth=0),
                            whiskerprops=dict(color="black", linewidth=0),  
                            capprops=dict(color="black", linewidth=0),  
                            showmeans=False, showfliers=False)

                plt.xticks(range(len(comparison_types)), comparison_types, rotation=45)

                plt.gcf().text(0.92, 0.87, stats_text, fontsize=10, bbox=dict(facecolor='white',
                                                                              alpha=0.5, edgecolor='black'),
                               verticalalignment='top', horizontalalignment='left')

                plt.xlabel("Comparison Type")
                plt.ylabel("Mean STPR Value")
                plt.title(f"Mean STPR Values for Animal {animal_id} ({movement_type}, {stimulus_type}) - Box Plot")
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.show()

                # output_file = os.path.join(output_dir, f"_{animal_id}_Box_plot_{stimulus_type}.png")
                # plt.savefig(output_file, bbox_inches="tight", dpi=600)
                # plt.close()


def plot_scatter_and_boxplot_stpr_at_0_lag_for_all_animals(stpr_at_lag_0_dict):
    if not stpr_at_lag_0_dict:
        logging.warning("stpr_at_lag_0_dict is empty. No plots will be generated.")
        return

    comparison_types = ["TH-TH", "VCX-VCX", "HCX-HCX", "Undefined-Undefined", "PFC-PFC",
                        "TH-PFC", "VCX-PFC", "HCX-PFC", "Undefined-PFC",
                        "PFC-TH", "PFC-VCX", "PFC-HCX", "PFC-Undefined"]

    movement_types = sorted(set(k[4] for k in stpr_at_lag_0_dict.keys()))
    stimulus_types = sorted(set(k[6] for k in stpr_at_lag_0_dict.keys()))
    animal_ids = sorted(set(k[2] for k in stpr_at_lag_0_dict.keys()))
    cmap = plt.get_cmap("rainbow", len(animal_ids))
    color_dict = {animal_id: cmap(i) for i, animal_id in enumerate(animal_ids)}

    for movement in movement_types:
        for stimulus in stimulus_types:
            plt.figure(figsize=(8, 5))
            stats_text = ""
            box_plot_data = [] 

            for i, comparison in enumerate(comparison_types):
                values = [(k[2], k[5], v) for k, v in stpr_at_lag_0_dict.items() if
                          k[3] == comparison and k[4] == movement and k[6] == stimulus]
                mean_stpr = np.nanmean([v for _, _, v in values])
                std_stpr = np.nanstd([v for _, _, v in values])
                stats_text += f"{comparison}: Mean = {mean_stpr:.3f}, SD = {std_stpr:.3f}\n"
                box_plot_data.append([v for _, _, v in values])

                jitter = np.random.uniform(-0.1, 0.1, size=len(values))
                x_positions = np.full(len(values), i) + jitter
                for (animal_id, _, stpr_value), x_pos in zip(values, x_positions):
                    plt.scatter(x_pos, stpr_value, color=color_dict[animal_id], alpha=0.3)

                plt.errorbar(i, mean_stpr, yerr=std_stpr, fmt='o', color="black", capsize=5,
                             label="Mean ± Std" if i == 0 else "")

            plt.gcf().text(0.92, 0.87, stats_text, fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'),
                           verticalalignment='top', horizontalalignment='left')
            plt.xticks(range(len(comparison_types)), comparison_types, rotation=45)
            plt.xlabel("Comparison Type")
            plt.ylabel("STPR at Lag 0")
            plt.title(f"Comparison of STPR at Lag 0 Across Conditions ({movement}, {stimulus}) - Scatter")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()

            # --- Second Plot: Scatter-Like with Box Plots for Means ---
            plt.figure(figsize=(8, 5))

            mean_stpr_values = [np.nanmean(data) if data else np.nan for data in box_plot_data]

            plt.scatter(comparison_types, mean_stpr_values, color="black", s=50, alpha=0, label="Mean STPR")

            plt.boxplot(box_plot_data, positions=range(len(comparison_types)), widths=0.3,
                        patch_artist=True, boxprops=dict(facecolor="lightgray", alpha=0.5),
                        medianprops=dict(color="red"), whiskerprops=dict(color="black"),
                        capprops=dict(color="black"), showmeans=False, showfliers=False)

            plt.gcf().text(0.92, 0.87, stats_text, fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'),
                           verticalalignment='top', horizontalalignment='left')

            plt.xticks(range(len(comparison_types)), comparison_types, rotation=45)
            plt.xlabel("Comparison Type")
            plt.ylabel("STPR at Lag 0")
            plt.title(f"Comparison of STPR at Lag 0 Across Conditions ({movement}, {stimulus}) - Box Plot")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()

            # --- Third Plot: Box Plots without whiskers ---

            plt.figure(figsize=(8, 5))

            mean_stpr_values = [np.nanmean(data) if data else np.nan for data in box_plot_data]

            plt.scatter(comparison_types, mean_stpr_values, color="black", s=50, alpha=0, label="Mean STPR")

            plt.boxplot(box_plot_data, positions=range(len(comparison_types)), widths=0.3,
                        patch_artist=True, boxprops=dict(facecolor="lightgray", alpha=0.5),
                        medianprops=dict(color="red", linewidth=0),
                        whiskerprops=dict(color="black", linewidth=0),  
                        capprops=dict(color="black", linewidth=0),  
                        showmeans=False, showfliers=False)

            plt.gcf().text(0.92, 0.87, stats_text, fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'),
                           verticalalignment='top', horizontalalignment='left')

            plt.xticks(range(len(comparison_types)), comparison_types, rotation=45)
            plt.xlabel("Comparison Type")
            plt.ylabel("STPR at Lag 0")
            plt.title(f"Comparison of STPR at Lag 0 Across Conditions ({movement}, {stimulus}) - Box Plot")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
            
            # output_file = os.path.join(output_dir, f"_{animal_id}_STPR_scatter_all_Animals_{bin_width}s_{movement_type}.png")
            # plt.savefig(output_file, bbox_inches="tight", dpi=600)
            # plt.close()


def plot_scatter_stpr_comparison_between_regions(stpr_at_lag_0_dict):
    if not stpr_at_lag_0_dict:
        logging.warning("stpr_at_lag_0_dict is empty. No plots will be generated.")
        return

    comparison_types = ["TH-TH", "VCX-VCX", "HCX-HCX", "Undefined-Undefined", "PFC-PFC",
                        "TH-PFC", "VCX-PFC", "HCX-PFC", "Undefined-PFC",
                        "PFC-TH", "PFC-VCX", "PFC-HCX", "PFC-Undefined"]

    movement_types = sorted(set(k[4] for k in stpr_at_lag_0_dict.keys()))
    stimulus_types = sorted(set(k[6] for k in stpr_at_lag_0_dict.keys()))
    unique_sessions = sorted(set((k[2], k[5], k[6]) for k in stpr_at_lag_0_dict.keys()))
    comparison_pairs = list(itertools.combinations(comparison_types, 2))

    for movement_type in movement_types:
        for stimulus_type in stimulus_types:
            for animal_id, extracted_date, stim in unique_sessions:
                if stim != stimulus_type:
                    continue
                plt.figure(figsize=(15, 10))
                for idx, (comp_x, comp_y) in enumerate(comparison_pairs, 1):
                    plt.subplot(2, 3, idx)
                    values_x = [v for k, v in stpr_at_lag_0_dict.items() if
                                (k[2], k[5], k[6]) == (animal_id, extracted_date, stimulus_type) and k[3] == comp_x and
                                k[4] == movement_type]
                    values_y = [v for k, v in stpr_at_lag_0_dict.items() if
                                (k[2], k[5], k[6]) == (animal_id, extracted_date, stimulus_type) and k[3] == comp_y and
                                k[4] == movement_type]
                    min_length = min(len(values_x), len(values_y))
                    values_x, values_y = values_x[:min_length], values_y[:min_length]

                    if len(values_x) > 0:
                        plt.scatter(values_x, values_y, alpha=0.6, label=f"Neurons ({len(values_x)})")
                        slope, intercept, r_value, _, _ = linregress(values_x, values_y)
                        x_vals = np.array([min(values_x), max(values_x)])
                        y_vals = slope * x_vals + intercept
                        plt.plot(x_vals, y_vals, color="red", linestyle="--", label=f"Best Fit (r={r_value:.2f})")
                        # plt.plot(x_vals, x_vals, color="black", linestyle=":", label="y = x")
                        plt.xlabel(f"{comp_x} STPR at Lag 0")
                        plt.ylabel(f"{comp_y} STPR at Lag 0")
                        plt.title(f"{comp_x} vs {comp_y} ({movement_type}, {stimulus_type})")
                        plt.legend()

                plt.suptitle(
                    f"STPR comparisons for Animal {animal_id}_{extracted_date} ({movement_type}, {stimulus_type})")
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

            plt.figure(figsize=(15, 10))
            for idx, (comp_x, comp_y) in enumerate(comparison_pairs, 1):
                plt.subplot(2, 3, idx)
                values_x_all, values_y_all = [], []
                for animal_id, extracted_date, stim in unique_sessions:
                    if stim != stimulus_type:
                        continue
                    values_x = [v for k, v in stpr_at_lag_0_dict.items() if
                                (k[2], k[5], k[6]) == (animal_id, extracted_date, stimulus_type) and k[3] == comp_x and
                                k[4] == movement_type]
                    values_y = [v for k, v in stpr_at_lag_0_dict.items() if
                                (k[2], k[5], k[6]) == (animal_id, extracted_date, stimulus_type) and k[3] == comp_y and
                                k[4] == movement_type]
                    min_length = min(len(values_x), len(values_y))
                    values_x, values_y = values_x[:min_length], values_y[:min_length]
                    values_x_all.extend(values_x)
                    values_y_all.extend(values_y)

                if len(values_x_all) > 0:
                    plt.scatter(values_x_all, values_y_all, alpha=0.4, label=f"Neurons ({len(values_x_all)})")
                    slope, intercept, r_value, _, _ = linregress(values_x_all, values_y_all)
                    x_vals = np.array([min(values_x_all), max(values_x_all)])
                    y_vals = slope * x_vals + intercept
                    plt.plot(x_vals, y_vals, color="red", linestyle="--", label=f"Best Fit (r={r_value:.2f})")
                    # plt.plot(x_vals, x_vals, color="black", linestyle=":", label="y = x")
                    plt.xlabel(f"{comp_x} STPR at Lag 0")
                    plt.ylabel(f"{comp_y} STPR at Lag 0")
                    plt.title(f"{comp_x} vs {comp_y} (All Animals, {movement_type}, {stimulus_type})")
                    plt.legend()

            plt.suptitle(f"STPR comparisons Across All Animals ({movement_type}, {stimulus_type})")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()


def plot_scatter_and_boxplot_stpr_at_0_lag_per_animal(stpr_at_lag_0_dict):
    if not stpr_at_lag_0_dict:
        logging.warning("stpr_at_lag_0_dict is empty. No plots will be generated.")
        return

    comparison_types = ["TH-TH", "VCX-VCX", "HCX-HCX", "Undefined-Undefined", "PFC-PFC",
                        "TH-PFC", "VCX-PFC", "HCX-PFC", "Undefined-PFC",
                        "PFC-TH", "PFC-VCX", "PFC-HCX", "PFC-Undefined"]

    unique_animals = sorted(set(k[2] for k in stpr_at_lag_0_dict.keys()))  
    movement_types = sorted(set(k[4] for k in stpr_at_lag_0_dict.keys())) 
    stimulus_types = sorted(set(k[6] for k in stpr_at_lag_0_dict.keys()))  

    for movement in movement_types:
        for stimulus in stimulus_types:
            for comparison_type in comparison_types:
                data = []  
                mean_stpr_values = []  

                for animal_id in unique_animals:
                    stpr_values = [v for k, v in stpr_at_lag_0_dict.items() if
                                   k[2] == animal_id and k[3] == comparison_type and k[4] == movement and k[
                                       6] == stimulus]
                    if stpr_values:
                        data.append(stpr_values)
                        mean_stpr_values.append(np.mean(stpr_values))  
                    else:
                        data.append([])
                        mean_stpr_values.append(np.nan)  

                if not any(data): 
                    continue

                plt.figure(figsize=(10, 6))

                positions = range(len(unique_animals)) 
                plt.scatter(positions, mean_stpr_values, color="black", s=50, alpha=0, label="Mean STPR")

                plt.boxplot(data, positions=positions, widths=0.4,
                            patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.5),
                            medianprops=dict(color="red"), whiskerprops=dict(color="black"),
                            capprops=dict(color="black"), flierprops=dict(marker='o', markersize=5, alpha=0.5),
                            showmeans=False, showfliers=False)

                plt.xticks(positions, unique_animals)
                plt.xlabel("Animal ID")
                plt.ylabel(f"STPR at Lag 0")
                plt.title(f"STPR at Lag 0 per Animal for {comparison_type} ({movement}, {stimulus})")
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.show()

                # Box Plots without whiskers --

                plt.figure(figsize=(10, 6))

                positions = range(len(unique_animals))  
                plt.scatter(positions, mean_stpr_values, color="black", s=50, alpha=0, label="Mean STPR")

                plt.boxplot(data, positions=positions, widths=0.4,
                            patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.5),
                            medianprops=dict(color="red", linewidth=0), whiskerprops=dict(color="black", linewidth=0),
                            capprops=dict(color="black", linewidth=0), showmeans=False, showfliers=False)

                plt.xticks(positions, unique_animals)
                plt.xlabel("Animal ID")
                plt.ylabel(f"STPR at Lag 0")
                plt.title(f"STPR at Lag 0 per Animal for {comparison_type} ({movement}, {stimulus})")
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.show()

#==================================================================================


def convert_tuple_keys_to_str(data_dict):
    """Convert tuple keys in a dictionary to strings for JSON serialization."""
    return {str(key): value if not isinstance(value, tuple) else [list(v) for v in value] for key, value in
            data_dict.items()}

def convert_str_keys_to_tuple(data_dict):
    """Convert string keys back to tuples after loading from JSON."""
    converted_dict = {}
    for key, value in data_dict.items():
        tuple_key = tuple(eval(key))
        if isinstance(value, list) and all(isinstance(v, list) for v in value):
            converted_value = tuple(np.array(v) for v in value)
        else:
            converted_value = value
        converted_dict[tuple_key] = converted_value
    return converted_dict

def load_cache(cache_file):
    """Load cached STPR values from a JSON file."""
    global stpr_at_lag_0_dict, stpr_data_dict
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"No cache file found at {cache_file}. Please run the full script to generate the cache first.")
    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        stpr_at_lag_0_dict.update(convert_str_keys_to_tuple(cache['stpr_at_lag_0_dict']))
        stpr_data_dict.update(convert_str_keys_to_tuple(cache['stpr_data_dict']))
        logging.info(
            f"Loaded cache from {cache_file}. Number of entries: stpr_at_lag_0_dict={len(stpr_at_lag_0_dict)}, stpr_data_dict={len(stpr_data_dict)}")
    except Exception as e:
        raise RuntimeError(f"Error loading cache from {cache_file}: {e}")




cache_save_file = r"C:\Users\mamoh\PycharmProjects\GitHub\iBSc-Research-project\Mohamed Asham - Code"
cache_file = os.path.join(cache_save_file, "stpr_cache.json")

load_cache(cache_file)

logging.debug(f"After loading cache: stpr_at_lag_0_dict has {len(stpr_at_lag_0_dict)} entries, stpr_data_dict has {len(stpr_data_dict)} entries")
logging.debug(f"stpr_data_dict keys: {list(stpr_data_dict.keys())}")
logging.debug(f"stpr_at_lag_0_dict keys: {list(stpr_at_lag_0_dict.keys())}")


# plot_stpr_for_one_vs_population(stpr_data_dict)
plot_stpr_overlayed(stpr_data_dict)
# plot_scatter_and_boxplot_stpr_for_mean_of_all_comparisonTypes(stpr_data_dict)
# plot_scatter_and_boxplot_stpr_at_0_lag_for_all_animals(stpr_at_lag_0_dict)
# plot_scatter_stpr_comparison_between_regions(stpr_at_lag_0_dict)
# plot_scatter_and_boxplot_stpr_at_0_lag_per_animal(stpr_at_lag_0_dict)
