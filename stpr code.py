import numpy as np
from numba import jit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import logging
import re
import itertools
from scipy.stats import linregress
import json

matplotlib.set_loglevel("error")
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
def log_blank_line():
    logging.getLogger().handlers[0].stream.write("\n")

folder_path_main = r"----" # <--- Place path to data folder 

stpr_at_lag_0_dict = {}
stpr_data_dict = {}


def find_matching_files(folder_path):
    files = os.listdir(folder_path)
    pattern = r"(M240\d+)_(\d{8}_File\d+)_Shank(\d+)"

    grouped_files = {}

    for file_name in files:
        match = re.match(pattern, file_name)
        if match:
            animal_id = match.group(1)  
            date = match.group(2)  
            shank = match.group(3) 

            key = (animal_id, date) 

            if key not in grouped_files:
                grouped_files[key] = {"Shank0": {}, "Shank1": {}}

            if "SpikeTimes" in file_name:
                grouped_files[key][f"Shank{shank}"]["SpikeTimes"] = os.path.join(folder_path, file_name)
            elif "SpikeLabels" in file_name:
                grouped_files[key][f"Shank{shank}"]["SpikeLabels"] = os.path.join(folder_path, file_name)
            elif "Behaviour" in file_name:
                grouped_files[key][f"Shank{shank}"]["Behaviour"] = os.path.join(folder_path, file_name)

    for key, shank_data in grouped_files.items():
        if all(k in shank_data["Shank0"] for k in ["SpikeTimes", "SpikeLabels", "Behaviour"]) and \
                all(k in shank_data["Shank1"] for k in ["SpikeTimes", "SpikeLabels", "Behaviour"]):
            yield (
                shank_data["Shank0"]["SpikeTimes"], shank_data["Shank1"]["SpikeTimes"],
                shank_data["Shank0"]["SpikeLabels"], shank_data["Shank1"]["SpikeLabels"],
                shank_data["Shank0"]["Behaviour"], shank_data["Shank1"]["Behaviour"] )
    return None, None


def load_spike_data(file_path):
    if re.search(r"_SpikeTimes\.csv$", file_path):
        spike_data = pd.read_csv(file_path, header=None, names=["unit_id", "spike_time"])
        return spike_data
    else:
        raise ValueError(f"Invalid file for SpikeTimes: {file_path}")


def load_behaviour_file(file_path):
    if re.search(r"_Behaviour\.csv$", file_path):
        behaviour_data = pd.read_csv(file_path)
        return behaviour_data
    else:
        raise ValueError(f"Invalid file for Behaviour: {file_path}")


def load_spike_labels_file(file_path):
    if re.search(r"_SpikeLabels\.csv$", file_path):
        spike_labels_data = pd.read_csv(file_path)
        return spike_labels_data
    else:
        raise ValueError(f"Invalid file for SpikeLabels: {file_path}")


def filter_spike_data_by_movement(file_path1, file_path2, behaviour0, behaviour1, filter_for_running=False,
                                  speed_threshold=1.0):
    spike_data1 = load_spike_data(file_path1)
    spike_data2 = load_spike_data(file_path2)
    behaviour_data_0 = load_behaviour_file(behaviour0)
    behaviour_data_1 = load_behaviour_file(behaviour1)

    if "TimeStamp" not in behaviour_data_0.columns or "Speed" not in behaviour_data_0.columns:
        raise ValueError("Shank 0 behaviour file missing 'TimeStamp' or 'Speed'.")
    if "TimeStamp" not in behaviour_data_1.columns or "Speed" not in behaviour_data_1.columns:
        raise ValueError("Shank 1 behaviour file missing 'TimeStamp' or 'Speed'.")

    behaviour_data_0["Speed"] = behaviour_data_0["Speed"].abs()
    behaviour_data_1["Speed"] = behaviour_data_1["Speed"].abs()

    condition = behaviour_data_0["Speed"] >= speed_threshold if filter_for_running else behaviour_data_0[
                                                                                            "Speed"] < speed_threshold
    selected_times_0 = behaviour_data_0[condition]["TimeStamp"].to_numpy()
    condition = behaviour_data_1["Speed"] >= speed_threshold if filter_for_running else behaviour_data_1[
                                                                                            "Speed"] < speed_threshold
    selected_times_1 = behaviour_data_1[condition]["TimeStamp"].to_numpy()

    selected_times_0 = np.sort(selected_times_0)
    selected_times_1 = np.sort(selected_times_1)

    def filter_spikes(spike_times, selected_times, atol=0.0167):
        mask = np.zeros(len(spike_times), dtype=bool)

        spike_times = spike_times.to_numpy()
        indices = np.searchsorted(selected_times, spike_times)

        for i, (spike_time, idx) in enumerate(zip(spike_times, indices)):
            if idx > 0 and np.abs(selected_times[idx - 1] - spike_time) <= atol:
                mask[i] = True
            elif idx < len(selected_times) and np.abs(selected_times[idx] - spike_time) <= atol:
                mask[i] = True

        return mask
    mask_0 = filter_spikes(spike_data1["spike_time"], selected_times_0)
    filtered_shank_0 = spike_data1[mask_0]
    mask_1 = filter_spikes(spike_data2["spike_time"], selected_times_1)
    filtered_shank_1 = spike_data2[mask_1]

    return filtered_shank_0, filtered_shank_1, spike_data1, spike_data2


def filter_spike_data_by_stimulus(spike_data1, spike_data2, behaviour0, behaviour1, filter_for_stimulus_on=False,
                                  atol=0.0167):
    @jit(nopython=True)
    def filter_spikes(spike_times, selected_times):
        mask = np.zeros(len(spike_times), dtype=np.bool_)
        indices = np.searchsorted(selected_times, spike_times)

        for i in range(len(spike_times)):
            idx = indices[i]
            spike_time = spike_times[i]
            if idx > 0 and abs(selected_times[idx - 1] - spike_time) <= atol:
                mask[i] = True
            elif idx < len(selected_times) and abs(selected_times[idx] - spike_time) <= atol:
                mask[i] = True

        return mask

    behaviour_data_0 = load_behaviour_file(behaviour0).copy()
    behaviour_data_1 = load_behaviour_file(behaviour1).copy()

    if "TimeStamp" not in behaviour_data_0.columns or "TimeStamp" not in behaviour_data_1.columns:
        raise ValueError("Behaviour files are missing required 'TimeStamp' column.")
    if "Contrast" not in behaviour_data_0.columns or "Contrast" not in behaviour_data_1.columns:
        logging.info(
            f"Contrast column missing in behaviour files ({behaviour0} or {behaviour1}). Skipping stimulus filtering.")
        return spike_data1, spike_data2

    if filter_for_stimulus_on:
        selected_times_0 = behaviour_data_0[behaviour_data_0["Contrast"] == 1]["TimeStamp"].to_numpy()
        selected_times_1 = behaviour_data_1[behaviour_data_1["Contrast"] == 1]["TimeStamp"].to_numpy()
    else:
        selected_times_0 = behaviour_data_0[behaviour_data_0["Contrast"] == 0]["TimeStamp"].to_numpy()
        selected_times_1 = behaviour_data_1[behaviour_data_1["Contrast"] == 0]["TimeStamp"].to_numpy()

    selected_times_0 = np.sort(selected_times_0)
    selected_times_1 = np.sort(selected_times_1)

    spike_times_0 = spike_data1["spike_time"].to_numpy()
    mask_0 = filter_spikes(spike_times_0, selected_times_0)
    filtered_shank_0 = spike_data1[mask_0]

    spike_times_1 = spike_data2["spike_time"].to_numpy()
    mask_1 = filter_spikes(spike_times_1, selected_times_1)
    filtered_shank_1 = spike_data2[mask_1]

    return filtered_shank_0, filtered_shank_1


def combine_spike_data(filtered_spikeTime0, filtered_spikeTime1, spike_labels_1, spike_labels_2):
    spike_data1 = filtered_spikeTime0
    spike_data2 = filtered_spikeTime1
    spike_labels1 = load_spike_labels_file(spike_labels_1)
    spike_labels2 = load_spike_labels_file(spike_labels_2)
    
    spike_data2["unit_id"] += 10000
    spike_labels2["ClusterID"] += 10000

    spike_data1 = spike_data1.merge(spike_labels1, left_on="unit_id", right_on="ClusterID", how="left")
    spike_data2 = spike_data2.merge(spike_labels2, left_on="unit_id", right_on="ClusterID", how="left")

    combined_spike_data = pd.concat([spike_data1, spike_data2], ignore_index=True)

    return combined_spike_data


def gaussian_kernel(sigma, bin_width, num_sigma=3):
    size = int(np.ceil(2 * num_sigma * sigma / bin_width)) + 1
    if size % 2 == 0:
        size += 1
    t = np.arange(-(size // 2), size // 2 + 1) * bin_width
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def compute_stpr(spike_data, neuron_id, bin_width, animal_id1, extracted_date, animal_movement, max_lag, step_size,
                 compare_across_regions=False, population_spikes_data=None, stimulus_on=False,
                 stimulus_filtering_applied=False,
                 shank0_area=None):
    kernel = gaussian_kernel(sigma=0.1, bin_width=bin_width)

    single_neuron_spikes = spike_data[spike_data["unit_id"] == neuron_id]["spike_time"].values
    single_neuron_area = spike_data[spike_data["unit_id"] == neuron_id]["UnitArea"].iloc[0]

    is_shank0_neuron = neuron_id < 10000 

    if population_spikes_data is None:
        population_spikes_data = spike_data

    if compare_across_regions:

        if is_shank0_neuron:
            population_spikes = population_spikes_data[population_spikes_data["unit_id"] >= 10000][
                "spike_time"]  
            comparison_type = f"{single_neuron_area}-PFC"
        else:
            if shank0_area is None:
                raise ValueError("For Shank1 neurons in cross-regional comparison, shank0_area must be specified.")
            population_spikes = population_spikes_data[
                (population_spikes_data["unit_id"] < 10000) & (population_spikes_data["UnitArea"] == shank0_area)][
                "spike_time"]
            comparison_type = f"PFC-{shank0_area}"
    else:
        if is_shank0_neuron:
            population_spikes = population_spikes_data[
                (population_spikes_data["unit_id"] != neuron_id) & (population_spikes_data["unit_id"] < 10000) &
                (population_spikes_data["UnitArea"] == single_neuron_area)]["spike_time"]
            comparison_type = f"{single_neuron_area}-{single_neuron_area}"
        else:
            population_spikes = population_spikes_data[
                (population_spikes_data["unit_id"] != neuron_id) & (population_spikes_data["unit_id"] >= 10000)][
                "spike_time"]
            comparison_type = "PFC-PFC"

    movement_type = "Running" if animal_movement else "Stationary"
    if stimulus_filtering_applied:
        stimulus_type = "On" if stimulus_on else "Off"
    else:
        stimulus_type = "N/A"

    max_time = spike_data["spike_time"].max()
    bins = np.arange(0, max_time + bin_width, bin_width)

    neuron_spike_rate, _ = np.histogram(single_neuron_spikes, bins=bins)

    neuron_spike_rate_smoothed = np.convolve(neuron_spike_rate, kernel, mode='same')

    sum_of_single_neuron = np.sum(neuron_spike_rate_smoothed)

    if sum_of_single_neuron == 0:
        logging.warning(f"Sum of single neuron rate is zero for neuron {neuron_id}. Returning zero STPR values.")
        num_lags = int(max_lag / step_size)
        lags = np.arange(-num_lags, num_lags + 1) * step_size
        stpr_values = np.zeros(len(lags))
        key = (neuron_id, bin_width, animal_id1, comparison_type, movement_type, extracted_date, stimulus_type)
        stpr_at_lag_0_dict[key] = 0.0
        stpr_data_dict[key] = (lags, stpr_values)
        return lags, stpr_values

    num_lags = int(max_lag / step_size)
    lags = np.arange(-num_lags, num_lags + 1) * step_size
    stpr_values = np.zeros(len(lags))

    for idx, lag in enumerate(lags):
        shifted_population_spikes = population_spikes + lag

        shifted_population_spikes = shifted_population_spikes[shifted_population_spikes >= 0]

        population_rate, _ = np.histogram(shifted_population_spikes, bins=bins)

        min_bins = min(len(neuron_spike_rate), len(population_rate))
        neuron_spike_rate_trimmed = neuron_spike_rate[:min_bins]
        population_rate = population_rate[:min_bins]

        neuron_spike_rate_smoothed_trimmed = np.convolve(neuron_spike_rate_trimmed, kernel, mode='same')
        population_rate_smoothed = np.convolve(population_rate, kernel, mode='same')

        mean_population = np.mean(population_rate_smoothed)

        dot_product = np.dot(neuron_spike_rate_smoothed_trimmed, population_rate_smoothed)

        stpr_values[idx] = (dot_product - sum_of_single_neuron * mean_population) / sum_of_single_neuron

    lag_0_idx = np.where(lags == 0)[0]
    if len(lag_0_idx) > 0:
        stpr_at_lag_0 = stpr_values[lag_0_idx[0]]
    else:
        stpr_at_lag_0 = np.nan
        
    stpr_at_lag_0_dict[(neuron_id, bin_width, animal_id1, comparison_type, movement_type, extracted_date, stimulus_type)] = stpr_at_lag_0

    stpr_data_dict[(neuron_id, bin_width, animal_id1, comparison_type, movement_type, extracted_date, stimulus_type)] = (lags, stpr_values)

    return lags, stpr_values

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def convert_tuple_keys_to_str(data_dict):
    return {str(key): value if not isinstance(value, tuple) else [list(v) for v in value] for key, value in
            data_dict.items()}


def convert_str_keys_to_tuple(data_dict):
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
    global stpr_at_lag_0_dict, stpr_data_dict
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            stpr_at_lag_0_dict.update(convert_str_keys_to_tuple(cache.get('stpr_at_lag_0_dict', {})))
            stpr_data_dict.update(convert_str_keys_to_tuple(cache.get('stpr_data_dict', {})))
            logging.info(f"Loaded cache from {cache_file}\n")
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
    else:
        logging.info(f"No cache file found at {cache_file}. Starting fresh.")


def save_cache(cache_file):
    try:
        existing_cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                existing_cache = json.load(f)
            existing_stpr_at_lag_0_dict = convert_str_keys_to_tuple(existing_cache.get('stpr_at_lag_0_dict', {}))
            existing_stpr_data_dict = convert_str_keys_to_tuple(existing_cache.get('stpr_data_dict', {}))
        else:
            existing_stpr_at_lag_0_dict = {}
            existing_stpr_data_dict = {}

        existing_stpr_at_lag_0_dict.update(stpr_at_lag_0_dict)
        existing_stpr_data_dict.update(stpr_data_dict)

        cache = {
            'stpr_at_lag_0_dict': convert_tuple_keys_to_str(existing_stpr_at_lag_0_dict),
            'stpr_data_dict': convert_tuple_keys_to_str(existing_stpr_data_dict)
        }
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=4)
        logging.info(
            f"Saved cache to {cache_file}. Entries: stpr_at_lag_0_dict={len(existing_stpr_at_lag_0_dict)}, stpr_data_dict={len(existing_stpr_data_dict)}\n")
    except Exception as e:
        logging.error(f"Error saving cache to {cache_file}: {e}")
        raise RuntimeError(f"Failed to save cache: {e}")


def compute_stpr_with_cache(spike_data, neuron_id, bin_width, animal_id1, extracted_date, animal_movement, max_lag,
                            step_size, compare_across_regions=False,
                            population_spikes_data=None, stimulus_on=False, stimulus_filtering_applied=False,
                            shank0_area=None):
    is_shank0_neuron = neuron_id < 10000

    single_neuron_area = spike_data[spike_data["unit_id"] == neuron_id]["UnitArea"].iloc[0]

    if compare_across_regions:
        if is_shank0_neuron:
            comparison_type = f"{single_neuron_area}-PFC"  
        else:
            if shank0_area is None:
                raise ValueError("For Shank1 neurons in cross-regional comparison, shank0_area must be specified.")
            comparison_type = f"PFC-{shank0_area}"
    else:
        if is_shank0_neuron:
            comparison_type = f"{single_neuron_area}-{single_neuron_area}" 
        else:
            comparison_type = "PFC-PFC"

    movement_type = "Running" if animal_movement else "Stationary"

    if stimulus_filtering_applied:
        stimulus_type = "On" if stimulus_on else "Off"
    else:
        stimulus_type = "N/A"

    key = (neuron_id, bin_width, animal_id1, comparison_type, movement_type, extracted_date, stimulus_type)

    if key in stpr_data_dict:
        logging.debug(f"Using cached STPR values for {key}")
        lags, stpr_values = stpr_data_dict[key]
        return lags, stpr_values

    logging.debug(f"Computing STPR values for {key}")
    lags, stpr_values = compute_stpr(spike_data, neuron_id, bin_width, animal_id1, extracted_date, animal_movement,
                                     max_lag, step_size, compare_across_regions,
                                     population_spikes_data, stimulus_on, stimulus_filtering_applied, shank0_area)
    return lags, stpr_values


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

bin_widths = [0.0167]

cache_save_file = r"----"  # <---  Place path to where you want to save the cache file
cache_file = os.path.join(cache_save_file, "stpr_cache.json")

load_cache(cache_file)

for bin_width in bin_widths:

    file_count = 0
    global_prev_data_size = len(stpr_data_dict)
    global_prev_lag_0_size = len(stpr_at_lag_0_dict)

    for files in find_matching_files(folder_path_main):
        file_count += 1
        logging.info(f"Processing file group {file_count}: {files}\n")

        spike_times0, spike_times1, spike_labels0, spike_labels1, behaviour0, behaviour1 = files


        def extract_mouse_id(file_path):
            pattern = r"M240(\d+)_(\d{8})_File\d+_Shank(\d+)"
            match = re.search(pattern, file_path)
            if match:
                return f"{match.group(1)}_Shank_{match.group(3)}", f"{match.group(2)}", f"{match.group(1)}"
            else:
                return "UNKNOWN", "UNKNOWN", "UNKNOWN"


        extracted_id1, extracted_date, animal_id1 = extract_mouse_id(spike_times0)
        extracted_id2, _, _ = extract_mouse_id(spike_times1)
        extracted_id_sl1, _, _ = extract_mouse_id(spike_labels0)
        extracted_id_sl2, _, _ = extract_mouse_id(spike_labels1)
        extracted_id_bh1, _, _ = extract_mouse_id(behaviour0)
        extracted_id_bh2, _, _ = extract_mouse_id(behaviour1)

        logging.debug(f"SpikeTimes 1:  {extracted_id1} --- SpikeTimes 2:  {extracted_id2}")
        logging.debug(f"SpikeLabels 1: {extracted_id_sl1} --- SpikeLabels 2: {extracted_id_sl2}")
        logging.debug(f"Behaviour 1:   {extracted_id_bh1} --- Behaviour 2:   {extracted_id_bh2}\n")

        for animal_movement in [False]:
            logging.info(f"Filtering for movement: '{'Running' if animal_movement else 'Stationary'}' for animal {animal_id1} ({extracted_date})")

            filtered_spikeTime0, filtered_spikeTime1, all_spikeTime0, all_spikeTime1 = filter_spike_data_by_movement(
                spike_times0, spike_times1,
                behaviour0, behaviour1, filter_for_running=animal_movement, speed_threshold=1.0)

            for stimulus_on in [True, False]:
                logging.info(f"Filtering for stimulus: '{'On' if stimulus_on else 'Off'}' for animal {animal_id1} ({extracted_date})\n")

                filtered_spikeTime0_stim, filtered_spikeTime1_stim = filter_spike_data_by_stimulus(filtered_spikeTime0,
                                                                    filtered_spikeTime1, behaviour0, behaviour1,
                                                                    filter_for_stimulus_on=stimulus_on)

                stimulus_filtering_applied = not (filtered_spikeTime0_stim.equals(filtered_spikeTime0) and filtered_spikeTime1_stim.equals(filtered_spikeTime1))

                if stimulus_filtering_applied:
                    filtered_spikeTime0 = filtered_spikeTime0_stim
                    filtered_spikeTime1 = filtered_spikeTime1_stim

                logging.debug("Filtered Spike Times 0 (First & Last 5 Rows):\n%s\n",
                              pd.concat([filtered_spikeTime0.head(), filtered_spikeTime0.tail()]))
                logging.debug("Filtered Spike Times 1 (First & Last 5 Rows):\n%s\n",
                              pd.concat([filtered_spikeTime1.head(), filtered_spikeTime1.tail()]))

                spike_data = combine_spike_data(filtered_spikeTime0, filtered_spikeTime1, spike_labels0, spike_labels1)
                population_spike_data = combine_spike_data(all_spikeTime0, all_spikeTime1, spike_labels0, spike_labels1)

                logging.debug(
                    f"Combined spike data: {len(spike_data)} spikes, Population data: {len(population_spike_data)} spikes")
                logging.debug("Combined Spike Data (First & Last 5 Rows):\n%s",
                              pd.concat([spike_data.head(), spike_data.tail()]))

                shank0_areas = spike_data[spike_data["unit_id"] < 10000]["UnitArea"].unique()
                logging.debug(f"Unique brain areas in Shank0: {shank0_areas}")

                for compare_across_regions in [False, True]:
                    log_blank_line()
                    logging.info(f"Compare across regions: {compare_across_regions}")

                    for neuron_id in spike_data["unit_id"].unique():

                        is_shank0_neuron = neuron_id < 10000

                        if compare_across_regions and not is_shank0_neuron:
                            for shank0_area in shank0_areas:
                                lags, stpr_values = compute_stpr_with_cache(
                                    spike_data, neuron_id, bin_width, animal_id1, extracted_date,
                                    animal_movement, max_lag=1, step_size=0.01,
                                    compare_across_regions=compare_across_regions,
                                    population_spikes_data=population_spike_data,
                                    stimulus_on=stimulus_on,
                                    stimulus_filtering_applied=stimulus_filtering_applied,
                                    shank0_area=shank0_area)
                                logging.debug(f"STPR computed for neuron {neuron_id} vs {shank0_area}")
                        else:
                            single_neuron_area = spike_data[spike_data["unit_id"] == neuron_id]["UnitArea"].iloc[0]

                            logging.debug(f"Computing STPR for neuron {neuron_id} | Area: {single_neuron_area}")
                            lags, stpr_values = compute_stpr_with_cache(
                                spike_data, neuron_id, bin_width, animal_id1, extracted_date,
                                animal_movement, max_lag=1, step_size=0.01,
                                compare_across_regions=compare_across_regions,
                                population_spikes_data=population_spike_data,
                                stimulus_on=stimulus_on,
                                stimulus_filtering_applied=stimulus_filtering_applied)
                  
                log_blank_line()
                logging.info(f"Saving cache for {animal_id1}, {extracted_date}, movement: {'Running' if animal_movement else 'Stationary'}, stimulus: {'On' if stimulus_on else 'Off'}")
                save_cache(cache_file)

    new_global_data_size = len(stpr_data_dict)
    new_global_lag_0_size = len(stpr_at_lag_0_dict)
    if new_global_data_size == global_prev_data_size and new_global_lag_0_size == global_prev_lag_0_size:
        logging.warning(f"No new STPR values added across all files")
        break

logging.info("Script completed")



