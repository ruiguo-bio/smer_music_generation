

import random

import gc
import numpy as np
import copy
import data_convert
import math
from collections import Counter
import pretty_midi
import music21
import sys
# import data_convert
from preprocessing import event_2midi,midi_2event
from data_convert import remi_2midi
import pickle
from preprocessing import remove_empty_track

import re
import logging
import coloredlogs

import os

import tension_calculation
from joblib import Parallel, delayed

control_bins = np.arange(0, 1, 0.1)
tensile_bins = np.arange(0, 2.1, 0.2).tolist() + [4]
diameter_bins = np.arange(0, 4.1, 0.4).tolist() + [5]

tempo_bins = np.array([0] + list(range(60, 190, 30)) + [200])



all_key_names = ['C major', 'G major', 'D major', 'A major',
                 'E major', 'B major', 'F major', 'B- major',
                 'E- major', 'A- major', 'D- major', 'G- major',
                 'A minor', 'E minor', 'B minor', 'F# minor',
                 'C# minor', 'G# minor', 'D minor', 'G minor',
                 'C minor', 'F minor', 'B- minor', 'E- minor',
                 ]

major_enharmonics = {'C#':'D-',
               'D#':'E-',
               'F#':'G-',
               'G#':'A-',
               'A#':'B-'}


minor_enharmonics = {'D-':'C#',
               'D#':'E-',
               'G-':'F#',
               'A-':'G#',
               'A#':'B-'}

all_major_names = np.array(['C major', 'D- major', 'D major', 'E- major',
                   'E major', 'F major', 'G- major', 'G major',
                   'A- major', 'A major', 'B- major', 'B major'])

all_minor_names = np.array(['A minor', 'B- minor', 'B minor', 'C minor',
                   'C# minor', 'D minor', 'E- minor', 'E minor',
                   'F minor', 'F# minor', 'G minor', 'G# minor'])


key_token = [f'k_{num}' for num in range(len(all_key_names))]
key_to_token = {name: f'k_{i}' for i, name in enumerate(all_key_names)}
token_to_key = {v: k for k, v in key_to_token.items()}

def bar_track_density(track_events,track_length):
    total_track_num = 0
    bar_track_note_num = 0

    for track_event in track_events:
        for event_index in range(len(track_event) - 1):
            if track_event[event_index][0] == 'p' and track_event[event_index + 1][0] != 'p':
                total_track_num += 1
                bar_track_note_num += 1
    bar_track_density = bar_track_note_num / track_length


    return bar_track_density



def note_density(track_events, track_length,total_track_length):
    total_track_densities = []
    bar_track_densities = {}
    tracks = track_events.keys()
    for track_name in tracks:
        bar_track_densities[track_name] = []
    # print(tracks)
    for track_name in tracks:
        # print(track_name)
        total_track_num = 0
        bar_track_note_num = 0
        this_track_events = track_events[track_name]
        # print(this_track_events)
        for track_event in this_track_events:
            for event_index in range(len(track_event) - 1):
                if track_event[event_index][0] == 'p' and track_event[event_index + 1][0] != 'p':
                    total_track_num += 1
                    bar_track_note_num += 1

            bar_track_densities[track_name].append(bar_track_note_num / track_length)
            bar_track_note_num = 0
        #         print(note_num / track_length)
        total_track_densities.append(total_track_num / total_track_length)
    return total_track_densities,bar_track_densities


def bar_track_occupation_polyphony_rate(pm, sixteenth_notes_time):
    try:
        piano_roll = pm.get_piano_roll(fs=1 / sixteenth_notes_time)


        if piano_roll.shape[1] == 0:
            bar_occupation_rate = 0
        else:
            bar_occupation_rate = np.count_nonzero(np.any(piano_roll, 0)) / piano_roll.shape[1]
        if np.count_nonzero(np.any(piano_roll, 0)) == 0:
            bar_polyphony_rate = 0
        else:
            bar_polyphony_rate = np.count_nonzero(np.count_nonzero(piano_roll, 0) > 1) / np.count_nonzero(np.any(piano_roll, 0))


        return bar_occupation_rate, bar_polyphony_rate
    except:
        return -1, -1


def occupation_polyphony_rate(pm,bar_sixteenth_note_number,sixteenth_notes_time):
    occupation_rate = []
    polyphony_rate = []
    bar_occupation_rate = {}
    bar_polyphony_rate = {}

    total_roll = pm.get_piano_roll(fs=1/sixteenth_notes_time)

    total_bar_number = math.ceil(total_roll.shape[1] / bar_sixteenth_note_number)

    for inst_idx, instrument in enumerate(pm.instruments):

        piano_roll = instrument.get_piano_roll(fs=1/sixteenth_notes_time)
        if piano_roll.shape[1] == 0:
            occupation_rate.append(0)
        else:
            occupation_rate.append(np.count_nonzero(np.any(piano_roll, 0)) / total_roll.shape[1])
        if np.count_nonzero(np.any(piano_roll, 0)) == 0:
            polyphony_rate.append(0)
        else:
            polyphony_rate.append(
                np.count_nonzero(np.count_nonzero(piano_roll, 0) > 1) / np.count_nonzero(np.any(piano_roll, 0)))

        bar_occupation_rate[inst_idx] = []
        bar_polyphony_rate[inst_idx] = []


        for bar_idx in range(total_bar_number):
            if piano_roll.shape[1] < bar_idx*bar_sixteenth_note_number:
                bar_occupation_rate[inst_idx].append(0)
                bar_polyphony_rate[inst_idx].append(0)
            else:
                this_bar_track_roll = piano_roll[:,bar_idx*bar_sixteenth_note_number:bar_idx*bar_sixteenth_note_number + bar_sixteenth_note_number]

                if np.count_nonzero(np.any(this_bar_track_roll, 0)) == 0:
                    bar_polyphony_rate[inst_idx].append(0)

                    bar_occupation_rate[inst_idx].append(0)
                else:
                    bar_occupation_rate[inst_idx].append(np.count_nonzero(np.any(this_bar_track_roll, 0)) / bar_sixteenth_note_number)

                    bar_polyphony_rate[inst_idx].append(np.count_nonzero(np.count_nonzero(this_bar_track_roll, 0) > 1) / np.count_nonzero(np.any(this_bar_track_roll, 0)))

    return occupation_rate, polyphony_rate,bar_occupation_rate,bar_polyphony_rate


def to_category(array, bins):
    result = []
    for item in array:
        result.append(int(np.where((item - bins) >= 0)[0][-1]))
    return result



def cal_tension(pm,key_name=None):


    result = tension_calculation.extract_notes(pm, 0)

    if result:

        pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result
    else:
        return None
    if key_name is None:
        key_name = tension_calculation.all_key_names

    result = tension_calculation.cal_tension(
        piano_roll, beat_time, beat_indices, down_beat_time,
        down_beat_indices, -1, key_name,sixteenth_time,pm)

    if result:
        tensiles, diameters, key_name,\
        changed_key_name, key_change_beat = result
        if key_change_beat != -1:
            return None
    else:
        return None

    # total_tension = np.array(tensiles) + 0.8 * np.array(diameters)

    # tension_category = to_category(total_tension, tension_bin)

    tensile_category = to_category(tensiles,tensile_bins)
    diameter_category = to_category(diameters, diameter_bins)

    # print(f'key is {key_name}')

    return tensile_category, diameter_category,key_name



def check_remi_event(file_events,header_events):

    new_file_events = file_events
    for event in header_events[::-1]:
        new_file_events = np.insert(new_file_events, 0, event)

    pm = data_convert.remi_2midi(new_file_events.tolist())
    pm = remove_empty_track(pm)
    if pm is None or len(pm.instruments) < 1:
        return None

    if '_' not in new_file_events[1]:
        tempo = float(new_file_events[1])
        tempo_category = int(np.where((tempo - tempo_bins) >= 0)[0][-1])
        new_file_events[1] = f't_{tempo_category}'


    return new_file_events





def remove_continue(file_events,is_continue,header_events):

    bar_pos = np.where(file_events == 'bar')[0]
    new_file_events = []

    for idx,event in enumerate(file_events):
        if event == 'continue' and idx<bar_pos[1] and is_continue:
            continue
        else:
            new_file_events.append(event)


    for event in header_events[::-1]:
        new_file_events = np.insert(new_file_events, 0, event)

    if '_' not in new_file_events[1]:
        tempo = float(new_file_events[1])
        tempo_category = int(np.where((tempo - tempo_bins) >= 0)[0][-1])
        new_file_events[1] = f't_{tempo_category}'


    return new_file_events



def remove_continue_add_control_event(file_events,header_events,key,tensiles,diameters,add_control=False,rest_multi=True,remove_continue=False,add_bar=False):


    bar_pos = np.where(file_events == 'bar')[0]
    new_file_events = []

    for idx,event in enumerate(file_events):
        if event == 'continue' and idx<bar_pos[1] and remove_continue:
            continue
        else:
            new_file_events.append(event)


    for event in header_events[::-1]:
        new_file_events = np.insert(new_file_events, 0, event)

    # pm = pretty_midi.PrettyMIDI(midi_name)
    if rest_multi:
        pm = event_2midi(new_file_events.tolist())[0]
    else:
        pm = data_convert.remi_2midi(new_file_events.tolist())
    pm_new = remove_empty_track(pm)
    if pm_new is None or len(pm_new.instruments) < 1:
        return None

    if '_' not in new_file_events[1]:
        tempo = float(new_file_events[1])
        tempo_category = int(np.where((tempo - tempo_bins) >= 0)[0][-1])
        new_file_events[1] = f't_{tempo_category}'


    if add_control:

        bar_pos = np.where(new_file_events == 'bar')[0]

        total_bars = len(bar_pos)
        if total_bars < 2:
            return None
            # if total_bars < len(bar_pos):
            #     # print(f'total bars is {total_bars}. less than original {len(bar_pos)}')
            #     bar_pos = bar_pos[:total_bars + 1]
            #     new_file_events = new_file_events[:bar_pos[-1]]
            #     bar_pos = bar_pos[:-1]

            # if total_bars < len(tensiles):
            #     # print(f'total bars is {total_bars}. less than tensile {len(tensiles)}')
            #     tensiles = tensiles[:total_bars]
            #     diameters = diameters[:total_bars]

        bar_beats = int(str(header_events[0])[0])
        # track_length: number of 16th notes in a bar
        # total_track_length: number of 16th notes in total
        if bar_beats != 6:
            bar_sixteenth_notes_number = int(bar_beats * 4)
            total_sixteenth_notes_number = bar_sixteenth_notes_number * len(bar_pos)

        else:
            bar_sixteenth_notes_number = int(bar_beats / 2 * 4)
            total_sixteenth_notes_number = bar_sixteenth_notes_number * len(bar_pos)
        #     print(f'bar length is {bar_length}')

        r = re.compile('track_\d')

        track_names = list(set(filter(r.match, new_file_events)))
        track_names.sort()

        track_pos_dict = {}
        for track_idx, track_name in enumerate(track_names):
            track_pos_dict[track_name] = track_idx


        track_events = {}

        for track_name in track_names:
            track_events[track_name] = []

        for bar_index in range(len(bar_pos) - 1):
            bar = bar_pos[bar_index]
            next_bar = bar_pos[bar_index + 1]
            bar_events = new_file_events[bar:next_bar]
            #         print(bar_events)

            track_pos = []

            for track_name in track_names:
                track_pos.append(np.where(track_name == bar_events)[0][0])
            #         print(track_pos)
            track_index = 0
            if len(track_names) == 1:
                track_event = bar_events[track_pos[track_index]:]
                #             print(track_event)
                track_events[track_names[track_index]].append(track_event)

            else:
                for track_index in range(len(track_names)-1):
                    track_event = bar_events[track_pos[track_index]:track_pos[track_index + 1]]
                    track_events[track_names[track_index]].append(track_event)
                #             print(track_event)
                else:
                    track_index += 1
                    track_event = bar_events[track_pos[track_index]:]
                    #             print(track_event)
                    track_events[track_names[track_index]].append(track_event)

        else:
            bar = bar_pos[bar_index+1]
            bar_events = new_file_events[bar:]
            #         print(bar_events)

            track_pos = []

            for track_name in track_names:
                track_pos.append(np.where(track_name == bar_events)[0][0])
            #         print(track_pos)
            track_index = 0
            if len(track_names) == 1:
                track_event = bar_events[track_pos[track_index]:]
                #             print(track_event)
                track_events[track_names[track_index]].append(track_event)

            else:
                for track_index in range(len(track_names) - 1):
                    track_event = bar_events[track_pos[track_index]:track_pos[track_index + 1]]
                    track_events[track_names[track_index]].append(track_event)
                #             print(track_event)
                else:
                    track_index += 1
                    track_event = bar_events[track_pos[track_index]:]
                    #             print(track_event)
                    track_events[track_names[track_index]].append(track_event)
        total_track_densities,bar_track_densities = note_density(track_events, bar_sixteenth_notes_number, total_sixteenth_notes_number)

        # densities = note_density(track_events, track_length,total_track_length)
        total_density_category = to_category(total_track_densities, control_bins)
        for track_name in bar_track_densities.keys():
            bar_track_densities[track_name] = to_category(bar_track_densities[track_name],control_bins)

        # density_category = to_category(densities, control_bins)
        # if rest_multi:
        #     pm = event_2midi(new_file_events.tolist())[0]
        # else:
        #     pm = data_convert.remi_2midi(new_file_events.tolist())
        #
        beat_time = pm.get_beats()
        if int(header_events[0][0]) != 6:
            sixteenth_notes_time = (beat_time[1] - beat_time[0]) / 4
        else:
            sixteenth_notes_time = (beat_time[1] - beat_time[0]) / 6

        occupation_rate, polyphony_rate,bar_occupation_rate,bar_polyphony_rate = occupation_polyphony_rate(pm,bar_sixteenth_notes_number,sixteenth_notes_time)



        if add_bar:
            if len(list(bar_track_densities.values())[0]) != len(bar_pos) or len(list(bar_occupation_rate.values())[0]) != len(bar_pos) or len(list(bar_polyphony_rate.values())[0]) != len(bar_pos):
            # print('invalid')
                return None


        total_occupation_category = to_category(occupation_rate, control_bins)
        total_polyphony_category = to_category(polyphony_rate, control_bins)
        # pitch_register_category = pitch_register(track_events)

        if len(total_density_category) != len(track_names) or len(total_occupation_category) != len(track_names) or len(total_polyphony_category) != len(track_names):
            print('track invalid')
            print(new_file_events)
            return 'what'
        density_token = [f'd_{category}' for category in total_density_category]
        occupation_token = [f'o_{category}' for category in total_occupation_category]
        polyphony_token = [f'y_{category}' for category in total_polyphony_category]
        # pitch_register_token = [f'r_{category}' for category in pitch_register_category]

        # track_control_tokens = density_token + occupation_token + polyphony_token + pitch_register_token


        track_control_tokens = density_token + occupation_token + polyphony_token

        key = key_to_token[key]

        new_file_events = new_file_events.tolist()
        new_file_events.insert(2, key)

        for token in track_control_tokens[::-1]:
            new_file_events.insert(3, token)

        if tensiles is not None:

            tension_positions = np.where(np.array(new_file_events) == track_names[0])[0]
            assert len(tension_positions) == len(bar_pos)
            total_insert = 0


            for i, pos in enumerate(tension_positions):
                new_file_events.insert(pos + total_insert, f's_{tensiles[i]}')
                total_insert += 1
                # new_file_events.insert(pos + total_insert, f'a_{diameters[i]}')
                # total_insert += 1
                # new_file_events.insert(pos + total_insert, f'l_{tension[i]}')
                # total_insert += 1

        if add_bar:
            for track_idx, track_name in enumerate(track_names):

                this_track_bar_occupation = to_category(bar_occupation_rate[track_idx], control_bins)
                this_track_bar_polyphony = to_category(bar_polyphony_rate[track_idx], control_bins)
                bar_track_pos = np.where(np.array(new_file_events) == track_name)[0] + 1
                total_insert = 0
                # assert len(this_track_bar_occupation) == len(bar_pos)
                # assert len(this_track_bar_polyphony) == len(bar_pos)
                for i, pos in enumerate(bar_track_pos):
                    if i > len(bar_track_densities[track_name]):

                        new_file_events.insert(pos + total_insert, 'd_0')
                    else:
                        new_file_events.insert(pos + total_insert, f'd_{bar_track_densities[track_name][i]}')
                    total_insert += 1
                    if i >= len(this_track_bar_occupation):
                        new_file_events.insert(pos + total_insert, 'o_0')
                    else:
                        new_file_events.insert(pos + total_insert, f'o_{this_track_bar_occupation[i]}')
                    total_insert += 1
                    if i >= len(this_track_bar_polyphony):
                        new_file_events.insert(pos + total_insert, 'y_0')
                    else:
                        new_file_events.insert(pos + total_insert, f'y_{this_track_bar_polyphony[i]}')
                    total_insert += 1

        # bar_track_0_density = bar_track_densities
        # for i, pos in enumerate(bar_track_0_pos):


    return new_file_events





# def gen_batches(num_tokens, data_lengths):
#     """
#      Returns the batched data
#              Parameters:
#                      num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
#                      data_lengths (dict): A dict with keys of tuples (length of English sentence, length of corresponding French sentence)
#                                          and values of the indices that correspond to these parallel sentences
#              Returns:
#                      batches (arr): List of each batch (which consists of an array of indices)
#      """
#
#     # Shuffle all the indices
#     for k, v in data_lengths.items():
#         random.shuffle(v)
#
#     batches = []
#     prev_tokens_in_batch = 1e10
#     for k in sorted(data_lengths):
#         # v contains indices of the sentences
#         v = data_lengths[k]
#         total_tokens = (k[0] + k[1]) * len(v)
#
#         # Repeat until all the sentences in this key-value pair are in a batch
#         while total_tokens > 0:
#             tokens_in_batch = min(total_tokens, num_tokens) - min(total_tokens, num_tokens) % (k[0] + k[1])
#             sentences_in_batch = tokens_in_batch // (k[0] + k[1])
#
#             # Combine with previous batch if it can fit
#             if tokens_in_batch + prev_tokens_in_batch <= num_tokens:
#                 batches[-1].extend(v[:sentences_in_batch])
#                 prev_tokens_in_batch += tokens_in_batch
#             else:
#                 batches.append(v[:sentences_in_batch])
#                 prev_tokens_in_batch = tokens_in_batch
#             # Remove indices from v that have been added in a batch
#             v = v[sentences_in_batch:]
#
#             total_tokens = (k[0] + k[1]) * len(v)
#     return batches
#
#
# def load_data(data_path_1, data_path_2, max_seq_length):
#     """
#     Loads the pickle files created in preprocess-data.py
#             Parameters:
#                         data_path_1 (str): Path to the English pickle file processed in process-data.py
#                         data_path_2 (str): Path to the French pickle file processed in process-data.py
#                         max_seq_length (int): Maximum number of tokens in each sentence pair
#
#             Returns:
#                     data_1 (arr): Array of tokenized English sentences
#                     data_2 (arr): Array of tokenized French sentences
#                     data_lengths (dict): A dict with keys of tuples (length of English sentence, length of corresponding French sentence)
#                                          and values of the indices that correspond to these parallel sentences
#     """
#     with open(data_path_1, 'rb') as f:
#         data_1 = pickle.load(f)
#     with open(data_path_2, 'rb') as f:
#         data_2 = pickle.load(f)
#
#     data_lengths = {}
#     for i, (str_1, str_2) in enumerate(zip(data_1, data_2)):
#         if 0 < len(str_1) <= max_seq_length and 0 < len(str_2) <= max_seq_length - 2:
#             if (len(str_1), len(str_2)) in data_lengths:
#                 data_lengths[(len(str_1), len(str_2))].append(i)
#             else:
#                 data_lengths[(len(str_1), len(str_2))] = [i]
#     return data_1, data_2, data_lengths
#
#
# def getitem(idx):
#     """
#     Retrieves a batch given an index
#             Parameters:
#                         idx (int): Index of the batch
#                         data (arr): Array of tokenized sentences
#                         batches (arr): List of each batch (which consists of an array of indices)
#                         src (bool): True if the language is the source language, False if it's the target language
#
#             Returns:
#                     batch (arr): Array of tokenized English sentences, of size (num_sentences, num_tokens_in_sentence)
#                     masks (arr): key_padding_masks for the sentences, of size (num_sentences, num_tokens_in_sentence)
#     """
#
#     event = self.batches[idx]
#     if src:
#         batch = [data[i] for i in sentence_indices]
#     else:
#         # If it's in the target language, add [SOS] and [EOS] tokens
#         batch = [[2] + data[i] + [3] for i in sentence_indices]
#
#     # Get the maximum sentence length
#     seq_length = 0
#     for sentence in batch:
#         if len(sentence) > seq_length:
#             seq_length = len(sentence)
#
#     masks = []
#     for i, sentence in enumerate(batch):
#         # Generate the masks for each sentence, False if there's a token, True if there's padding
#         masks.append([False for _ in range(len(sentence))] + [True for _ in range(seq_length - len(sentence))])
#         # Add 0 padding
#         batch[i] = sentence + [0 for _ in range(seq_length - len(sentence))]
#
#     return np.array(batch), np.array(masks)
#
# #
#
def walk(folder_name,suffix):
    files = []
    for p, d, f in os.walk(folder_name):
        for file_name in f:

            if file_name[-len(suffix):] == suffix:
                files.append(os.path.join(p, file_name))
    return files


def walk_remi(folder_name):
    files = []
    for p, d, f in os.walk(folder_name):
        for file_name in f:

            if file_name[-4:] == 'remi':
                files.append(os.path.join(p, file_name))
    return files


def shift_event_keys(event):
    all_shifted_event = []
    shift_nums = np.random.choice(np.arange(-5,7),5,replace=False)

    for shift_num in shift_nums:
        if shift_num == 0:
            continue
        new_event_list = []

        for token in event:
            if token[0] == 'p':
                new_pitch = int(token[2:]) + shift_num
                if new_pitch > 108:
                    new_pitch -= 12
                if new_pitch < 21:
                    new_pitch += 12
                new_token = 'p_' + str(new_pitch)
                new_event_list.append(new_token)
            else:
                new_event_list.append(token)

        all_shifted_event.extend([new_event_list])
        # pm,_ = event_2midi(event.tolist())
        # pm.write('./temp.mid')

        # pm, _ = event_2midi(new_event_list)
        # pm.write(f'./shifted.mid')
    return all_shifted_event


def shift_event_keys_with_direction(event):
    all_shifted_event = []
    key_idx = int(event[2][2:])
    this_key = all_key_names[key_idx]
    # print(f'this key is {this_key}')
    key_mode = this_key[-5:]
    # key = this_key[:-6]


    if key_mode == 'major':
        # return all_shifted_event
        if random.random() > 0.5:
            if this_key == 'A major':
                target_keys = ['E major']
            elif this_key == 'E major':
                target_keys = ['A major', 'D major']
            elif this_key == 'G major':
                target_keys = ['B major']

            elif this_key == 'B major':
                target_keys = ['G major', 'F major']

            elif this_key == 'B- major':
                target_keys = ['E- major']

            elif this_key == 'E- major':
                target_keys = ['B- major']

            elif this_key == 'E major':
                target_keys = ['A- major']

            elif this_key == 'A- major':
                target_keys = ['D- major']

            elif this_key == 'B- major':
                target_keys = ['G- major']
            else:
                return all_shifted_event



            key_idx = np.where(this_key == all_major_names)[0][0]
            shift_nums = []
            for target_key in target_keys:
                target_idx = np.where(target_key == all_major_names)[0][0]
                idx_diff = target_idx - key_idx
                shift_nums.append(idx_diff)
            for idx,shift_num in enumerate(shift_nums):


                new_event_list = []
                if key_idx + shift_num > 11:
                    new_idx = key_idx + shift_num - 12
                else:
                    new_idx = key_idx + shift_num
                new_key_name = all_major_names[new_idx]
                assert new_key_name == target_keys[idx]
                for token in event:
                    if token[0] == 'p':
                        new_pitch = int(token[2:]) + shift_num
                        if new_pitch > 108:
                            new_pitch -= 12
                        if new_pitch < 21:
                            new_pitch += 12
                        new_token = 'p_' + str(new_pitch)
                        new_event_list.append(new_token)
                    else:
                        new_event_list.append(token)
                new_event_list[2] = key_to_token[new_key_name]
                all_shifted_event.extend([new_event_list])
                # pm,_ = event_2midi(new_event_list)
                # pm.write(f'./{shift_num}.mid')

    else:
        if this_key in ['A minor', 'E minor', 'D minor', 'C minor', 'G minor', 'F minor']:

            key_idx = np.where(this_key == all_minor_names)[0][0]
            for shift_num in range(-5, 7):
                if shift_num == 0:
                    continue
                new_event_list = []
                if key_idx + shift_num > 11:
                    new_idx = key_idx + shift_num - 12
                else:
                    new_idx = key_idx + shift_num
                new_key_name = all_minor_names[new_idx]

                for token in event:
                    if token[0] == 'p':
                        new_pitch = int(token[2:]) + shift_num
                        if new_pitch > 108:
                            new_pitch -= 12
                        if new_pitch < 21:
                            new_pitch += 12
                        new_token = 'p_' + str(new_pitch)
                        new_event_list.append(new_token)
                    else:
                        new_event_list.append(token)
                new_event_list[2] = key_to_token[new_key_name]
                all_shifted_event.extend([new_event_list])
                # pm, _ = event_2midi(new_event_list)
                # pm.write(f'./{shift_num}.mid')
    return all_shifted_event


# # keydata = json.load(open(tension_folder + '/files_result.json','r'))
#
#
def cal_separate_file(files,i,augment=False,add_control=False,rest_multi=True,add_bar=False):

    return_list = []

    # print(f'file {i} {files[i]}')
    # file_events = np.array(pickle.load(open('/home/ruiguo/dataset/lmd/lmd_melody_bass_event_new/A/V/M/TRAVMSO12903CF02EE/2077456af444d348c6e4c241710ff187_event', 'rb')))
    # event, pm = midi_2event('../dataset/0519_first.mid')
    # file_events = np.array(event)

    file_events = np.array(pickle.load(open(files[i], 'rb')))

    if rest_multi:
        total_pm,_ = event_2midi(file_events)
    else:
        total_pm = remi_2midi(file_events)
    if add_control:

        result = cal_tension(total_pm)
        if result:
            tensiles, diameters, first_key = result
            # print(f'first cal key is {first_key}')

        result_list = []
        result_list.append(first_key)
        total_pm.write(files[i] + '.mid')

        s = music21.converter.parse(files[i] + '.mid')
        os.remove(files[i] + '.mid')
        # s = music21.converter.parse(files[i][:-12] + '_remi.mid')

        p = music21.analysis.discrete.KrumhanslSchmuckler()
        p1 = music21.analysis.discrete.TemperleyKostkaPayne()
        p2 = music21.analysis.discrete.BellmanBudge()
        key1 = p.getSolution(s).name
        key2 = p1.getSolution(s).name
        key3 = p2.getSolution(s).name

        key1_name = key1.split()[0].upper()
        key1_mode = key1.split()[1]

        key2_name = key2.split()[0].upper()
        key2_mode = key2.split()[1]

        key3_name = key3.split()[0].upper()
        key3_mode = key3.split()[1]

        if key1_mode == 'major':
            if key1_name in major_enharmonics:
                result_list.append(major_enharmonics[key1_name] + ' ' + key1_mode)
            else:
                result_list.append(key1_name + ' ' + key1_mode)
        else:
            if key1_name in minor_enharmonics:
                result_list.append(minor_enharmonics[key1_name] + ' ' + key1_mode)
            else:
                result_list.append(key1_name + ' ' + key1_mode)

        if key2_mode == 'major':
            if key2_name in major_enharmonics:
                result_list.append(major_enharmonics[key2_name] + ' ' + key2_mode)
            else:
                result_list.append(key2_name + ' ' + key2_mode)
        else:
            if key2_name in minor_enharmonics:
                result_list.append(minor_enharmonics[key2_name] + ' ' + key2_mode)
            else:
                result_list.append(key2_name + ' ' + key2_mode)

        if key3_mode == 'major':
            if key3_name in major_enharmonics:
                result_list.append(major_enharmonics[key3_name] + ' ' + key3_mode)
            else:
                result_list.append(key3_name + ' ' + key3_mode)
        else:
            if key3_name in minor_enharmonics:
                result_list.append(minor_enharmonics[key3_name] + ' ' + key3_mode)
            else:
                result_list.append(key3_name + ' ' + key3_mode)

        count_result = Counter(result_list)

        result_key = ''
        for key, value in count_result.items():
            if value >= 3:
                result_key = key

        if result_key != '' and result_key != first_key:
            # print(f'corrected key is {result_key}')
            result = cal_tension(total_pm,[result_key])
            if result:
                tensiles, diameters, key = result
                # print(f'result key is {key}')
            else:
                return None
        elif result_key == '':
            return None
        else:
            pass


    # folder_name = os.path.dirname(files[i])
    # midi_name = folder_name + '/' + files[i].split('/')[-1].split('_')[0] + '_remi.mid'
    # if not os.path.exists(midi_name):
    #     return None
    # key_file = '/' + files[i][len(event_folder):-6] + '.mid'
    # key = keys[key_file]
    r = re.compile('i_\d')
    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)


    # file_events = pickle.load(open('/home/ruiguo/dataset/chinese_event/今生今世_event', 'rb'))
    # changed_file_events = file_events[1:3+num_of_tracks]
    # changed_file_events.extend(['bar'])
    # changed_file_events.extend( file_events[3+num_of_tracks:])
    # changed_file_events = np.array(changed_file_events)
    # file_events = np.array(pickle.load(open(files[i], 'rb')))


    if num_of_tracks < 1:
        logger.info(f'omit file {files[i]} with no track')
        # return None

    header_events = file_events[:2+num_of_tracks]



    # time_signature = file_events[1]
    # tempo = file_events[2]


    bar_pos = np.where(file_events == 'bar')[0]

    is_continue = False

    if add_control:

        total_bars = min(len(tensiles), len(diameters), len(bar_pos))
        if total_bars < len(bar_pos):
            file_events = file_events[:bar_pos[total_bars]]
        bar_pos = bar_pos[:total_bars]



    bar_beginning_pos = bar_pos[::8]

    # meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0] - 2]
    # meta_without_track_control = np.concatenate([meta_events[0:3],np.array(track_program)],axis=0)
    # # < 16 bar

    if len(bar_beginning_pos) == 1:
        pos = 0
        if add_control:
            if rest_multi:
                is_continue = True
            return_events = remove_continue_add_control_event(file_events[bar_beginning_pos[0]:], header_events, key,
                                                              tensiles, diameters,
                                                              add_control=add_control, rest_multi=rest_multi,
                                                              remove_continue=is_continue,add_bar=add_bar)



        else:
            return_events = remove_continue(file_events[bar_beginning_pos[0]:], True, header_events)

        if return_events is not None:
            if return_events == 'what':
                print(f'skip file {i} bar pos {pos}')
            else:

                return_list.append(return_events)

                if augment:
                    # shift keys to all the key in same mode for 2/4, 6/8, 3/4 time
                    if return_events[0] in ['2/4', '3/4', '6/8']:
                        if random.random() > 0.8:
                            shifted_events = shift_event_keys(return_events)
                            return_list.extend(shifted_events)
                            # print(f'not 4/4, shift key to all')
                    else:
                        if add_control:

                            # print(f'shift key to all')
                            if random.random() > 0.5:
                                shifted_events = shift_event_keys_with_direction(return_events)

                                return_list.extend(shifted_events)
        else:
            pass
            # print(f'skip file {i} bar pos {pos}')
    else:
        for pos in range(len(bar_beginning_pos) - 1):
            if pos == 0:
                is_continue = True
            else:
                is_continue = False

            if add_control:
                tension_pos = int(8 * pos)
                if pos == len(bar_beginning_pos) - 2:

                    # detect empty_event(

                    return_events = remove_continue_add_control_event(file_events[bar_beginning_pos[pos]:], header_events, key,
                                                                      tensiles[tension_pos:],diameters[tension_pos:],add_control=add_control,rest_multi=rest_multi,remove_continue=is_continue,add_bar=add_bar)

                else:
                    return_events = remove_continue_add_control_event(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]],
                                                       header_events, key,tensiles[tension_pos:tension_pos+16],diameters[tension_pos:tension_pos+16], add_control=add_control,rest_multi=rest_multi,remove_continue=is_continue,add_bar=add_bar)
            else:

                if pos == len(bar_beginning_pos) - 2:
                    return_events = remove_continue(file_events[bar_beginning_pos[pos]:],is_continue,header_events)
                else:
                    return_events = remove_continue(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]],is_continue,header_events)



            if return_events is not None:
                if return_events == 'what':
                    print(f'skip file {i} bar pos {pos}')
                else:

                    return_list.append(return_events)

                    if augment:
                        # shift keys to all the key in same mode for 2/4, 6/8, 3/4 time
                        if return_events[0] in ['2/4','3/4','6/8']:
                            if random.random() > 0.8:
                                shifted_events = shift_event_keys(return_events)
                                return_list.extend(shifted_events)
                                # print(f'not 4/4, shift key to all')
                        else:
                            if add_control:

                                # print(f'shift key to all')
                                if random.random() > 0.5:
                                    shifted_events = shift_event_keys_with_direction(return_events)

                                    return_list.extend(shifted_events)
            else:
                pass
                # print(f'skip file {i} bar pos {pos}')


    logger.info(f'number of data of this song is {len(return_list)}')
    if len(return_list) > 0:
        base_name = os.path.basename(files[i])
        base_dir = os.path.dirname(files[i])
        if rest_multi:
            if add_bar:
                dir_name = os.path.abspath(os.path.join(base_dir,'../../smer_bar_track/'))
            else:
                dir_name = os.path.abspath(os.path.join(base_dir, '../smer_track/'))
        else:
            if add_bar:
                dir_name = os.path.abspath(os.path.join(base_dir,'../remi_bar_track/'))
            else:
                dir_name = os.path.abspath(os.path.join(base_dir, '../remi_track/'))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        control_name = base_name + '_control'
        pickle.dump(return_list,open(os.path.join(dir_name,control_name),'wb'))
    return []





def cal_remi_file(files,i,augment=True):

    return_list = []

    logger.info(f'file {i} {files[i]}')
    # file_events = np.array(pickle.load(open('/home/ruiguo/dataset/lmd/lmd_melody_bass_event_new/A/V/M/TRAVMSO12903CF02EE/2077456af444d348c6e4c241710ff187_event', 'rb')))
    # event, pm = midi_2event('../dataset/0519_first.mid')
    # file_events = np.array(event)
    file_events = np.array(pickle.load(open(files[i], 'rb')))
    # key_file = '/' + files[i][len(event_folder):-6] + '.mid'
    # key = keys[key_file]
    r = re.compile('i_\d')
    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)


    # file_events = pickle.load(open('/home/ruiguo/dataset/chinese_event/今生今世_event', 'rb'))
    # changed_file_events = file_events[1:3+num_of_tracks]
    # changed_file_events.extend(['bar'])
    # changed_file_events.extend( file_events[3+num_of_tracks:])
    # changed_file_events = np.array(changed_file_events)
    # file_events = np.array(pickle.load(open(files[i], 'rb')))


    if num_of_tracks < 1:
        logger.info(f'omit file {files[i]} with no track')
        # return None

    header_events = file_events[:2+num_of_tracks]

    # time_signature = file_events[1]
    # tempo = file_events[2]


    bar_pos = np.where(file_events == 'bar')[0]

    bar_beginning_pos = bar_pos[::8]

    # meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0] - 2]
    # meta_without_track_control = np.concatenate([meta_events[0:3],np.array(track_program)],axis=0)
    # # < 16 bar
    for pos in range(len(bar_beginning_pos) - 1):
        if pos == len(bar_beginning_pos) - 2:

            # detect empty_event(
            return_events = check_remi_event(file_events[bar_beginning_pos[pos]:], header_events)
        else:
            return_events = check_remi_event(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]],
                                              header_events)
        if return_events is not None:

            return_list.append(return_events)

            if augment:
                # shift keys to all the key in same mode for 2/4, 6/8, 3/4 time
                if return_events[0] in ['2/4','3/4','6/8']:
                    if random.random() > 0.8:
                        shifted_events = shift_event_keys(return_events)
                        return_list.extend(shifted_events)
                        # print(f'not 4/4, shift key to all')
                # else:
                #
                #         # print(f'shift key to all')
                #     shifted_events = shift_event_keys_with_direction(return_events)
                #
                #     return_list.extend(shifted_events)
        else:
            print(f'skip file {i} bar pos {pos}')


    logger.info(f'number of data of this song is {len(return_list)}')

    return return_list


def add_whole_control_event(file_events,header_events):
    file_events = np.copy(file_events)
    num_of_tracks = len(header_events[2:])

    # if file_events[1] not in time_signature_token:
    #     file_events = np.insert(file_events,1,time_signature)
    #     file_events = np.insert(file_events, 2, tempo)
    #     for i, program in enumerate(header_events[2:]):
    #         file_events = np.insert(file_events, 3+i, program)



    bar_pos = np.where(file_events == 'bar')[0]
    pm = event_2midi(file_events.tolist())[0]
    pm = remove_empty_track(pm)
    if len(pm.instruments) < 1:
        return None

    tensiles,diameters,key,_,_ = cal_tension(pm)

    if tensiles is not None:
        total_bars = min(len(tensiles), len(diameters), len(bar_pos))
        if total_bars < len(bar_pos):
            print(f'total bars is {total_bars}. less than original {len(bar_pos)}')
            bar_pos = bar_pos[:total_bars + 1]
            file_events = file_events[:bar_pos[-1]]
            bar_pos = bar_pos[:-1]

        if total_bars < len(tensiles):
            print(f'total bars is {total_bars}. less than tensile {len(tensiles)}')
            tensiles = tensiles[:total_bars]
            diameters = diameters[:total_bars]



    #     print(f'number of bars is {len(bar_pos)}')
    #     print(f'time signature is {file_event[1]}')
    bar_length = int(file_events[0][0])

    if bar_length != 6:
        bar_length = bar_length * 4 * len(bar_pos)
    else:
        bar_length = bar_length / 2 * 4 * len(bar_pos)
    #     print(f'bar length is {bar_length}')

    track_events = {}

    for i in range(num_of_tracks):
        track_events[f'track_{i}'] = []
    track_names = list(track_events.keys())
    for bar_index in range(len(bar_pos) - 1):
        bar = bar_pos[bar_index]
        next_bar = bar_pos[bar_index + 1]
        bar_events = file_events[bar:next_bar]
        #         print(bar_events)

        track_pos = []

        for track_name in track_names:
            track_pos.append(np.where(track_name == bar_events)[0][0])
        #         print(track_pos)
        track_index = 0
        for track_index in range(len(track_names) - 1):
            track_event = bar_events[track_pos[track_index]:track_pos[track_index + 1]]
            track_events[track_names[track_index]].append(track_event)
        #             print(track_event)
        else:
            if track_index == 0:
                track_event = bar_events[track_pos[track_index]:]
                #             print(track_event)
                track_events[track_names[track_index]].append(track_event)
            else:
                track_index += 1
                track_event = bar_events[track_pos[track_index]:]
                #             print(track_event)
                track_events[track_names[track_index]].append(track_event)

    densities = note_density(track_events, bar_length)
    density_category = to_category(densities, control_bins)
    pm, _ = event_2midi(file_events.tolist())
    occupation_rate, polyphony_rate = occupation_polyphony_rate(pm)
    occupation_category = to_category(occupation_rate, control_bins)
    polyphony_category = to_category(polyphony_rate, control_bins)
    #     print(densities)
    #     print(occupation_rate)
    #     print(polyphony_rate)
    #     print(density_category)
    #     print(occupation_category)
    #     print(polyphony_category)

    #     key_token =  key_to_token[key]

    density_token = [f'd_{category}' for category in density_category]
    occupation_token = [f'o_{category}' for category in occupation_category]
    polyphony_token = [f'y_{category}' for category in polyphony_category]

    track_control_tokens = density_token + occupation_token + polyphony_token

    # print(track_control_tokens)

    file_events = file_events.tolist()



    key = key_to_token[key]
    file_events.insert(2, key)


    for token in track_control_tokens[::-1]:
        file_events.insert(3, token)

    if '_' not in file_events[1]:
        tempo = float(file_events[1])
        tempo_category = int(np.where((tempo - tempo_bins) >= 0)[0][-1])
        file_events[1] = f't_{tempo_category}'

    if tensiles is not None:

        tension_positions = np.where(np.array(file_events) == 'track_0')[0]

        total_insert = 0

        for i, pos in enumerate(tension_positions):
            file_events.insert(pos + total_insert, f's_{tensiles[i]}')
            total_insert += 1
            file_events.insert(pos + total_insert, f'a_{diameters[i]}')
            total_insert += 1

    return file_events

def cal_whole_file(files,i,augment=False):
    return_list = []
    print(f'file {i} {files[i]}')
    # file_events = np.array(pickle.load(open('/home/ruiguo/dataset/lmd/lmd_more_event/R/R/T/TRRRTLE12903CA241F/e88a04b4b6e986efac223636a14d63bb_event', 'rb')))
    file_events = np.array(pickle.load(open(files[i], 'rb')))
    # key_file = '/' + files[i][len(event_folder):-6] + '.mid'
    # key = keys[key_file]
    r = re.compile('i_\d')
    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)

    if num_of_tracks < 1:
        print(f'omit file {files[i]} with no track')
        # return None

    header_events = file_events[:2+num_of_tracks]

    return_events = add_whole_control_event(file_events, header_events)

    if return_events is not None:
        # if key[0] != all_key_names[int(return_events[2][2:])]:
        #     print(f'whole song key is {key[0]}')
        #     if key[2] != -1:
        #         print(f'change key is {key[2]}')
        #     print(f'16 bar key is {all_key_names[int(return_events[2][2:])]}')

        return return_events
    else:
        return None


def add_control_event(file_events,header_events):

    file_events = np.copy(file_events)
    num_of_tracks = len(header_events[2:])

    # if file_events[1] not in time_signature_token:
    #     file_events = np.insert(file_events,1,time_signature)
    #     file_events = np.insert(file_events, 2, tempo)
    #     for i, program in enumerate(header_events[2:]):
    #         file_events = np.insert(file_events, 3+i, program)


    for event in header_events[::-1]:
        file_events = np.insert(file_events, 0, event)

    bar_pos = np.where(file_events == 'bar')[0]
    pm = event_2midi(file_events.tolist())[0]
    pm = remove_empty_track(pm)
    if pm is None or len(pm.instruments) < 1:
        return None

    tensiles,diameters, tension, key = cal_tension(pm,['C major'])


    if tensiles is not None:
        total_bars = min(len(tensiles), len(diameters), len(bar_pos))
        if total_bars < 8:
            return None
        if total_bars < len(bar_pos):
            print(f'total bars is {total_bars}. less than original {len(bar_pos)}')
            bar_pos = bar_pos[:total_bars + 1]
            file_events = file_events[:bar_pos[-1]]
            bar_pos = bar_pos[:-1]

        if total_bars < len(tensiles):
            print(f'total bars is {total_bars}. less than tensile {len(tensiles)}')
            tensiles = tensiles[:total_bars]
            diameters = diameters[:total_bars]
            tension = tension[:total_bars]



    #     print(f'number of bars is {len(bar_pos)}')
    #     print(f'time signature is {file_event[1]}')
    bar_beats = int(file_events[0][0])

    if bar_beats != 6:
        total_sixteenth_notes = bar_beats * 4 * len(bar_pos)
    else:
        total_sixteenth_notes = bar_beats / 2 * 4 * len(bar_pos)
    #     print(f'bar length is {bar_length}')

    track_events = {}

    for i in range(num_of_tracks):
        track_events[f'track_{i}'] = []
    track_names = list(track_events.keys())
    for bar_index in range(len(bar_pos) - 1):
        bar = bar_pos[bar_index]
        next_bar = bar_pos[bar_index + 1]
        bar_events = file_events[bar:next_bar]
        #         print(bar_events)

        track_pos = []

        for track_name in track_names:
            track_pos.append(np.where(track_name == bar_events)[0][0])
        #         print(track_pos)
        track_index = 0
        for track_index in range(len(track_names) - 1):
            track_event = bar_events[track_pos[track_index]:track_pos[track_index + 1]]
            track_events[track_names[track_index]].append(track_event)
        #             print(track_event)
        else:
            if track_index == 0:
                track_event = bar_events[track_pos[track_index]:]
                #             print(track_event)
                track_events[track_names[track_index]].append(track_event)
            else:
                track_index += 1
                track_event = bar_events[track_pos[track_index]:]
                #             print(track_event)
                track_events[track_names[track_index]].append(track_event)

    densities = note_density(track_events, total_sixteenth_notes)
    density_category = to_category(densities, control_bins)
    pm, _ = event_2midi(file_events.tolist())
    occupation_rate, polyphony_rate = occupation_polyphony_rate(pm)
    occupation_category = to_category(occupation_rate, control_bins)
    polyphony_category = to_category(polyphony_rate, control_bins)
    #     print(densities)
    #     print(occupation_rate)
    #     print(polyphony_rate)
    #     print(density_category)
    #     print(occupation_category)
    #     print(polyphony_category)

    #     key_token =  key_to_token[key]

    density_token = [f'd_{category}' for category in density_category]
    occupation_token = [f'o_{category}' for category in occupation_category]
    polyphony_token = [f'y_{category}' for category in polyphony_category]

    track_control_tokens = density_token + occupation_token + polyphony_token

    # print(track_control_tokens)

    file_events = file_events.tolist()



    key = key_to_token[key]
    file_events.insert(2, key)


    for token in track_control_tokens[::-1]:
        file_events.insert(3, token)

    if '_' not in file_events[1]:
        tempo = float(file_events[1])
        tempo_category = int(np.where((tempo - tempo_bins) >= 0)[0][-1])
        file_events[1] = f't_{tempo_category}'

    new_files_events = copy.deepcopy(file_events)
    if tensiles is not None:

        tension_positions = np.where(np.array(file_events) == 'track_0')[0]

        total_insert = 0
        new_total_insert = 0

        for i, pos in enumerate(tension_positions):
            file_events.insert(pos + total_insert, f's_{tensiles[i]}')
            total_insert += 1
            file_events.insert(pos + total_insert, f'a_{diameters[i]}')
            total_insert += 1
            new_files_events.insert(pos + new_total_insert, f's_{tension[i]}')
            new_total_insert += 1

    return file_events,new_files_events




def cal_separate_event(event):


    file_events = np.array(event)
    r = re.compile('i_\d')
    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)
    if num_of_tracks < 2:
        print(f'omit file  with only one track')
        return None

    time_signature = file_events[1]
    tempo = file_events[2]


    bar_pos = np.where(file_events == 'bar')[0]

    bar_beginning_pos = bar_pos[::8]

    # meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0] - 2]
    # meta_without_track_control = np.concatenate([meta_events[0:3],np.array(track_program)],axis=0)
    # # < 16 bar
    for pos in range(len(bar_beginning_pos) - 1):
        if pos == len(bar_beginning_pos) - 2:

            # detect empty_event(
            return_events = add_control_event(file_events[bar_beginning_pos[pos]:], time_signature, tempo,
                                              track_program)
        else:
            return_events = add_control_event(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]],
                                              time_signature, tempo, track_program)
        if return_events is not None:
            return return_events
        else:
            print(f'skip file')



def gen_batches(files,max_token_length=2200,augment=False,add_control=False,rest_multi=True,add_bar=False):


    logger.info(f'total files {len(files)}')


    logger.info(f'augment is {augment}')
    logger.info(f'add control is {add_control}')
    logger.info(f'rest multi is {rest_multi}')
    logger.info(f'bar control is {add_bar}')
    random.seed(99)


    #debug
    return_events = []
    total_number = 0


    # for i in range(44, len(files)):
    #
    #     event = cal_separate_file(files, i,  augment=augment,add_control=add_control,rest_multi=rest_multi)
    #     if event:
    #         # return_events.append(event)
    #         total_number += len(event)

    return_events = Parallel(n_jobs=20)(delayed(cal_separate_file)(files,i,augment=augment,add_control=add_control,rest_multi=rest_multi,add_bar=add_bar) for i in range(0,len(files)))
    # logger.info(f'total number is {len(return_events)}')
    # batches = []
    # for file_events in return_events:
    #     if file_events:
    #         for event in file_events:
    #             batches.append(event)
    #
    # batches.sort(key=len)
    # i = 0
    # while i < len(batches) - 1:
    #     if np.array_equal(batches[i],batches[i + 1]):
    #         del batches[i + 1]
    #     else:
    #         i += 1
    #
    # batches_new = []
    # this_batch_total_length = 0
    #
    # while len(batches) > 0:
    #     if this_batch_total_length + len(batches[0]) < max_token_length:
    #         if len(batches_new) > 0:
    #             batches_new[-1].append(batches[0])
    #         else:
    #             batches_new.append([batches[0]])
    #         this_batch_total_length += len(batches[0])
    #     else:
    #         if len(batches[0]) > max_token_length:
    #             logger.info(
    #                 f'the event size {len(batches[0])} is greater than {max_token_length}, skip this file, or increase the max token length')
    #             this_batch_total_length = 0
    #         else:
    #             batches_new.append([batches[0]])
    #             this_batch_total_length = len(batches[0])
    #     del batches[0]
    # del batches
    # gc.collect()
    # batch_lengths = {}
    # for index, item in enumerate(batches_new):
    #     if len(item) not in batch_lengths:
    #         batch_lengths[len(item)] = [index]
    #     else:
    #         batch_lengths[len(item)].append(index)
    #
    #
    # return batches_new, batch_lengths


def validate_event_data(batches):
    for batch in batches:
        for events in batch:
            print(f'{len(np.where(np.array(events) == "bar")[0])}')
            midi = event_2midi(events)[0]
            midi.write('./temp.mid')
            new_events = midi_2event('./temp.mid')[0]
            print(f'{len(np.where(np.array(new_events) == "bar")[0])}')
            added_control_event = cal_separate_event(new_events)
            print(f'{len(np.where(np.array(added_control_event) == "bar")[0])}')
            # for i,event in enumerate(events):
            if len(added_control_event) < len(events):
                print(f'added event length{len(added_control_event)} is less than { len(events)}')
                # else:
                #     if event != added_control_event[i]:
                #         print('not equal')


# #
# #


# def gen_new_batches(max_token_length=2400, batch_window_size=8):
#
#     batches = pickle.load(open('./sync/new_data','rb'))
#     for i,batch in enumerate(batches):
#         batches[i] = batch.tolist()
#
#     batches.sort(key=len)
#     i = 0
#     while i < len(batches) - 1:
#         if batches[i] == batches[i + 1]:
#             del batches[i + 1]
#         else:
#             i += 1
#
#     batches_new = []
#     this_batch_total_length = 0
#
#     while len(batches) > 0:
#         if this_batch_total_length + len(batches[0]) < max_token_length:
#             if len(batches_new) > 0:
#                 batches_new[-1].append(batches[0])
#             else:
#                 batches_new.append([batches[0]])
#             this_batch_total_length += len(batches[0])
#         else:
#             if len(batches[0]) > max_token_length:
#                 print(
#                     f'the event size {len(batches[0])} is greater than {max_token_length}, skip this file, or increase the max token length')
#                 this_batch_total_length = 0
#             else:
#                 batches_new.append([batches[0]])
#                 this_batch_total_length = len(batches[0])
#         del batches[0]
#     del batches
#     gc.collect()
#     batch_lengths = {}
#     for index, item in enumerate(batches_new):
#         if len(item) not in batch_lengths:
#             batch_lengths[len(item)] = [index]
#         else:
#             batch_lengths[len(item)].append(index)
#     return batches_new, batch_lengths
# #
# #
# all_batches,batch_length = gen_new_batches()
# pickle.dump(all_batches, open('./sync/all_batches_new','wb'))
# pickle.dump(batch_length, open('./sync/batch_length_new','wb'))
# sys.exit()

# #
# vocab = WordVocab(all_tokens)
# event_folder = '/Users/ruiguo/Downloads/score_transformer/jay_event'
# event_folder = '/home/ruiguo/dataset/lmd/lmd_event_corrected_0723/'
# event_folder = '/home/data/guorui/dataset/lmd/only_melody_bass_event'
# event_folder = '/home/ruiguo/dataset/pop909'
# event_folder = '/home/ruiguo/dataset/pop909'


# files_step_multi = walk(event_folder,suffix='step_multi')
# files_rest_single = walk(event_folder,suffix='rest_single')
# # files_chinese = walk('/home/ruiguo/dataset/chinese/event',suffix='event')
# # print(len(files_chinese))
# # assert len(files_sheet) == len(files_remi) == len(files_step_multi) == len(files_rest_single)
# # # # #



#
#
# # #
#
# #
# create_test = False
# logger = logging.getLogger(__name__)
#
# logger.handlers = []
#
# rest_multi = False
# add_control = True
# add_bar = True
#
#
#
# # event_folder = '/home/data/guorui/dataset/lmd/'
#
# event_folder = '/its/home/rg408/dataset/events'
#
#
# # # event_folder = './dataset/lmd_event_corrected_0723/'
# # # event_folder = '/home/ruiguo/dataset/chinese_event'
# files_smer = walk(event_folder+'/smer_events/',suffix='event')
# files_remi = walk(event_folder+'/remi_events/',suffix='step_single')
# print(len(files_smer))
# # all_files_order = pickle.dump(files_smer,open(event_folder + 'files_order','wb'))
#
#
# if rest_multi:
#     files = files_smer
# else:
#     files = files_remi
# #
#
# if create_test:
#     if add_control:
#         if rest_multi:
#             logfile = 'rest_multi_control_test.log'
#         else:
#             logfile = 'step_single_control_test.log'
#     else:
#         if rest_multi:
#             logfile = 'rest_multi_test.log'
#         eulse:
#             logfile = 'step_single_test.log'
# else:
#     if add_control:
#         if rest_multi:
#             logfile = 'dataset_rest_multi_all_control_training_augment.log'
#         else:
#             logfile = 'dataset_step_single_all_control_training_augment.log'
#     else:
#         if rest_multi:
#             logfile = 'dataset_rest_multi_training_augment.log'
#         else:
#             logfile = 'dataset_step_single_training_augment.log'
#
#
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
#                         datefmt='%Y-%m-%d %H:%M:%S', filename=logfile, filemode='w')
#
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# # set a format which is simpler for console use
# formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
#                               datefmt='%Y-%m-%d %H:%M:%S')
# console.setFormatter(formatter)
# logger.addHandler(console)
#
# coloredlogs.install(level='INFO', logger=logger, isatty=True)
#
#
# # #
# # for idx,file_name in enumerate(files):
# #     cal_separate_file(files,idx)
# # keydata = json.load(open(event_folder + '/keys.json','r'))
#
#
# #
# # load a model for key prediction
#
# #
# # checkpoint_epoch = 21
# # config_folder = '/home/data/guorui/wandb/run-20210423_094640-sw0lyk9u/'
# # folder_prefix = '/home/ruiguo/'
# # with open(os.path.join(config_folder,"files/config.yaml")) as file:
# #
# #     config = yaml.full_load(file)
# #
# #
# # vocab = WordVocab(all_tokens)
# # model_prediction = ScoreTransformer(vocab.vocab_size, config['d_model']['value'], config['nhead']['value'], config['num_encoder_layers']['value'],
# #                                  config['num_encoder_layers']['value'], 2048, 2400,
# #                                  0.1, 0.1)
# #
# # model_prediction_dict = torch.load(os.path.join(config_folder,"files/checkpoint_21"))
# # model_prediction_state = model_prediction_dict['model_state_dict']
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #
# # from collections import OrderedDict
# # new_prediction_state_dict = OrderedDict()
# # for k, v in model_prediction_state.items():
# #     name = k[7:]  # remove `module.`
# #     new_prediction_state_dict[name] = v
# #
# # # new_state_dict = model_state
# #
# # model_prediction.load_state_dict(new_prediction_state_dict)
# # model_prediction.to(device)
# #
# # create test data
#
# #
# if create_test:
#     total_events = []
#     total_names = []
#     # start_num = int(len(files) * .9)
#     # end_num = int(len(files) * 1)
#
#     start_num = 0
#     end_num = len(files)
#     print(f'start number {start_num} end number {end_num}')
#     for idx,file_name in enumerate(files):
#         events = cal_separate_file(files,idx,augment=False,add_control=add_control,rest_multi=rest_multi)
#
#         if events:
#             total_events.append(events)
#             # h5_file_name = '/home/ruiguo/dataset/lmd/lmd_matched_h5/' + '/'.join(file_name.split('/')[7:-1]) + '.h5'
#             # with tables.open_file(h5_file_name) as h5:
#             #     print((h5.root.metadata.songs.cols.title[0],
#             #                         h5.root.metadata.songs.cols.artist_name[0]))
#             #     total_names.append((h5.root.metadata.songs.cols.title[0],
#             #                         h5.root.metadata.songs.cols.artist_name[0]))
#
#
#     if rest_multi:
#         pickle.dump(total_events, open(f'/home/data/guorui/score_transformer/sync/rest_multi_no_control_test_batches', 'wb'))
#         # pickle.dump(total_names,
#         #             open(f'/home/data/guorui/score_transformer/sync/rest_multi_no_control_test_batch_names', 'wb'))
#     else:
#         pickle.dump(total_events, open(f'/home/data/guorui/score_transformer/sync/step_single_no_control_test_batches', 'wb'))
#         # pickle.dump(total_names,
#         #             open(f'/home/data/guorui/score_transformer/sync/step_single_no_control_test_batch_names', 'wb'))


import shutil

# sheet_more = 0
# remi_more = 0
#
# sheet_more_number = 0
# remi_more_number = 0
# sheet_0 = pickle.load(open(files_sheet[0], 'rb'))
# remi_0 = pickle.load(open(files_remi[0], 'rb'))
# for i in range(25881):
#     sheet = pickle.load(open(files_sheet[i], 'rb'))
#     remi = pickle.load(open(files_remi[i], 'rb'))
#     if len(sheet) > len(remi):
#         sheet_more_number += len(sheet) - len(remi)
#     else:
#         remi_more_number += len(remi) - len(sheet)
#
#
#     sheet_total += os.path.getsize(one_file)
#     shutil.copy(one_file,'/home/ruiguo/dataset/all_sheet')

#
#
#

# 
# 
# start_num = int(len(files) * .9)
# end_num = int(len(files) * .91)
# # end_num = len(files)
# logger.info(f'start file num is {start_num}')
# logger.info(f'end file num is {end_num}')

# #
# # mock_0 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_mock_batches', 'rb'))
# # mock_3 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_mock_batches', 'rb'))
# #
# # mock_1 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_mock_batch_lengths', 'rb'))
# #
# # mock_2 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_mock_batch_lengths', 'rb'))
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=True,add_control=add_control,rest_multi=rest_multi)
#
# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_mock_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_mock_batch_lengths', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_mock_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_mock_batch_lengths',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_mock_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_mock_batch_lengths',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_mock_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_mock_batch_lengths',
#                          'wb'))
#
#
# pickle.dump(training_all_batches, open(f'./dataset/rest_multi_augment_all_control_mock_batches', 'wb'))
# pickle.dump(training_batch_length,
#             open(f'./dataset/rest_multi_augment_all_control_mock_batch_lengths', 'wb'))
# #
# #
# start_num = int(len(files) * .8)
# end_num = int(len(files) * .9)
# # end_num = len(files)
# logger.info(f'start file num is {start_num}')
# logger.info(f'end file num is {end_num}')
#
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=True,add_control=add_control,rest_multi=rest_multi)
#
# #
# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_validation_batch_lengths', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_validation_batch_lengths',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_validation_batch_lengths',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_validation_batch_lengths',
#                          'wb'))
#
# start_num = int(len(files) * .9)
# end_num = int(len(files) * 1)
# # end_num = len(files)
# logger.info(f'start file num is {start_num}')
# logger.info(f'end file num is {end_num}')
# 
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=True,add_control=add_control,rest_multi=rest_multi)
# 
# 
# #
# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_test_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_test_batch_lengths', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_test_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_test_batch_lengths',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_test_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_test_batch_lengths',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_test_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_test_batch_lengths',
#                          'wb'))
# #
# # #
#
# pickle.dump(files,open('/home/data/guorui/dataset/lmd/smer/total_file_names','wb'))
# start_num = int(len(files) * 0)
# end_num = int(len(files))
# # end_num = len(files)
# logger.info(f'start file num is {start_num}')
# logger.info(f'end file num is {end_num}')


# output_folder = '/home/data/guorui/dataset/lmd/'
# output_folder = '/its/home/rg408/dataset/events'



# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=False,add_control=add_control,rest_multi=rest_multi,add_bar=add_bar)
#
# if add_control:
#     if rest_multi:
#         if add_bar:
#             output_folder = output_folder + '/smer_bar_track'
#             pickle.dump(training_all_batches, open(os.path.join(output_folder,'smer_all_control_bar_training_batches'), 'wb'))
#             pickle.dump(training_batch_length,
#                         open(os.path.join(output_folder, 'smer_all_control_bar_training_batch_lengths'), 'wb'))
#         else:
#             output_folder = output_folder + '/smer_track'
#             pickle.dump(training_all_batches,
#                         open(os.path.join(output_folder, 'smer_all_control_training_batches'), 'wb'))
#             pickle.dump(training_batch_length,
#                         open(os.path.join(output_folder, 'smer_all_control_training_batch_lengths'), 'wb'))
#
#
#     else:
#         if add_bar:
#             output_folder = output_folder + '/remi_bar_track'
#             pickle.dump(training_all_batches,
#                         open(os.path.join(output_folder, 'remi_all_control_bar_training_batches'), 'wb'))
#             pickle.dump(training_batch_length,
#                         open(os.path.join(output_folder, 'remi_all_control_bar_training_batch_lengths'), 'wb'))
#         else:
#             output_folder = output_folder + '/remi_track'
#             pickle.dump(training_all_batches,
#                         open(os.path.join(output_folder, 'remi_all_control_training_batches'), 'wb'))
#             pickle.dump(training_batch_length,
#                         open(os.path.join(output_folder, 'remi_all_control_training_batch_lengths'), 'wb'))
#
#
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_training_batches_0', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_training_batch_lengths_0',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_training_batches_0', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_training_batch_lengths_0',
#                          'wb'))



# #
# start_num = int(len(files) * .6)
# end_num = int(len(files) * .8)
# # end_num = len(files)
# logger.info(f'start file num is {start_num}')
# logger.info(f'end file num is {end_num}')
#
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=True,add_control=add_control,rest_multi=rest_multi)
#
#

# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_training_batches_2', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_training_batch_lengths_2', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batches_2', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batch_lengths_2',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_training_batches_2', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_training_batch_lengths_2',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_training_batches_2', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_training_batch_lengths_2',
#                          'wb'))
#




#
# start_num = int(len(files) * .8)
# end_num = int(len(files) * .9)
# # end_num = len(files)
# logger.info(f'start file num is {start_num}')
# logger.info(f'end file num is {end_num}')
#
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=False,add_control=add_control,rest_multi=rest_multi)
#
# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_validation_batch_lengths', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_validation_batch_lengths',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_validation_batch_lengths',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_validation_batch_lengths',
#                          'wb'))
#


#
#
# if add_control:
#     training_all_batches_0 = pickle.load(open(f'./dataset/rest_multi_augment_all_control_training_batches_0', 'rb'))
#     training_batch_length_0 = pickle.load(open(f'./dataset/rest_multi_augment_all_control_training_batch_lengths_0', 'rb'))
#
#
#     training_all_batches_1 = pickle.load(open(f'./dataset/rest_multi_augment_all_control_training_batches_1', 'rb'))
#     training_batch_length_1 = pickle.load(open(f'./dataset/rest_multi_augment_all_control_training_batch_lengths_1', 'rb'))
# else:
#     training_all_batches_0 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batches_0', 'rb'))
#     training_batch_length_0 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batch_lengths_0', 'rb'))
#
#     training_all_batches_1 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batches_1', 'rb'))
#     training_batch_length_1 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batch_lengths_1', 'rb'))
#
#     training_all_batches_2 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batches_2', 'rb'))
#     training_batch_length_2 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batch_lengths_2', 'rb'))
#     #
#
# length_0 = len(training_all_batches_0)
# length_1 = len(training_all_batches_1)
# training_all_batches_0.extend(training_all_batches_1)
# training_all_batches_0.extend(training_all_batches_2)
#
# length_1_shifted = copy.copy(training_batch_length_1)
# length_2_shifted = copy.copy(training_batch_length_2)
#
# for key1,values1 in length_1_shifted.items():
#     values = [value + length_0 for value in values1]
#     length_1_shifted[key1] = values
#
#
# for key1, values1 in length_1_shifted.items():
#     if key1 in training_batch_length_0:
#         training_batch_length_0[key1].extend(values1)
#     else:
#         training_batch_length_0[key1] = values1
#
#
#
#
# for key2,values2 in length_2_shifted.items():
#     values = [value + length_0 + length_1 for value in values2]
#     length_2_shifted[key2] = values
#
#
# for key2, values2 in length_2_shifted.items():
#     if key2 in training_batch_length_0:
#         training_batch_length_0[key2].extend(values2)
#     else:
#         training_batch_length_0[key2] = values2
#
#
#
# total_length = 0
# for key,values in training_batch_length_0.items():
#     total_length += len(values)
#
#
# if add_control:
#     pickle.dump(training_all_batches_0, open(f'./dataset/rest_multi_augment_all_control_training_batches', 'wb'))
#     pickle.dump(training_batch_length_0,
#                 open(f'./dataset/rest_multi_augment_all_control_training_batch_lengths', 'wb'))
#
# else:
#     pickle.dump(training_all_batches_0,
#                 open(f'./dataset/rest_multi_augment_two_track_training_batches', 'wb'))
#     pickle.dump(training_batch_length_0,
#                 open(f'./dataset/rest_multi_augment_two_track_training_batch_lengths',
#                      'wb'))

#
#
# training_all_batches_0 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batches_0', 'rb'))
# training_batch_length_0 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batch_lengths_0', 'rb'))
#
#
# training_all_batches_1 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batches_1', 'rb'))
# training_batch_length_1 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batch_lengths_1', 'rb'))
# #
# training_all_batches_0.extend(training_all_batches_1)
#
# length_1_shifted = copy.copy(training_batch_length_1)
#
# for key,values in length_1_shifted.items():
#     values = [value + len(training_all_batches_0) - len(training_all_batches_1) for value in values]
#     length_1_shifted[key] = values
#
#
# for key1, values1 in length_1_shifted.items():
#     if key1 in training_batch_length_0:
#         training_batch_length_0[key1].extend(values1)
#     else:
#         training_batch_length_0[key1] = values1
#
#
# total_length = 0
# for key,values in training_batch_length_0.items():
#     total_length += len(values)
#
#
#
# pickle.dump(training_all_batches_0, open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batches', 'wb'))
# pickle.dump(training_batch_length_0,
#             open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batch_lengths', 'wb'))



# print('')


#
#


