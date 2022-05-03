import numpy as np
from einops import rearrange

import os
import logging
import coloredlogs
import argparse
import copy
import torch
import random


import sys

from preprocessing import event_2midi,bar_event_2_midi
from data_convert import remi_2midi
import re
import itertools
from vocab import *
import vocab as vocab_class
from model import ScoreTransformer
import tension_calculation
from create_dataset import note_density, occupation_polyphony_rate,to_category,bar_track_density,bar_track_occupation_polyphony_rate


control_bins = np.arange(0, 1, 0.1)
tensile_bins = np.arange(0, 2.1, 0.2).tolist() + [4]
diameter_bins = np.arange(0, 4.1, 0.4).tolist() + [5]

tempo_bins = np.array([0] + list(range(60, 190, 30)) + [200])
tension_bin = np.arange(0,6.5,0.5)
tension_bin[-1] = 6.5


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



def cal_bar_tension(bar_tokens,headers,key_names=None):

    new_tokens = []
    new_tokens.append('bar')
    for event in bar_tokens:
        if event != 'continue' and event != '<eos>':
            new_tokens.append(event)

    pm = bar_event_2_midi(new_tokens,headers,)


    result = tension_calculation.extract_notes(pm, 3)

    if result:

        pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result
    else:
        return None

    if key_names is None:
        key_names = tension_calculation.all_key_names

    result = tension_calculation.cal_tension(
        piano_roll, beat_time, beat_indices, down_beat_time,
        down_beat_indices, -1, key_names,sixteenth_time,pm)

    if result:
        tensiles, diameters, key_name,\
        changed_key_name, key_change_beat = result
    else:
        return None



    tensile_category = to_category(tensiles,tensile_bins)
    diameter_category = to_category(diameters, diameter_bins)

    # print(f'key is {key_name}')

    return tensile_category, diameter_category,key_name



def cal_tension(pm,key_names=None):


    result = tension_calculation.extract_notes(pm, 3)

    if result:

        pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result
    else:
        return None

    if key_names is None:
        key_names = tension_calculation.all_key_names

    result = tension_calculation.cal_tension(
        piano_roll, beat_time, beat_indices, down_beat_time,
        down_beat_indices, -1, key_names,sixteenth_time,pm)

    if result:
        tensiles, diameters, key_name,\
        changed_key_name, key_change_beat = result
    else:
        return None



    tensile_category = to_category(tensiles,tensile_bins)
    diameter_category = to_category(diameters, diameter_bins)

    # print(f'key is {key_name}')

    return tensile_category, diameter_category,key_name
#
# def cal_bar_tension_control(bar_events,key):
#     return
def cal_bar_track_control(track_events,headers,sixteenth_time):
#
#     ###
    file_events = []
    for event in track_events:
        if event != 'continue':
            file_events.append(event)

#
#     bar_pos = np.where(file_events == 'bar')[0]
#
#     total_bars = len(bar_pos)
#
    bar_beats = int(headers[0][0])
#     # track_length: number of 16th notes in a bar
#     # total_track_length: number of 16th notes in total
    if bar_beats != 6:
        bar_sixteenth_notes_number = int(bar_beats * 4)


    else:
        bar_sixteenth_notes_number = int(bar_beats / 2 * 4)


    pm = bar_event_2_midi(['bar'] + file_events, headers)

    bar_track_densities = bar_track_density(file_events, bar_sixteenth_notes_number)
    bar_density_category = to_category([bar_track_densities], control_bins)


    occupation, polyphony = bar_track_occupation_polyphony_rate(pm,sixteenth_time)

    if occupation == -1 or polyphony == -1:
        return bar_density_category, -1, -1

    bar_occupation_category = to_category([occupation], control_bins)
    bar_polyphony_category = to_category([polyphony], control_bins)

    return bar_density_category, bar_occupation_category, bar_polyphony_category


def cal_track_control(file_events,pm):

    ###
    file_events = np.array(file_events)
    bar_pos = np.where(file_events == 'bar')[0]

    total_bars = len(bar_pos)

    bar_beats = int(file_events[0][0])
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

    track_names = list(set(filter(r.match, file_events)))
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
        bar_events = file_events[bar:next_bar]
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

    else:
        bar = bar_pos[bar_index + 1]
        bar_events = file_events[bar:]
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
    total_track_densities, bar_track_densities = note_density(track_events, bar_sixteenth_notes_number,
                                                              total_sixteenth_notes_number)

    # densities = note_density(track_events, track_length,total_track_length)
    total_density_category = to_category(total_track_densities, control_bins)
    # for track_name in bar_track_densities.keys():
    #     bar_track_densities[track_name] = to_category(bar_track_densities[track_name], control_bins)

    # density_category = to_category(densities, control_bins)
    # if rest_multi:
    #     pm = event_2midi(new_file_events.tolist())[0]
    # else:
    #     pm = data_convert.remi_2midi(new_file_events.tolist())
    #
    beat_time = pm.get_beats()
    if int(file_events[0][0]) != 6:
        sixteenth_notes_time = (beat_time[1] - beat_time[0]) / 4
    else:
        sixteenth_notes_time = (beat_time[1] - beat_time[0]) / 6

    occupation_rate, polyphony_rate, bar_occupation_rate, bar_polyphony_rate = occupation_polyphony_rate(pm,
                                                                                                         bar_sixteenth_notes_number,
                                                                                                         sixteenth_notes_time)
    total_occupation_category = to_category(occupation_rate, control_bins)
    total_polyphony_category = to_category(polyphony_rate, control_bins)


    density_token = [f'd_{category}' for category in total_density_category]
    occupation_token = [f'o_{category}' for category in total_occupation_category]
    polyphony_token = [f'y_{category}' for category in total_polyphony_category]

    track_control_tokens = density_token + occupation_token + polyphony_token

    return track_control_tokens,bar_track_densities,bar_occupation_rate,bar_polyphony_rate


def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


def sampling(logit,vocab, p=None, t=1.0,no_pitch=False,no_duration=False,no_rest=False,no_whole_duration=False,no_eos=False,no_continue=False,no_sep=False,is_density=False,is_polyphony=False,is_occupation=False,is_tensile=False,no_control=False):
    logit = logit.squeeze().cpu().numpy()
    if no_pitch:

        logit = np.array([-100 if i in vocab.pitch_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_duration:
        logit = np.array([-100 if i in vocab.duration_only_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_continue:
        logit = np.array([-100 if i == vocab.continue_index else logit[i] for i in range(vocab.vocab_size)])

    if no_rest:
        logit = np.array([-100 if i in vocab.rest_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_sep:
        logit = np.array([-100 if i in vocab.sep_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_whole_duration:
        logit = np.array([-100 if i == vocab.duration_only_indices[0] else logit[i] for i in range(vocab.vocab_size)])

    if no_eos:
        logit = np.array([-100 if i == vocab.eos_index else logit[i] for i in range(vocab.vocab_size)])

    if is_density:
        logit = np.array([-100 if i not in vocab.density_indices else
                             logit[i] for i in range(vocab.vocab_size)])

    if is_occupation:
        logit = np.array([-100 if i not in vocab.occupation_indices else
                             logit[i] for i in range(vocab.vocab_size)])

    if is_polyphony:
        logit = np.array([-100 if i not in vocab.polyphony_indices else
                             logit[i] for i in range(vocab.vocab_size)])

    if is_tensile:
        logit = np.array([-100 if i not in vocab.tensile_indices else
                             logit[i] for i in range(vocab.vocab_size)])


    logit = np.array([-100 if i in vocab.program_indices + vocab.structure_indices + vocab.time_signature_indices + vocab.tempo_indices  else logit[i] for i in range(vocab.vocab_size)])
    if no_control:
        logit = np.array([-100 if i in vocab.control_indices.values() else
                             logit[i] for i in range(vocab.vocab_size)])

    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word




def sampling_step_single(logit,vocab, p=None, t=1.0,no_pitch=False,no_duration=False,no_step=False):
    logit = logit.squeeze().cpu().numpy()
    if no_pitch:

        logit = np.array([-100 if i in vocab.pitch_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_duration:
        logit = np.array([-100 if i in vocab.duration_only_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_step:
        logit = np.array([-100 if i in vocab.step_indices else logit[i] for i in range(vocab.vocab_size)])


    logit = np.array([-100 if i in vocab.program_indices + vocab.structure_indices + vocab.time_signature_indices + vocab.tempo_indices  else logit[i] for i in range(vocab.vocab_size)])

    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word




def gen_nopeek_mask(length):
    """
     Returns the nopeek mask
             Parameters:
                     length (int): Number of tokens in each sentence in the target batch
             Returns:
                     mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                      [0., 0., -inf],
                                                      [0., 0., 0.]]
     """
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


def model_generate(model, src, tgt,device,return_weights=False):

    src = src.clone().detach().unsqueeze(0).long().to(device)
    tgt = torch.tensor(tgt).unsqueeze(0).to(device)
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    tgt_mask = tgt_mask.clone().detach().unsqueeze(0).to(device)


    output,weights = model.forward(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                           tgt_mask=tgt_mask)
    if return_weights:
        return output.squeeze(0).to('cpu'), weights.squeeze(0).to('cpu')
    else:
        return output.squeeze(0).to('cpu')

# mask_tracks:0,1,2
# mask_bars:[4,5,6,7]
# todo implement mask mode == 0
def mask_bar_and_track(event,vocab,mode,bar_track_control=False,bar_track_control_at_end=False,
                       mask_tracks=[],mask_bars=[],):
    control_types = set(vocab.token_class_ranges.values())

    if bar_track_control:
        total_track_control_types = 0
        if 'density' in control_types:
            total_track_control_types += 1
        if 'occupation' in control_types:
            total_track_control_types += 1
        if 'polyphony' in control_types:
            total_track_control_types += 1

        if 'tensile' in control_types:
            tension_control = True
        else:
            tension_control = False

    mask_mode = mode
    tokens = []

    decoder_target = []
    masked_indices_pairs = []
    mask_bar_names = []
    mask_track_names = []
    r = re.compile('track_\d')

    track_names = list(set(filter(r.match, event)))
    track_names.sort()

    bar_poses = np.where(np.array(event) == 'bar')[0]

    r = re.compile('i_\d')

    track_program = list(filter(r.match, event))
    track_nums = len(track_program)

    track_poses = []
    for track_name in track_names:
        track_pos = np.where(track_name == np.array(event))[0]
        track_poses.extend(track_pos)
    track_poses.extend(bar_poses)

    all_track_pos = list(np.sort(track_poses))
    all_track_pos.append(len(event))

    bar_with_track_poses = []

    for i, pos in enumerate(all_track_pos[1:]):
        if i % (track_nums + 1) == 0:
            this_bar_poses = []
            this_bar_pairs = []
            this_bar_poses.append(pos)

        else:

            this_bar_poses.append(pos)
            if i % (track_nums + 1) == track_nums:
                for j in range(len(this_bar_poses) - 1):
                    this_bar_pairs.append((this_bar_poses[j] + 1, this_bar_poses[j + 1]))

                bar_with_track_poses.append(this_bar_pairs)
    ###
    if mask_mode == 1:
        for bar_num, tracks_in_a_bar in enumerate(bar_with_track_poses):

            for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
                if track_pos in mask_tracks:
                    track_start, track_end = track_star_end_poses

                    mask_bar_names.append(bar_num)
                    mask_track_names.append(track_pos)

                    if bar_track_control:
                        token_start = track_start + total_track_control_types
                        if bar_track_control_at_end:
                            if tension_control and event[track_end - 1] in vocab.name_to_tokens['tensile']:
                                tensile_end = 1
                            else:
                                tensile_end = 0

                            token_end = track_end - total_track_control_types - tensile_end
                        else:

                            token_end = track_end
                    else:
                        token_start = track_start
                        token_end = track_end

                    masked_indices_pairs.append((token_start, token_end))
                    if bar_track_control_at_end:
                        for i in range(total_track_control_types + tensile_end):
                            masked_indices_pairs.append((token_end + i, token_end + 1 + i))


    if mask_mode == 2:
        # random bars
      
        if len(bar_poses) > mask_bars[-1]:

            bar_mask_poses = mask_bars
        else:
            return None


        ### to be tested

        for bar_mask_pos in bar_mask_poses:
            tracks_in_a_bar = bar_with_track_poses[bar_mask_pos]
            for track_idx, track_star_end_poses in enumerate(tracks_in_a_bar):
                mask_bar_names.append(bar_mask_pos)
                mask_track_names.append(track_idx)

                track_start, track_end = track_star_end_poses

                if bar_track_control:
                    token_start = track_start + total_track_control_types
                    if bar_track_control_at_end:
                        if tension_control and event[track_end - 1] in \
                                vocab.name_to_tokens['tensile']:
                            tensile_end = 1
                        else:
                            tensile_end = 0
                        token_end = track_end - total_track_control_types - tensile_end
                    else:

                        token_end = track_end
                else:
                    token_start = track_start
                    token_end = track_end

                masked_indices_pairs.append((token_start, token_end))
                if bar_track_control_at_end:
                    for i in range(total_track_control_types + tensile_end):
                        masked_indices_pairs.append((token_end + i, token_end + 1 + i))

               
        ###
        
        
    #     for mask_bar in mask_bars:
    #
    #         track_mask_poses = mask_tracks
    #
    #
    #         for track_mask_pos in track_mask_poses:
    #             mask_track_names.append(track_mask_pos)
    #             mask_bar_names.append(mask_bar)
    #             bar_with_track_poses[mask_bar][track_mask_pos]
    #             masked_indices_pairs.append(bar_with_track_poses[mask_bar][track_mask_pos])
    # elif mask_mode == 1:
    #     # mask whole tracks
    #     # if track_nums == 1:
    #     #     return None
    #
    #     # track_mask_number = np.random.randint(0, track_nums-1)
    #     if track_nums > mask_tracks[-1]:
    #         track_mask_poses = mask_tracks
    #     else:
    #         return None
    #     for bar_num, tracks_in_a_bar in enumerate(bar_with_track_poses):
    #         for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
    #             if track_pos in track_mask_poses:
    #                 mask_bar_names.append(bar_num)
    #                 mask_track_names.append(track_pos)
    #                 masked_indices_pairs.append(track_star_end_poses)
    #
    # else:
    #     # mask whole bars
    #
    #     bar_mask_poses = mask_bars
    #
    #     for bar_mask_pos in bar_mask_poses:
    #         for track_name in range(track_nums):
    #             mask_bar_names.append(bar_mask_pos)
    #             mask_track_names.append(track_name)
    #             masked_indices_pairs.append(bar_with_track_poses[bar_mask_pos][track_name])

    assert len(mask_bar_names) == len(mask_track_names)

    token_events = event.copy()

    for masked_pairs in masked_indices_pairs:
        masked_token = event[masked_pairs[0]:masked_pairs[1]]
        decoder_target.append(vocab.mask_indices[0])
        for token in masked_token:
            decoder_target.append(vocab.char2index(token))
        else:
            decoder_target.append(vocab.eos_index)

    for masked_pairs in masked_indices_pairs[::-1]:
        # print(masked_pairs)
        # print(token_events[masked_pairs[0]:masked_pairs[1]])
        for pop_time in range(masked_pairs[1] - masked_pairs[0]):
            token_events.pop(masked_pairs[0])
        token_events.insert(masked_pairs[0], 'm_0')

    for token in token_events:
        tokens.append(vocab.char2index(token))

    tokens = np.array(tokens)
    decoder_target = np.array(decoder_target)

    return tokens, decoder_target, mask_track_names, mask_bar_names





def get_note_duration_dict(beat_duration,curr_time_signature):
    duration_name_to_time = {}
    if curr_time_signature[1] == 4:
        # 4/4, 2/4, 3/4
        quarter_note_duration = beat_duration
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
        # quarter_triplets_duration = half_note_duration / 3
        # eighth_triplets_duration = quarter_note_duration / 3
        # sixteenth_triplets_duration = eighth_note_duration / 3
        if curr_time_signature[0] >= 4:
            whole_note_duration = 4 * quarter_note_duration
        bar_duration = curr_time_signature[0] * quarter_note_duration

    else:
        # 6/8

        quarter_note_duration = int(beat_duration / 3 * 2)
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
        # quarter_triplets_duration = half_note_duration / 3
        # eighth_triplets_duration = quarter_note_duration / 3
        # sixteenth_triplets_duration = eighth_note_duration / 3

        bar_duration = int(curr_time_signature[0] * eighth_note_duration)

    duration_name_to_time['half'] = half_note_duration
    duration_name_to_time['quarter'] = quarter_note_duration
    duration_name_to_time['eighth'] = eighth_note_duration
    duration_name_to_time['sixteenth'] = sixteenth_note_duration

    basic_names = duration_name_to_time.keys()
    name_pairs = itertools.combinations(basic_names, 2)
    name_triple = itertools.combinations(basic_names, 3)
    name_quadruple = itertools.combinations(basic_names, 4)

    for name1,name2 in name_pairs:
        duration_name_to_time[name1+'_'+name2] = duration_name_to_time[name1] + duration_name_to_time[name2]

    for name1, name2,name3 in name_triple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3] = duration_name_to_time[name1] + duration_name_to_time[name2] + duration_name_to_time[name3]

    for name1, name2, name3, name4 in name_quadruple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3 + '_' + name4] = duration_name_to_time[name1] + duration_name_to_time[
            name2] + duration_name_to_time[name3] + duration_name_to_time[name4]


    duration_name_to_time['zero'] = 0

    # duration_name_to_time['quarter_triplets'] = quarter_triplets_duration
    # duration_name_to_time['eighth_triplets'] = eighth_triplets_duration
    # duration_name_to_time['sixteenth_triplets'] = sixteenth_triplets_duration

    if curr_time_signature[0] >= 4 and curr_time_signature[1] == 4:
        duration_name_to_time['whole'] = whole_note_duration

    duration_time_to_name = {v: k for k, v in duration_name_to_time.items()}

    duration_times = np.sort(np.array(list(duration_time_to_name.keys())))
    return duration_name_to_time,duration_time_to_name,duration_times,bar_duration


def total_duration(duration_list,duration_name_to_time):
    total = 0
    if duration_list:

        for duration in duration_list:
            total += duration_name_to_time[duration]
    return total


def time2single_durations(note_duration, duration_time_to_name, duration_times):

    duration_index = np.argmin(np.abs(note_duration - duration_times))

    return f"n_{duration_index}"


def time2durations(note_duration, duration_time_to_name, duration_times):

    duration_index = np.argmin(np.abs(note_duration - duration_times))
    duration_name = duration_time_to_name[duration_times[duration_index]]
    if duration_name == 'zero':
        return []

    duration_elements = duration_name.split('_')
    return duration_elements



def check_track_total_time(events,duration_name_to_time,duration_time_to_name,duration_times, bar_duration):


    current_time = 0
    in_duration = False
    duration_list = []
    previous_time = 0
    in_rest_s = False
    new_events = []

    if len(events) == 2:
        last_total_time_adjusted = time2durations(bar_duration, duration_time_to_name, duration_times)
        for token in last_total_time_adjusted[::-1]:
            events.insert(-1,token)
        events.insert(-1,'rest_e')
        return False, events

    total_time = 0

    for event in events:
        new_events.append(event)

        if in_duration and event not in duration_multi:
            total_time = total_duration(duration_list,duration_name_to_time)
            if in_rest_s:
                current_time = previous_time + total_time
                in_rest_s = False
            else:
                previous_time = current_time
                current_time = current_time + total_time

            in_duration = False
            if current_time >= bar_duration:
                break
            duration_list = []



        if event in duration_multi:
            in_duration = True
            duration_list.append(event)

        if event == 'rest_s':
            in_rest_s = True

    else:
        if duration_list:
            total_time = total_duration(duration_list, duration_name_to_time)
            if in_rest_s:
                current_time = previous_time + total_time

            else:

                current_time = current_time + total_time

    if len(new_events) < 4:
        logger.info(new_events)
    while len(new_events) > 0 and new_events[-1] not in duration_multi:
        new_events.pop()
    if current_time == bar_duration:
        return True,new_events
    else:
        if current_time == 0:
            return False, new_events
        if current_time > bar_duration:
            difference = current_time - bar_duration
            last_total_time_adjusted = total_time - difference

        else:
            difference = bar_duration - current_time
            last_total_time_adjusted = total_time + difference

        last_duration_list = time2durations(last_total_time_adjusted, duration_time_to_name, duration_times)
        for _ in range(len(duration_list)):
            new_events.pop()

        new_events.extend(last_duration_list)

        return False, new_events
    # except:
    #     print(new_events)
        # print(duration_name_to_time)
        # print(duration_time_to_name)
        # print(duration_times)
        # print(bar_duration)



def restore_marked_input(src_token, generated_output):
    src_token = np.array(src_token, dtype='<U9')

    # restore with generated output
    restored_with_generated_token = src_token.copy()

    generated_output = np.array(generated_output)

    generation_mask_indices = np.where(generated_output == 'm_0')[0]

    if len(generation_mask_indices) == 1:

        mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
        generated_result_sec = generated_output[generation_mask_indices[0] + 1:]

        #         logger.info(len(generated_result_sec))
        restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])
        for token in generated_result_sec[::-1]:
            #             logger.info(token)
            restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)


    else:

        for i in range(len(generation_mask_indices) - 1):
            #         logger.info(i)
            mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
            generated_result_sec = generated_output[generation_mask_indices[i] + 1:generation_mask_indices[i + 1]]

            #             logger.info(len(generated_result_sec))
            #             logger.info(mask_indices[i])
            restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])

            for token in generated_result_sec[::-1]:
                #                 logger.info(token)
                restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)

        else:
            #         logger.info(i)
            mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
            generated_result_sec = generated_output[generation_mask_indices[i + 1] + 1:]

            #             logger.info(len(generated_result_sec))
            restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])
            for token in generated_result_sec[::-1]:
                #                 logger.info(token)
                restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)

    return restored_with_generated_token





def generation_all(model, events, device, vocab, mask_mode, vocab_mode,mask_tracks,mask_bars,bar_track_control=False,bar_track_control_at_end=False,control_types=[],use_correct_control=False):
    bar_poses = np.where(np.array(events) == 'bar')[0]
    headers = events[:bar_poses[0]]

    r = re.compile('track_\d')

    track_names = list(set(filter(r.match, batch)))
    track_names.sort()

    track_start_idx = 0
    track_end_idx = len(track_names)

    result = mask_bar_and_track(events, vocab, mask_mode,mask_tracks=mask_tracks,mask_bars=mask_bars,bar_track_control=bar_track_control,bar_track_control_at_end=bar_track_control_at_end)
    if result is None:
        return result
    src, tgt_out, mask_track_names, mask_bar_names = result


    if int(events[0][0]) >= 4 and int(events[0][2]) == 4:
        no_whole_duration = False
    else:
        no_whole_duration = True

    if int(events[0][2]) == 8:
        duration_name_to_time, duration_time_to_name, duration_times, bar_duration = get_note_duration_dict(
            1.5, (int(events[0][0]), int(events[0][2])))
    else:
        duration_name_to_time, duration_time_to_name, duration_times, bar_duration = get_note_duration_dict(
            1, (int(events[0][0]), int(events[0][2])))
    sixteenth_time = duration_name_to_time['sixteenth']
    src_masked_nums = np.sum(src == vocab.char2index('m_0'))
    tgt_inp = []
    total_generated_events = []

    if src_masked_nums == 0:
        return None

    total_corrected_times = 0
    corrected_times = 0
    with torch.no_grad():
        mask_idx = 0


        if bar_track_control_at_end:
            all_controls = []
            for control_name in control_types:
                if control_name == 'd':
                    all_controls.extend(vocab.control_indices['density'])
                if control_name == 'o':
                    all_controls.extend(vocab.control_indices['occupation'])
                if control_name == 'p':
                    all_controls.extend(vocab.control_indices['polyphony'])
                if control_name == 't':
                    all_controls.extend(vocab.control_indices['tensile'])

            this_mask_group_idx = 0
            passed_bars = 0
            if len(control_types) > 0:
                if control_types == ['t']:
                    mask_group_length = len(track_names)
                else:
                    mask_group_length = 1 + len(control_types)
                    if 't' in control_types and len(control_types) > 2:
                        mask_bar_change_idx = []
                        if mask_mode == 2:
                            bar_change_idx = np.where(np.diff(mask_bar_names + [9999]) > 0)[0]

                            temp_mask_bar_idx = 0
                            for one_bar_idx in range(len(mask_bar_names)+1):
                                if one_bar_idx in bar_change_idx:
                                    temp_mask_bar_idx += mask_group_length

                                else:
                                    temp_mask_bar_idx += mask_group_length-1
                                mask_bar_change_idx.append(temp_mask_bar_idx)
                        if mask_mode == 1:
                            temp_mask_bar_idx = 0

                            if mask_tracks[0] < len(track_names) - 1:
                                mask_group_length = 4

                            for _ in range(len(mask_bar_names) + 1):
                                temp_mask_bar_idx += mask_group_length
                                mask_bar_change_idx.append(temp_mask_bar_idx)

        this_bar_tokens = []
        this_track_tokens = []

        while mask_idx < src_masked_nums:


            this_tgt_inp = []
            this_tgt_inp.append(vocab.char2index('m_0'))
            this_generated_events = []
            this_generated_events.append('m_0')
            total_grammar_correct_times = 0

            track_end = False
            bar_end = False
            # smer
            if vocab_mode == 0:
                in_pitch = False
                in_rest = False
                in_sep = False
                in_continue = False

            # remi
            else:
                no_pitch = True
                no_step = False
                no_duration = True


            while this_tgt_inp[-1] != vocab.char2index('<eos>') and len(this_tgt_inp) < 100:


                output, weight = model_generate(model, torch.tensor(src), tgt_inp + this_tgt_inp, device,
                                                return_weights=True)
                if vocab_mode == 0:

                    if in_sep:

                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_eos=True,no_whole_duration=True,no_control=True)
                        while index in vocab.rest_indices or index == vocab.eos_index or index == vocab.duration_only_indices[0]:
                            index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_eos=True,no_whole_duration=True,no_control=True)

                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info("in sep failed")
                                break

                        event = vocab.index2char(index)

                    elif in_continue:
                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_duration=True,no_continue=True,no_eos=True,no_control=True)
                        while index not in vocab.pitch_indices:
                            index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_duration=True, no_continue=True,
                                             no_eos=True,no_control=True)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info('in continue failed')
                                break

                        event = vocab.index2char(index)

                    elif in_pitch:
                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_continue=True,
                                         no_whole_duration=no_whole_duration,no_eos=True,no_control=True)
                        while index not in vocab.duration_only_indices and index not in vocab.pitch_indices:
                            index = sampling(output[-1], vocab, no_rest=True, no_sep=True,no_continue=True,
                                             no_whole_duration=no_whole_duration, no_eos=True,no_control=True)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info('in pitch failed')
                                break
                        event = vocab.index2char(index)



                    elif in_rest:
                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_pitch=True, no_rest=True,no_sep=True, no_continue=True,
                                         no_whole_duration=no_whole_duration,no_eos=True,no_control=True)
                        while index not in vocab.duration_only_indices:
                            index = sampling(output[-1], vocab, no_pitch=True, no_rest=True, no_sep=True, no_continue=True,
                                             no_whole_duration=no_whole_duration,no_eos=True,no_control=True)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info('in rest failed')
                                break
                        event = vocab.index2char(index)


                    elif len(this_tgt_inp) == 1:
                        if bar_track_control_at_end and this_mask_group_idx >= 1:
                            if control_types == ['t']:
                                if this_mask_group_idx == mask_group_length:
                                    index = sampling(output[-1], vocab, is_tensile=True)
                                else:
                                    index = sampling(output[-1], vocab, no_duration=True, no_control=True)
                                    sampling_times = 0
                                    while index in vocab.duration_only_indices:
                                        index = sampling(output[-1], vocab, no_duration=True, no_control=True)

                                        sampling_times += 1
                                        total_grammar_correct_times += 1
                                        if sampling_times > 10:
                                            logger.info('start failed')
                                            break
                            else:


                                this_target_control = control_types[this_mask_group_idx-1]
                                # print(this_target_control)
                                if this_target_control == 'd':
                                    track_end = True
                                    index = sampling(output[-1], vocab, is_density=True)
                                elif this_target_control == 'o':
                                    track_end = False
                                    if use_correct_control and occupation_index != -1:
                                        index = occupation_index
                                    else:
                                        index = sampling(output[-1], vocab, is_occupation=True)

                                elif this_target_control == 'p':
                                    track_end = False
                                    if use_correct_control and polyphony_index != -1:
                                        index = polyphony_index
                                    else:
                                        index = sampling(output[-1], vocab, is_polyphony=True)

                                else:
                                    bar_end = True
                                    index = sampling(output[-1], vocab, is_tensile=True)


                        else:
                            index = sampling(output[-1], vocab, no_duration=True,no_control=True)
                            sampling_times = 0
                            while index in vocab.duration_only_indices:
                                index = sampling(output[-1], vocab,no_duration=True,no_control=True)

                                sampling_times += 1
                                total_grammar_correct_times += 1
                                if sampling_times > 10:
                                    logger.info('start failed')
                                    break

                        event = vocab.index2char(index)

                    else:
                        # free state
                        index = sampling(output[-1], vocab, no_whole_duration=no_whole_duration,no_control=True)

                        event = vocab.index2char(index)

                    if index == vocab.continue_index:
                        in_continue = True
                        in_sep = False


                    if index in vocab.pitch_indices:
                        in_pitch = True
                        in_sep = False
                        in_continue = False


                    if index in vocab.duration_only_indices:
                        in_rest = False
                        in_pitch = False

                    if event == 'sep':
                        in_sep = True

                    if event == 'rest':
                        in_rest = True

                elif vocab_mode == 1:
                    # step or eos
                    if no_pitch and no_duration:
                        index = sampling_step_single(output[-1], vocab, no_pitch=no_pitch,no_step=no_step,no_duration=no_duration)
                        sampling_times = 0
                        # # step
                        while index not in vocab.step_indices and index != vocab.eos_index:
                            index = sampling_step_single(output[-1], vocab,  no_pitch=no_pitch,no_step=no_step,no_duration=no_duration)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info('empty track here')
                                break

                        event = vocab.index2char(index)


                        no_pitch = False
                        no_duration = True
                        no_step = True

                    # pitch
                    elif no_step and no_duration:

                        index = sampling_step_single(output[-1], vocab, no_step=no_step,
                                                     no_duration=no_duration)
                        sampling_times = 0
                        while index not in vocab.pitch_indices:
                            index = sampling_step_single(output[-1], vocab, no_step=no_step, no_duration=no_duration)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info('pitch failed here')
                                break
                        event = vocab.index2char(index)

                        no_duration = False
                        no_step = True
                        # if event != this_generated_events[-1]:
                        #     no_duration = False
                        #     no_step = True
                        # else:
                        #     continue


                    elif no_step:

                        index = sampling_step_single(output[-1], vocab, no_step=no_step)
                        sampling_times = 0
                        while index in vocab.step_indices:
                            index = sampling_step_single(output[-1], vocab,  no_step=no_step)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info('step failed here')
                                break
                        event = vocab.index2char(index)
                        if index in vocab.duration_only_indices:

                            no_pitch = True
                            no_duration = True
                            no_step = False
                    else:
                        pass



                if bar_track_control_at_end:

                    if index in all_controls:
                        if use_correct_control:
                            if bar_end:
                                # print(this_bar_tokens)
                                # calculate bar control



                                result = cal_bar_tension(this_bar_tokens,headers, [original_key_name])
                                if result:
                                    bar_tensile_category, _, _ = result
                                else:
                                    bar_tensile_category = []
                                if len(bar_tensile_category) == 0:
                                    this_tgt_inp.append(index)
                                    this_generated_events.append(event)
                                else:
                                    bar_tensile_category = f's_{bar_tensile_category[0]}'
                                    tensile_index = vocab.char2index(bar_tensile_category)
                                    # print(bar_tensile_category)
                                    this_bar_tokens = []
                                    this_tgt_inp.append(tensile_index)
                                    this_generated_events.append(bar_tensile_category)

                            elif track_end:
                                if len(this_track_tokens) == 0:
                                    print(total_generated_events)
                                # print(this_track_tokens)
                                if track_start_idx == track_end_idx:
                                    track_start_idx = 0
                                this_track_tokens.insert(0, f'track_{track_start_idx}')
                                track_start_idx += 1

                                this_bar_tokens.extend(copy.deepcopy(this_track_tokens))

                                # calculate track control

                                bar_track_control = cal_bar_track_control(this_track_tokens,headers,sixteenth_time)
                                this_track_tokens = []
                                density_token = f'd_{bar_track_control[0][0]}'
                                if bar_track_control[1] == -1:
                                    occupation_index = -1
                                    polyphony_index = -1
                                else:
                                    occupation_token = f'o_{bar_track_control[1][0]}'
                                    polyphony_token = f'y_{bar_track_control[2][0]}'
                                    # print(density_token)
                                    # print(occupation_token)
                                    # print(polyphony_token)

                                    density_index = vocab.char2index(density_token)
                                    occupation_index = vocab.char2index(occupation_token)
                                    polyphony_index = vocab.char2index(polyphony_token)

                                    this_tgt_inp.append(density_index)
                                    this_generated_events.append(density_token)
                            else:
                                this_tgt_inp.append(index)
                                this_generated_events.append(event)

                            # this_tgt_inp.append(index)
                            # this_generated_events.append(event)

                            # this_midi = event_2midi(total_generated_events)

                        else:
                            this_tgt_inp.append(index)
                            this_generated_events.append(event)
                        this_tgt_inp.append(vocab.char2index('<eos>'))
                        this_generated_events.append('<eos>')
                    else:
                        this_track_tokens.append(vocab.index2char(index))

                        this_tgt_inp.append(index)
                        this_generated_events.append(event)

                else:
                    this_tgt_inp.append(index)
                    this_generated_events.append(event)


            if bar_track_control_at_end:
                set_0 = False
                if this_mask_group_idx == 0 or (this_mask_group_idx != mask_group_length and control_types == ['t']):
                    if check_total_time:
                        is_time_correct, this_generated_events = check_track_total_time(this_generated_events,
                                                                                        duration_name_to_time,
                                                                                        duration_time_to_name,
                                                                                        duration_times,
                                                                                        bar_duration)
                    else:
                        is_time_correct = True

                    if is_time_correct:
                        if corrected_times > 5:
                            logger.info(f'iterated times is {corrected_times}')
                        mask_idx += 1
                        tgt_inp.extend(this_tgt_inp[:-1])
                        total_generated_events.extend(this_generated_events[:-1])
                        total_corrected_times += corrected_times
                        time_correct_list.append(corrected_times)
                        failed_times_list.append(0)
                        corrected_times = 0
                        this_mask_group_idx += 1


                    else:
                        corrected_times += 1
                        if corrected_times > 10:
                            failed_times_list.append(1)
                            logger.info(f'corrected times > 10, continue generation')
                            mask_idx += 1
                            this_mask_group_idx += 1
                            tgt_inp.extend(this_tgt_inp[:-1])
                            total_generated_events.extend(this_generated_events[:-1])
                            total_corrected_times += corrected_times
                            corrected_times = 0
                else:

                    this_mask_group_idx += 1
                    if 't' in control_types:
                        if len(control_types) > 2:
                            if passed_bars > 0:
                                if this_mask_group_idx + mask_bar_change_idx[passed_bars - 1] in mask_bar_change_idx:
                                    set_0 = True
                                    passed_bars += 1
                            else:
                                if this_mask_group_idx in mask_bar_change_idx:
                                    set_0 = True
                                    passed_bars += 1
                        else:
                            if this_mask_group_idx == mask_group_length + 1:
                                set_0 = True
                    else:
                        if this_mask_group_idx == mask_group_length:
                            set_0 = True
                    if set_0:
                        this_mask_group_idx = 0





                    mask_idx += 1
                    tgt_inp.extend(this_tgt_inp[:-1])
                    total_generated_events.extend(this_generated_events[:-1])



            else:
                if check_total_time:
                    is_time_correct, this_generated_events = check_track_total_time(this_generated_events,
                                                                                    duration_name_to_time,
                                                                                    duration_time_to_name,
                                                                                    duration_times,
                                                                                    bar_duration)
                else:
                    is_time_correct = True

                if is_time_correct:
                    if corrected_times > 5:
                        logger.info(f'iterated times is {corrected_times}')
                    mask_idx += 1
                    tgt_inp.extend(this_tgt_inp[:-1])
                    total_generated_events.extend(this_generated_events[:-1])
                    total_corrected_times += corrected_times
                    time_correct_list.append(corrected_times)
                    failed_times_list.append(0)
                    corrected_times = 0
                else:
                    corrected_times += 1
                    if corrected_times > 10:
                        failed_times_list.append(1)
                        logger.info(f'corrected times > 10, continue generation')
                        mask_idx += 1
                        tgt_inp.extend(this_tgt_inp[:-1])
                        total_generated_events.extend(this_generated_events[:-1])
                        total_corrected_times += corrected_times
                        corrected_times = 0


    src_token = []
    # if vocab_mode == 0:
        # if total_corrected_times > 0:
        #     logger.info(f'total time corrected times is {total_corrected_times}')
        #     if check_total_time:
        #         if control_number in [2,3]:
        #             correction_dict[control_number][0].append(total_corrected_times)
        #         elif control_number == 5:
        #             correction_dict[control_number][2].append(total_corrected_times)
        #         else:
        #             correction_dict[control_number][mask_tracks[0]].append(total_corrected_times)
    # logger.info(f'total grammar corrected times is {total_grammar_correct_times}')

    for i, token_idx in enumerate(src):
        src_token.append(vocab.index2char(token_idx.item()))

    tgt_output_events = []
    for i, token_idx in enumerate(tgt_out):
        if token_idx in vocab.structure_indices[1:]:
            tgt_output_events.append('m_0')
        if token_idx != vocab.char2index('<eos>'):
            tgt_output_events.append(vocab.index2char(token_idx.item()))

    return restore_marked_input(src_token, total_generated_events),restore_marked_input(src_token, tgt_output_events), mask_track_names, mask_bar_names

def get_args(default='.'):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--platform', default='local', type=str,
                        help="local mode")
    parser.add_argument('-v', '--vocab_mode', default=0, type=int,
                        help="vocab mode")

    parser.add_argument('-c', '--cuda', default=0, type=int,
                        help="cuda")

    parser.add_argument('-k', '--check_total_time', default=False, type=bool,
                        help="check total time validation on or off")



    parser.add_argument('-l', '--control_number', default=0, type=int,
                        help="control number")

    parser.add_argument('-w', '--control_mode', default=0, type=int,
                        help="control mode")

    parser.add_argument('-u', '--use_correct_control', default=False, type=bool,
                        help="correct control")

    return parser.parse_args()


args = get_args()
vocab_mode = args.vocab_mode
platform = args.platform
cuda_number = args.cuda
control_number = args.control_number
check_total_time = args.check_total_time

control_mode = args.control_mode
use_correct_control = args.use_correct_control

if control_number == 0:
    control_list = []
elif control_number == 1:
    control_list = ['key', 'tensile']
elif control_number == 2:
    control_list = ['key', 'density']
elif control_number == 3:
    control_list = ['key', 'polyphony']
elif control_number == 4:
    control_list = ['key', 'occupation']
elif control_number == 5:
    control_list = ['key', 'tensile', 'density', 'polyphony',
                    'occupation']
else:
    pass

if control_mode == 0:
    bar_track_control = False
    bar_track_control_at_end = False
elif control_mode == 1:
    bar_track_control = True
    bar_track_control_at_end = False
else:
    bar_track_control = True
    bar_track_control_at_end = True

logger = logging.getLogger(__name__)



use_unk = True
all_unk = True

if platform == 'local':

    output_folder_prefix = '/home/data/guorui/smer_transformer/smer_evaluation'

    checkpoint_prefix = '/home/data/guorui/smer_transformer/checkpoints'
    if vocab_mode == 0:
        if bar_track_control:
            batch_name = '/home/data/guorui/dataset/lmd/batches/bar_dataset/smer_bar_mock_evaluation_batch'
        else:
            batch_name = '/home/data/guorui/dataset/lmd/batches/track_dataset/smer_mock_test_evaluation_batch'
    else:
        batch_name = '/home/data/guorui/dataset/lmd/batches/track_dataset/remi_mock_test_evaluation_batch'
else:
    checkpoint_prefix = './checkpoints'
    if vocab_mode == 0:
        if bar_track_control:
            batch_name = '../dataset/batches/bar_track_dataset/smer_bar_evaluation_batch'
        else:
            batch_name = '../dataset/batches/track_dataset/smer_test_evaluation_batch'
    else:
        batch_name = '../dataset/batches/track_dataset/remi_test_evaluation_batch'
    if use_unk and all_unk:

        output_folder_prefix = './evaluation/all_unk'
    elif use_unk:

        output_folder_prefix = './evaluation/partial_unk'
    else:

        output_folder_prefix = './evaluation/'


if use_correct_control:
    output_folder_prefix += '/use_correct_control'

logger.handlers = []





#smer
if vocab_mode == 0:
    if bar_track_control_at_end:
        checkpoint_folder = '/new/sme_' + str(control_number)
        checkpoint_name = checkpoint_prefix + checkpoint_folder + '/checkpoint_9'
    elif bar_track_control:
        checkpoint_folder = '/new/sma_' + str(control_number)
        checkpoint_name = checkpoint_prefix + checkpoint_folder + '/checkpoint_9'
    else:
        checkpoint_folder = '/smt_' + str(control_number)
        checkpoint_name = checkpoint_prefix + checkpoint_folder + '/checkpoint_9'

else:
    checkpoint_folder = '/rmt_' + str(control_number)
    checkpoint_name = checkpoint_prefix + checkpoint_folder + '/checkpoint_9'

vocab = vocab_class.WordVocab(vocab_mode,control_list)
model = ScoreTransformer(vocab.vocab_size, 512, 8,
                         4,
                         4, 2048, 2400,
                         0.1, 0.1)

device = torch.device(f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu")

model_dict = torch.load(checkpoint_name, map_location=device)

model_state = model_dict['model_state_dict']
# optimizer_state = model_dict['optimizer_state_dict']

from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in model_state.items():
#     name = k[7:]  # remove `module.`
#     new_state_dict[name] = v

# new_state_dict = model_state

model.to(device)

model.load_state_dict(model_state)

# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)

window_size = int(16 / 2)

#

batches = pickle.load(open(batch_name, 'rb'))


if vocab_mode == 0:
    if bar_track_control_at_end:
        output_folder_prefix += '/smer/bar'
    elif bar_track_control:
        output_folder_prefix += '/smer/bar_no_end'
    else:
        output_folder_prefix += '/smer/track'

    if check_total_time:
        output_folder_prefix = output_folder_prefix + '_check'

    logname = f'{output_folder_prefix}/smer_evaluation_{control_number}.log'

else:
    output_folder_prefix += '/remi'
    logname = f'{output_folder_prefix}/remi_evaluation_{control_number}.log'
os.makedirs(output_folder_prefix, exist_ok=True)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S', filename=logname, filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logger.addHandler(console)

coloredlogs.install(level='INFO', logger=logger, isatty=True)
if vocab_mode == 0:
    logger.info(f'smer, platform is {platform}')
if vocab_mode == 1:
    logger.info(f'remi, platform is {platform}')

logger.info(f'batch name is {batch_name}')
logger.info(f'checkpoint name is {checkpoint_name}')
logger.info(f'log file is {logname}')
logger.info(f'control number is {control_number}')
logger.info(f'output folder is {output_folder_prefix}')
logger.info(f'total batch size is {len(batches)}')
logger.info(f'check total time is {check_total_time}')
logger.info(f'use correct control is {use_correct_control}')


logger.info(f'all unk is {use_unk}')
original_control_number = control_number
if control_number == 5:
    control_numbers = [1,2,3,4]
    output_folder_prefix += '/5'
else:
    control_numbers = [control_number]
try:
    for control_number in control_numbers:
        # tension
        bar_tension_original_calculated_diffs = []
        # predicted means bar track at end, the model predict the tension by itself
        # calculated means using the generated one to calculate the actual control
        # original means the control at the bar start
        bar_tension_predicted_calculated_diffs = []

        time_correct_list = []
        failed_times_list = []
        # track control
        changed_track_diff_dict = [{
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        },
        ]

        changed_track_other_diff_dict = [{
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        },
        ]

        # bar track control, mask bars,  with control at end

        bar_mask_track_predicted_calculated_diff_dict = [{
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        },
        ]

        # with both bar control at end or without
        bar_mask_track_calculated_original_diff_dict = [{
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        },
        ]

        # bar track control, mask tracks, with control at end

        track_mask_target_track_original_calculated_diff_dict = [{
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        },
        ]

        track_mask_target_track_predicted_calculated_diff_dict = [{
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        },
        ]

        track_mask_other_track_predicted_calculated_diff_dict = [{
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        },
        ]

        # this is used for both bar track at end and bar track not at end
        track_mask_other_track_calculated_original_diff_dict = [{
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        }, {
            'density': [],
            'occupation': [],
            'polyphony': []
        },
        ]

        mask_tracks = []
        new_track_control = ''

        mask_bars = []

        failed_times = 0
        feature_failed_times = 0

        generate_times = 0




        output_folder = f'{output_folder_prefix}/{control_number}'
        logger.info(f'control number is {control_number}, folder is {output_folder}')
        os.makedirs(output_folder, exist_ok=True)

        random.seed(99)
        for batch_idx, one_batches in enumerate(batches):

            logger.info(f'batch {batch_idx}')
            total_num = len(one_batches)
            idx_number = random.randint(0,total_num-1)

            logger.info(f'idx {idx_number}')

            batch = copy.copy(one_batches[idx_number])


            remove_idx = []
            for idx, token in enumerate(batch):
                if token not in vocab.control_tokens and \
                        token not in vocab.basic_tokens:
                    remove_idx.append(idx)

            for idx in remove_idx[::-1]:
                batch.pop(idx)

            if not isinstance(batch,list):
                batch = batch.tolist()

            r = re.compile('i_\d')


            track_program = list(filter(r.match, batch))
            track_nums = len(track_program)

            r = re.compile('track_\d')

            track_names = list(set(filter(r.match, batch)))
            track_names.sort()

            if bar_track_control:




                bar_poses = np.where(np.array(batch) == 'bar')[0]

                track_poses = []
                for track_name in track_names:
                    track_pos = np.where(track_name == np.array(batch))[0]
                    track_poses.extend(track_pos)
                track_poses.extend(bar_poses)

                all_track_pos = list(np.sort(track_poses))
                all_track_pos.append(len(batch))

                if bar_track_control:
                    total_control_types = []
                    total_track_control_types = 0
                    control_types = set(vocab.token_class_ranges.values())

                    if 'density' in control_types:

                        total_track_control_types += 1
                    if 'occupation' in control_types:

                        total_track_control_types += 1
                    if 'polyphony' in control_types:

                        total_track_control_types += 1

                    if 'tensile' in control_types:
                        tension_control = True

                    else:
                        tension_control = False


                    # copy the bar track/tension token to end
                    if bar_track_control_at_end:
                        # if last token is control, inserted before, continue
                        if batch[-1] in vocab.control_tokens:
                            continue

                        ## copy the bar_track control from track beginning to track end
                        ## copy the tensile control from the bar beginning to the bar end

                        for back_pos in range(len(all_track_pos) - 1, -1, -1):
                            if all_track_pos[back_pos] in bar_poses:
                                # print(back_pos)

                                bar_pos = all_track_pos[back_pos]
                                # print(bar_pos)
                                if back_pos + track_nums + 1 >= len(all_track_pos):
                                    print(back_pos + track_nums + 1)
                                next_bar_pos = all_track_pos[back_pos + track_nums + 1]

                                # print(next_bar_pos)
                                if tension_control:
                                    bar_control = batch[bar_pos + 1]
                                    batch.insert(next_bar_pos, bar_control)

                                if total_track_control_types > 0:
                                    for track_num in range(track_nums):
                                        track_start = all_track_pos[back_pos + track_num + 1] + (
                                            total_track_control_types) * track_num
                                        insert_pos = all_track_pos[back_pos + track_num + 2] + (
                                            total_track_control_types) * track_num
                                        track_controls = batch[
                                                         track_start + 1:track_start + total_track_control_types + 1]


                                        if all_unk:
                                            unk_length = len(track_controls)
                                            unks = ['unk' for _ in range(unk_length)]
                                            for one_unk in unks:
                                                batch.insert(insert_pos, one_unk)
                                        else:
                                            for track_control in track_controls[::-1]:
                                                batch.insert(insert_pos, track_control)


                    if all_unk:
                        for i in range(bar_poses[0],len(batch)):
                            if batch[i] in vocab_class.track_control_tokens:
                                batch[i] = 'unk'

                r = re.compile('track_\d')
                track_names = list(set(filter(r.match, batch)))
                track_names.sort()

                bar_poses = np.where(np.array(batch) == 'bar')[0]

                r = re.compile('i_\d')

                track_program = list(filter(r.match, batch))
                track_nums = len(track_program)

                track_poses = []
                for track_name in track_names:
                    track_pos = np.where(track_name == np.array(batch))[0]
                    track_poses.extend(track_pos)
                track_poses.extend(bar_poses)

                all_track_pos = list(np.sort(track_poses))
                all_track_pos.append(len(batch))

                bar_with_track_poses = []

                for i, pos in enumerate(all_track_pos[1:]):
                    if i % (track_nums + 1) == 0:
                        this_bar_poses = []
                        this_bar_pairs = []
                        this_bar_poses.append(pos)

                    else:

                        this_bar_poses.append(pos)
                        if i % (track_nums + 1) == track_nums:
                            for j in range(len(this_bar_poses) - 1):
                                this_bar_pairs.append((this_bar_poses[j] + 1, this_bar_poses[j + 1]))

                            bar_with_track_poses.append(this_bar_pairs)

            if control_number in [2,3,4]:
                r = re.compile('track_\d')

                track_match = list(set(filter(r.match, batch)))
                track_match.sort()
                # if 'track_1' in track_match:
                #     break

                track_idx_dict = {}
                for track_idx, track_name in enumerate(track_match):
                    track_idx_dict[track_idx] = int(track_name[-1])

            total_control_types = []

            original_key_token = batch[2]
            original_key_name = vocab_class.token_to_key[original_key_token]

            if control_number == 1:

                if original_control_number == 5:
                    r = re.compile('track_\d')
                    total_control_types = ['d','o','p','t']

                    track_match = list(set(filter(r.match, batch)))
                    track_match.sort()
                    # if 'track_1' in track_match:
                    #     break

                    track_idx_dict = {}
                    for track_idx, track_name in enumerate(track_match):
                        track_idx_dict[track_idx] = int(track_name[-1])
                else:
                    total_control_types = ['t']



                bar_poses = np.where(np.array(batch) == 'bar')[0]

                bar_number_weight = np.logspace(1, 2, num=len(bar_poses))[::-1]

                bar_mask_number = random.choices(range(len(bar_poses)), weights=bar_number_weight)[0] + 1
                # bar_mask_number = np.random.randint(0, len(bar_poses))

                if random.random() > .5:
                    # choose a position then continuously mask the next bars
                    start_bar_number = np.random.randint(0, len(bar_poses) - (bar_mask_number - 1))
                    bar_mask_poses = range(start_bar_number, start_bar_number + bar_mask_number)
                else:
                    # randomly choose position
                    bar_mask_poses = np.sort(np.random.choice(len(bar_poses), size=bar_mask_number, replace=False))

                mask_bars = bar_mask_poses

                mask_mode = 2


                changed_tensions = []

                if bar_track_control_at_end:
                    tension_predicted_calculated_diffs = []


                tension_pos_diff = 1
                changed_control_name = 'tensile'

                for mask_bar_num in mask_bars:
                    original_tension_token = batch[bar_poses[mask_bar_num] + tension_pos_diff]
                    original_level = int(original_tension_token.split('_')[-1])



                    if random.random() > 0.7:
                        new_bar_control = 'unk'
                    else:
                        new_bar_control = np.random.choice(vocab.name_to_tokens[changed_control_name])

                        while abs(int(int(original_tension_token.split('_')[-1])) - int(new_bar_control.split('_')[-1])) > 4:
                            new_bar_control = np.random.choice(vocab.name_to_tokens[changed_control_name])
                        # while new_bar_control != original_tension_token and int(new_bar_control.split('_')[-1]) > 8:
                        #     new_bar_control = np.random.choice(vocab.name_to_tokens[changed_control_name])

                    if use_unk:
                        ### all local to 'unk'
                        tracks_in_a_bar = bar_with_track_poses[mask_bar_num]

                        for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):

                            track_start, track_end = track_star_end_poses

                            for bar_track_control_pos in range(track_start,
                                                               track_start + total_track_control_types + 1):
                                # test all control to 'unk'
                                if batch[bar_track_control_pos] in vocab_class.track_control_tokens:
                                    batch[bar_track_control_pos] = 'unk'

                        ###



                    changed_tensions.append(new_bar_control)
                    # other_original_tensions.append(original_other_token)

                    batch[bar_poses[mask_bar_num] + tension_pos_diff] = new_bar_control

                    # record original bar track control for each track
                    # if bar_track_control:
                    #
                    #     tracks_in_a_bar = bar_with_track_poses[mask_bar_num]
                    #     for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
                    #         track_start,track_end = track_star_end_poses
                    #         this_bar_track_control = batch[track_start:track_start + total_track_control_types]
                    #
                    #         original_bar_track_controls.append(this_bar_track_control)
                    # logger.info(f'change bar {mask_bar_num} {changed_control_name} from {original_tension_token} to {new_bar_control} in key {original_key_name}')

            else:


                track_mask_number = 1
                track_mask_poses = np.sort(np.random.choice(track_nums, size=track_mask_number, replace=False))
                mask_mode = 1
                mask_tracks = track_mask_poses

                if control_number in [2,3,4]:

                    if control_number == 2:
                        selected_control_name = 'density'
                        if original_control_number == 5:
                            total_control_types = ['d', 'o', 'p','t']
                        else:
                            total_control_types = ['d']

                    if control_number == 3:
                        selected_control_name = 'polyphony'
                        if original_control_number == 5:
                            total_control_types = ['d', 'o', 'p','t']
                        else:
                            total_control_types = ['p']


                    if control_number == 4:
                        selected_control_name = 'occupation'
                        if original_control_number == 5:
                            total_control_types = ['d', 'o', 'p','t']
                        else:
                            total_control_types = ['o']


                    track_control_end_pos = np.where(np.array(batch) == track_program[0])[0][0]
                    for track_control_start_pos,token in enumerate(batch):
                        if token[0] == 'd' or token[0] == 'y' or token[0] == 'o':
                            break
                    original_track_control = batch[track_control_start_pos:track_control_end_pos]


                    track_num_pos = np.where(track_program[0] == np.array(batch))[0][0]
                    selected_track = mask_tracks[0]
                    for j, token in enumerate(original_track_control):
                        if vocab.token_class_ranges[vocab.char2index(token)] == selected_control_name and \
                                j % track_nums == selected_track:

                            # change token
                            original_track_token = original_track_control[j]


                            new_track_control = str(np.random.choice(vocab.name_to_tokens[selected_control_name]))

                            batch[track_control_start_pos + j] = new_track_control
                            logger.info(
                                f'change track {track_idx_dict[selected_track]} control from {original_track_token} to {new_track_control}')
                            break

                    original_control_diff = int(original_track_token[-1]) - int(new_track_control[-1])

                    original_track_control[j] = new_track_control


                    # change the target track control in each bar track control to the 'unk' or random around the track control
                    if bar_track_control:

                        for bar_num, tracks_in_a_bar in enumerate(bar_with_track_poses):

                            for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
                                if track_pos in mask_tracks:
                                    track_start, track_end = track_star_end_poses

                                    for bar_track_control_pos in range(track_start,track_start + total_track_control_types + 1):

                                        if use_unk:
                                        ### all local to 'unk'
                                            if batch[bar_track_control_pos] in vocab_class.track_control_tokens:

                                                batch[bar_track_control_pos]= 'unk'
                                        ###
                                        else:

                                            if batch[bar_track_control_pos] in vocab.name_to_tokens[selected_control_name]:
                                                # new_bar_track_control = int(new_track_control[-1]) + random.randint(-1,1)
                                                # if new_bar_track_control < 0:
                                                #     new_bar_track_control = 0
                                                # if new_bar_track_control > 9:
                                                #     new_bar_track_control = 9


                                                # batch[bar_track_control_pos] = batch[bar_track_control_pos][:2]  + str(new_bar_track_control)
                                                batch[bar_track_control_pos] = 'unk'


            result = generation_all(model, batch, device, vocab, mask_mode,vocab_mode,mask_tracks,mask_bars,bar_track_control=bar_track_control,bar_track_control_at_end=bar_track_control_at_end,control_types=total_control_types,use_correct_control=use_correct_control)

            while result is None:
                logger.info('failed')
                failed_times += 1
                result = generation_all(model, batch, device, vocab, mask_mode, vocab_mode, mask_tracks, mask_bars)

                if failed_times > 5:
                    logger.info(f'failed number is {failed_times}, next batch')
                    break

            if result is None:
                failed_times = 0
                # idx_number += 1
                continue


    # def compare_result(result):

            restored_with_generated_token, restored_with_target_token, mask_track_names, mask_bar_names = result
            restored_with_target_token = restored_with_target_token.tolist()

            original_bar_pos = np.where(np.array(restored_with_target_token) == 'bar')[0]
            restored_with_generated_token = restored_with_generated_token.tolist()
            mask_track_names = list(set(mask_track_names))
            generated_bar_pos = np.where(np.array(restored_with_generated_token) == 'bar')[0]


            # if track_mode and len(restored_with_generated_token) < 30:
            #     logger.info('total generated token too short')
            #     continue


            if vocab_mode == 0:

                result = event_2midi(restored_with_generated_token)
                if result:
                    generated_pm, _ = result

                    failed_times = 0
                else:
                    logger.info(f'generated failed, token is {restored_with_generated_token}')

                    failed_times += 1
                    if failed_times > 5:
                        logger.info(f'failed number is {failed_times}, next song')

                        failed_times = 0
                        idx_number = total_num
                        break
                    else:
                        continue

                result = event_2midi(restored_with_target_token)
                if result:
                    original_pm, _ = result
                else:

                    logger.info(f'original failed, token is {restored_with_target_token}')

                    failed_times = 0
                    idx_number = total_num
                    break

            else:

                generated_pm = remi_2midi(restored_with_generated_token)
                original_pm = remi_2midi(restored_with_target_token)


            if len(generated_pm.get_beats()) < 6 or len(original_pm.get_beats()) < 6:
                logger.info('too short')
                # idx_number += 1
                failed_times = 0
                continue



            failed_times = 0


            if control_number == 1:


                result = cal_tension(generated_pm,[original_key_name])

                if result:
                    tensiles, diameters, key = result
                else:
                    continue
                tensions, tensions_other, _ = result

                if len(tensions) <= mask_bars[-1]:
                    feature_failed_times += 1
                    if feature_failed_times > 5:
                        logger.info(f'failed number is {feature_failed_times}')
                        # idx_number += 1
                        feature_failed_times = 0
                    continue
                else:
                    new_mask_bars = []
                    if bar_track_control:

                        if original_control_number == 5:
                            _, calculated_bar_density, calculated_bar_occupation, calculated_bar_polyphony = cal_track_control(
                                restored_with_generated_token,
                                generated_pm)

                        track_pos = []

                        for track_name in track_names:
                            track_pos.append(np.where(track_name == np.array(restored_with_generated_token))[0])

                        track_pos = np.sort(np.concatenate(track_pos)).tolist()
                        track_pos.append(len(restored_with_generated_token))


                    for idx, mask_bar in enumerate(mask_bars):
                        if changed_tensions[idx] != 'unk':
                            logger.info(f'set tensile {changed_tensions[idx].split("_")[-1]} , generated is {tensions[mask_bar]}')

                            tension_diff_original_calculated_diff = abs(int(tensions[mask_bar]) - int(changed_tensions[idx].split('_')[-1]))

                            bar_tension_original_calculated_diffs.append(tension_diff_original_calculated_diff)
                        else:
                            logger.info(
                                f'set tensile {changed_tensions[idx]} , generated is {tensions[mask_bar]}')
                        if bar_track_control:

                            for track_idx in range(len(track_pos) - 1):
                                stop = False
                                # if total_track_control_types == 0:
                                #     bar_num = track_idx // len(track_names)
                                # else:
                                bar_num = track_idx // len(track_names)
                                track_num = track_idx % len(track_names)

                                if bar_num == mask_bar:


                                    if original_control_number == 5:
                                        original_track_controls = restored_with_generated_token[track_pos[track_idx] + 1:track_pos[
                                                                                                                             track_idx] + total_track_control_types + 1]

                                        if len(calculated_bar_density[track_names[track_num]]) < bar_num + 1:
                                            logger.info('bar density not long enough')
                                            continue
                                        if len(calculated_bar_occupation[track_num]) < bar_num + 1:
                                            logger.info('bar occupation not long enough')
                                            continue
                                        if len(calculated_bar_polyphony[track_num]) < bar_num + 1:
                                            logger.info('bar polyphony not long enough')
                                            continue

                                        this_bar_calculated_density = to_category([calculated_bar_density[track_names[track_num]][bar_num]],control_bins)


                                        this_bar_calculated_occupation = to_category([calculated_bar_occupation[track_num][bar_num]],control_bins)

                                        this_bar_calculated_polyphony = to_category([calculated_bar_polyphony[track_num][bar_num]], control_bins)

                                        if not use_unk:
                                            density_original_calculated_diff = abs(
                                                int(original_track_controls[0].split('_')[-1]) - this_bar_calculated_density[0])
                                            bar_mask_track_calculated_original_diff_dict[track_idx_dict[track_num]][
                                                'density'].append(density_original_calculated_diff)

                                            occupation_original_calculated_diff = abs(
                                                int(original_track_controls[1].split('_')[-1]) - this_bar_calculated_occupation[0])
                                            bar_mask_track_calculated_original_diff_dict[track_idx_dict[track_num]][
                                                'occupation'].append(occupation_original_calculated_diff)

                                            polyphony_original_calculated_diff = abs(
                                                int(original_track_controls[2].split('_')[-1]) -
                                                    this_bar_calculated_polyphony[0])
                                            bar_mask_track_calculated_original_diff_dict[track_idx_dict[track_num]][
                                                'polyphony'].append(polyphony_original_calculated_diff)

                                            logger.info(
                                                f'bar track ori/cal diff d:{density_original_calculated_diff}, o:{occupation_original_calculated_diff}, y:{polyphony_original_calculated_diff}')

                                    if bar_track_control_at_end:
                                        if original_control_number == 5:

                                            if track_idx + 1 == len(track_pos) - 1:
                                                predicted_track_controls = restored_with_generated_token[
                                                                           track_pos[
                                                                               track_idx + 1] - total_track_control_types - 1:
                                                                           track_pos[
                                                                               track_idx + 1] -1 ]
                                            elif track_num == len(track_names) - 1:

                                                predicted_track_controls = restored_with_generated_token[
                                                                       track_pos[track_idx + 1] - total_track_control_types - 3 :track_pos[
                                                                           track_idx + 1] - 3]
                                            else:
                                                predicted_track_controls = restored_with_generated_token[
                                                                           track_pos[
                                                                               track_idx + 1] - total_track_control_types:
                                                                           track_pos[
                                                                               track_idx + 1]]

                                            for one_control in predicted_track_controls:
                                                if one_control not in vocab.control_tokens:
                                                    logger.info('track control error')
                                                    stop = True
                                                    break


                                            if stop:
                                                continue

                                            density_predicted_calculated_diff = abs(int(predicted_track_controls[0].split('_')[-1]) - this_bar_calculated_density[0])
                                            bar_mask_track_predicted_calculated_diff_dict[track_idx_dict[track_num]]['density'].append(density_predicted_calculated_diff)
                                            # print(predicted_track_controls[0])
                                            # print(this_bar_calculated_density)
                                            occupation_predicted_calculated_diff = abs(
                                                int(predicted_track_controls[1].split('_')[-1]) -
                                                    this_bar_calculated_occupation[0])
                                            bar_mask_track_predicted_calculated_diff_dict[track_idx_dict[track_num]][
                                                'occupation'].append(occupation_predicted_calculated_diff)
                                            # print(predicted_track_controls[1])
                                            # print(this_bar_calculated_occupation)
                                            polyphony_predicted_calculated_diff = abs(
                                                int(predicted_track_controls[2].split('_')[-1]) -
                                                    this_bar_calculated_polyphony[0])
                                            bar_mask_track_predicted_calculated_diff_dict[track_idx_dict[track_num]][
                                                'polyphony'].append(polyphony_predicted_calculated_diff)
                                            # print(predicted_track_controls[2])
                                            # print(this_bar_calculated_polyphony)

                                            logger.info(f'bar track pre/cal diff d:{density_predicted_calculated_diff}, o:{occupation_predicted_calculated_diff}, y:{polyphony_predicted_calculated_diff}')
                                        if changed_tensions[idx] == 'unk':
                                            if track_num == len(track_names) -1:

                                                if track_pos[track_idx + 1] == len(restored_with_generated_token):

                                                    predicted_bar_tension = restored_with_generated_token[
                                                                            track_pos[track_idx + 1]-1]

                                                else:

                                                    predicted_bar_tension = restored_with_generated_token[
                                                                                track_pos[track_idx + 1] - 3:track_pos[track_idx + 1] - 2][0]

                                                if predicted_bar_tension[:2] != 's_':
                                                    logger.info('tension error')
                                                else:

                                                    logger.info(f'predicted bar tension is {predicted_bar_tension}')

                                                    calculated_predicted_tension_diff = abs(int(tensions[mask_bar]) - int(predicted_bar_tension.split('_')[-1]))
                                                    bar_tension_predicted_calculated_diffs.append(calculated_predicted_tension_diff)

                    # idx_number += 1
                    feature_failed_times = 0

            if control_number in [2,3,4]:
                # idx_number += 1

                feature_failed_times = 0
                r = re.compile('i_\d')

                generated_track_control,calculated_bar_density,calculated_bar_occupation,calculated_bar_polyphony = cal_track_control(restored_with_generated_token,
                                                            generated_pm)

                original_track_control,_,_,_ = cal_track_control(restored_with_target_token,
                                                           original_pm)

                track_program = list(filter(r.match, batch))
                track_nums = len(track_program)



                for i in range(0, len(generated_track_control), track_nums):
                    for track_num in range(track_nums):
                        if track_num in mask_track_names:

                            if selected_control_name == 'polyphony':
                                compare_name = 'y'
                            if selected_control_name == 'density':
                                compare_name = 'd'
                            if selected_control_name == 'occupation':
                                compare_name = 'o'

                            if original_track_control[i + track_num][0] == compare_name:
                                logger.info(
                                    f' generated control {generated_track_control[i + track_num]}')

                                if compare_name == 'd':


                                    changed_track_diff_dict[track_idx_dict[mask_tracks[0]]]['density'].append(
                                        int(new_track_control[-1]) - int(generated_track_control[i + track_num][-1]))



                                elif compare_name == 'o':

                                    changed_track_diff_dict[track_idx_dict[mask_tracks[0]]]['occupation'].append(
                                        int(new_track_control[-1]) - int(
                                            generated_track_control[i + track_num][-1]))


                                elif compare_name == 'y':

                                    changed_track_diff_dict[track_idx_dict[mask_tracks[0]]]['polyphony'].append(
                                        int(new_track_control[-1]) - int(
                                            generated_track_control[i + track_num][-1]))


                                else:
                                    pass

                            else:
                                logger.info(
                                    f' target track {track_idx_dict[track_num]} other control : {(original_track_control[i + track_num], generated_track_control[i + track_num])}')

                                if original_track_control[i + track_num][0] == 'd':

                                    changed_track_other_diff_dict[track_idx_dict[mask_tracks[0]]]['density'].append(
                                        int(original_track_control[i + track_num][-1]) - int(
                                            generated_track_control[i + track_num][-1]))

                                elif original_track_control[i + track_num][0] == 'o':
                                    changed_track_other_diff_dict[track_idx_dict[mask_tracks[0]]]['occupation'].append(
                                        int(original_track_control[i + track_num][-1]) - int(
                                            generated_track_control[i + track_num][-1]))

                                elif original_track_control[i + track_num][0] == 'y':
                                    changed_track_other_diff_dict[track_idx_dict[mask_tracks[0]]]['polyphony'].append(
                                        int(original_track_control[i + track_num][-1]) - int(
                                            generated_track_control[i + track_num][-1]))

                                else:

                                    pass

                if bar_track_control:

                    track_pos = []

                    for track_name in track_names:
                        track_pos.append(np.where(track_name == np.array(restored_with_generated_token))[0])

                    track_pos = np.sort(np.concatenate(track_pos)).tolist()
                    track_pos.append(len(restored_with_generated_token))

                    for track_idx in range(len(track_pos) - 1):
                        bar_num = track_idx // len(track_names)
                        track_num = track_idx % len(track_names)
                        if track_num == mask_tracks[0]:

                            if len(calculated_bar_density[track_names[track_num]]) < bar_num + 1:
                                logger.info('bar density not long enough')
                                continue
                            if len(calculated_bar_occupation[track_num]) < bar_num + 1:
                                logger.info('bar occupation not long enough')
                                continue
                            if len(calculated_bar_polyphony[track_num]) < bar_num + 1 :
                                logger.info('bar polyphony not long enough')
                                continue


                            this_bar_calculated_density = to_category(
                                [calculated_bar_density[track_names[track_num]][bar_num]], control_bins)

                            this_bar_calculated_occupation = to_category([calculated_bar_occupation[track_num][bar_num]],
                                                                          control_bins)

                            this_bar_calculated_polyphony = to_category([calculated_bar_polyphony[track_num][bar_num]],
                                                                         control_bins)


                            if not use_unk:
                                original_track_controls = restored_with_generated_token[
                                                          track_pos[track_idx] + 1:track_pos[
                                                                                       track_idx] + total_track_control_types + 1]

                                original_density = original_track_controls[0]
                                original_occupation = original_track_controls[1]
                                original_polyphony = original_track_controls[2]

                                if selected_control_name == 'polyphony':
                                    # track_mask_target_track_original_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                    #     'polyphony'].append(
                                    #     this_bar_calculated_polyphony[0] - int(original_polyphony[-1]))
                                    # print(this_bar_calculated_polyphony[0] - int(original_polyphony[-1]))
                                    track_mask_other_track_calculated_original_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'density'].append(
                                        this_bar_calculated_density[0] - int(original_density[-1]))
                                    # print(this_bar_calculated_density[0] - int(original_density[-1]))

                                    track_mask_other_track_calculated_original_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'occupation'].append(
                                        this_bar_calculated_occupation[0] - int(original_occupation[-1]))
                                    # print(this_bar_calculated_occupation[0] - int(original_occupation[-1]))

                                if selected_control_name == 'occupation':
                                    # track_mask_target_track_original_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                    #     'occupation'].append(
                                    #     this_bar_calculated_occupation[0] - int(original_occupation[-1]))
                                    # print(this_bar_calculated_occupation[0] - int(original_occupation[-1]))
                                    track_mask_other_track_calculated_original_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'density'].append(
                                        this_bar_calculated_density[0] - int(original_density[-1]))
                                    # print(this_bar_calculated_density[0] - int(original_density[-1]))

                                    track_mask_other_track_calculated_original_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'polyphony'].append(
                                        this_bar_calculated_polyphony[0] - int(original_polyphony[-1]))
                                    # print(this_bar_calculated_polyphony[0] - int(original_polyphony[-1]))

                                if selected_control_name == 'density':
                                    # track_mask_target_track_original_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                    #     'density'].append(
                                    #     this_bar_calculated_density[0] - int(original_density[-1]))
                                    # print('ori','d:',this_bar_calculated_density[0] - int(original_density[-1]),'y:',this_bar_calculated_polyphony[0] - int(original_polyphony[-1]),'o:',this_bar_calculated_occupation[0] - int(original_occupation[-1]))

                                    track_mask_other_track_calculated_original_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'polyphony'].append(
                                        this_bar_calculated_polyphony[0] - int(original_polyphony[-1]))

                                    track_mask_other_track_calculated_original_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'occupation'].append(
                                        this_bar_calculated_occupation[0] - int(original_occupation[-1]))

                if bar_track_control_at_end:
                    # generated_pm.write('./temp.mid')

                    track_pos = []

                    for track_name in track_names:
                        track_pos.append(np.where(track_name == np.array(restored_with_generated_token))[0])

                    track_pos = np.sort(np.concatenate(track_pos)).tolist()
                    track_pos.append(len(restored_with_generated_token))


                    for track_idx in range(len(track_pos) - 1):
                        bar_num = track_idx // len(track_names)
                        track_num = track_idx % len(track_names)
                        stop = False
                        if track_num == mask_tracks[0]:

                            if len(calculated_bar_density[track_names[track_num]]) < bar_num + 1:
                                logger.info('bar density not long enough')
                                continue
                            if len(calculated_bar_occupation[track_num]) < bar_num + 1:
                                logger.info('bar occupation not long enough')
                                continue
                            if len(calculated_bar_polyphony[track_num]) < bar_num + 1 :
                                logger.info('bar polyphony not long enough')
                                continue



                            this_bar_calculated_density = to_category(
                                [calculated_bar_density[track_names[track_num]][bar_num]], control_bins)

                            this_bar_calculated_occupation = to_category([calculated_bar_occupation[track_num][bar_num]],
                                                                         control_bins)

                            this_bar_calculated_polyphony = to_category([calculated_bar_polyphony[track_num][bar_num]],
                                                                        control_bins)


                            if selected_control_name == 'polyphony':
                                compare_name = 'y'

                            if selected_control_name == 'density':
                                compare_name = 'd'

                            if selected_control_name == 'occupation':
                                compare_name = 'o'

                            if original_control_number != 5:
                                if restored_with_generated_token[track_pos[track_idx + 1] - 1] == 'bar':
                                    bar_offset = 1
                                else:
                                    bar_offset = 0
                                predicted_track_controls = restored_with_generated_token[
                                                           track_pos[track_idx + 1] - bar_offset - total_track_control_types:track_pos[
                                                               track_idx + 1] - bar_offset]
                            else:
                                if track_idx + 1 == len(track_pos) - 1:
                                    predicted_track_controls = restored_with_generated_token[
                                                               track_pos[
                                                                   track_idx + 1] - total_track_control_types - 1:
                                                               track_pos[
                                                                   track_idx + 1] - 1]
                                elif track_num == len(track_names) - 1:

                                    predicted_track_controls = restored_with_generated_token[
                                                               track_pos[track_idx + 1] - total_track_control_types - 3:
                                                               track_pos[
                                                                   track_idx + 1] - 3]
                                else:
                                    predicted_track_controls = restored_with_generated_token[
                                                               track_pos[
                                                                   track_idx + 1] - total_track_control_types:
                                                               track_pos[
                                                                   track_idx + 1]]

                            for one_control in predicted_track_controls:
                                if one_control not in vocab.control_tokens:
                                    # print(one_control)
                                    stop = True
                                    break
                            # logger.info(f'bar {bar_num}')
                            # logger.info(f'ori: {original_track_controls}')
                            # logger.info(f'cal: ["d_{this_bar_calculated_density[0]}"], ["o_{this_bar_calculated_occupation[0]}"], ["y_{this_bar_calculated_polyphony[0]}"]')
                            #
                            # logger.info(f'pre: {predicted_track_controls}\n')


                            if stop:
                                continue


                            # logger.info(f' predicted track control {predicted_track_controls}')

                            if compare_name == 'd':

                                track_mask_target_track_predicted_calculated_diff_dict[track_idx_dict[mask_tracks[0]]]['density'].append(
                                    this_bar_calculated_density[0] - int(predicted_track_controls[0][-1]))


                            elif compare_name == 'o':
                                if total_track_control_types > 1:
                                    offset = 1
                                else:
                                    offset = 0


                                track_mask_target_track_predicted_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                    'occupation'].append(
                                    this_bar_calculated_occupation[0] - int(predicted_track_controls[offset][-1]))


                            elif compare_name == 'y':
                                if total_track_control_types > 1:
                                    offset = 2
                                else:
                                    offset = 0

                                track_mask_target_track_predicted_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                    'polyphony'].append(
                                    this_bar_calculated_polyphony[0] - int(predicted_track_controls[offset][-1]))

                            else:
                                pass

                            if original_control_number == 5:
                                predicted_density_other_control = predicted_track_controls[0]
                                predicted_occupation_other_control = predicted_track_controls[1]
                                predicted_polyphony_other_control = predicted_track_controls[2]

                                if selected_control_name == 'polyphony':
                                    track_mask_other_track_predicted_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'density'].append(
                                        this_bar_calculated_density[0] - int(predicted_density_other_control[-1]))

                                    track_mask_other_track_predicted_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'occupation'].append(
                                        this_bar_calculated_occupation[0] - int(predicted_occupation_other_control[-1]))

                                if selected_control_name == 'occupation':
                                    track_mask_other_track_predicted_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'density'].append(
                                        this_bar_calculated_density[0] - int(predicted_density_other_control[-1]))

                                    track_mask_other_track_predicted_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'polyphony'].append(
                                        this_bar_calculated_polyphony[0] - int(predicted_polyphony_other_control[-1]))

                                if selected_control_name == 'density':
                                    track_mask_other_track_predicted_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'occupation'].append(
                                        this_bar_calculated_occupation[0] - int(predicted_occupation_other_control[-1]))

                                    track_mask_other_track_predicted_calculated_diff_dict[track_idx_dict[mask_tracks[0]]][
                                        'polyphony'].append(
                                        this_bar_calculated_polyphony[0] - int(predicted_polyphony_other_control[-1]))


        if control_number in [2,3,4]:


            pickle.dump(changed_track_other_diff_dict, open(f'{output_folder}/changed_track_other_diff_dict', 'wb'))
            pickle.dump(changed_track_diff_dict, open(f'{output_folder}/changed_track_diff_dict', 'wb'))

            if bar_track_control:
                pickle.dump(track_mask_target_track_original_calculated_diff_dict,
                            open(f'{output_folder}/track_mask_target_track_original_calculated_diff_dict', 'wb'))

                pickle.dump(track_mask_other_track_calculated_original_diff_dict,
                            open(f'{output_folder}/track_mask_other_track_calculated_original_diff_dict', 'wb'))

                # if original_control_number == 5:
                #     pickle.dump(track_mask_other_track_calculated_original_diff_dict,
                #                 open(f'{output_folder}/track_mask_other_track_calculated_original_diff_dict', 'wb'))



            if bar_track_control_at_end:
                pickle.dump(track_mask_target_track_predicted_calculated_diff_dict, open(f'{output_folder}/track_mask_target_track_predicted_calculated_diff_dict', 'wb'))

                if original_control_number == 5:
                    pickle.dump(track_mask_other_track_predicted_calculated_diff_dict,
                                open(f'{output_folder}/track_mask_other_track_predicted_calculated_diff_dict', 'wb'))


        if control_number in [1]:

            pickle.dump(bar_tension_original_calculated_diffs, open(f'{output_folder}/bar_tension_original_calculated_diffs', 'wb'))

            if bar_track_control and original_control_number == 5:
                pickle.dump(bar_mask_track_calculated_original_diff_dict,
                            open(f'{output_folder}/bar_mask_track_calculated_original_diff_dict', 'wb'))


            if bar_track_control_at_end:
                pickle.dump(bar_tension_predicted_calculated_diffs, open(f'{output_folder}/bar_tension_predicted_calculated_diffs', 'wb'))

                if original_control_number == 5:
                    pickle.dump(bar_mask_track_predicted_calculated_diff_dict,
                                open(f'{output_folder}/bar_mask_track_predicted_calculated_diff_dict', 'wb'))


        if check_total_time:
            pickle.dump(time_correct_list,
                        open(f'{output_folder}/time_correct_list', 'wb'))

            pickle.dump(failed_times_list,
                    open(f'{output_folder}/failed_times_list', 'wb'))
except Exception as e:
    logger.warning(e)


sys.exit()



