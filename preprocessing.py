import pretty_midi
import numpy as np
import os
import math

import argparse
import itertools
import json
import logging
import coloredlogs
import _pickle as pickle
import copy


from vocab import *
from joblib import Parallel, delayed

import tension_calculation

def cal_tension(pm):


    result = tension_calculation.extract_notes(pm, 3)


    pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result

    key_name = tension_calculation.all_key_names

    result = tension_calculation.cal_tension(
        piano_roll, beat_time, beat_indices, down_beat_time,
        down_beat_indices, -1, key_name)
    return result

TRACK_0_RANGE = (21, 108)
# TRACK_1_RANGE = (28, 52)
# TRACK_2_RANGE = (28, 52)

TIME_SIGNATURE_MAX_CHANGE = 1
TEMPO_MAX_CHANGE = 1


MAX_TRACK = 3
V0=120
V1=100
V2=60

def get_args(default='.'):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder', default=default, type=str,
                        help="MIDI file input folder")
    parser.add_argument('-f', '--file_name', default='', type=str,
                        help="input MIDI file name")
    parser.add_argument('-o', '--output_folder',default=default,type=str,
                        help="MIDI file output folder")
    parser.add_argument('-w', '--window_size', default=-1, type=int,
                        help="Tension calculation window size, 1 for a beat, 2 for 2 beat etc., -1 for a downbeat")


    parser.add_argument('-t', '--track_num', default=0, type=int,
                        help="number of tracks used to calculate tension, e.g. 3 means use first 3 tracks, "
                             "default 0 means use all")


    return parser.parse_args()

def walk(folder_name):
    files = []
    for p, d, f in os.walk(folder_name):
        for file_name in f:

            if file_name[-3:] == 'mid':
                files.append(os.path.join(p, file_name))
    return files


def remove_drum_track(pm,track_names=None):
    instrument_idx = []
    if track_names:
        if track_names[-1] == 'drum':
            pm.instruments.pop()
    else:
        for idx in range(len(pm.instruments)):
            if pm.instruments[idx].is_drum:
                instrument_idx.append(idx)
        for idx in instrument_idx[::-1]:
            del pm.instruments[idx]
    return pm


def remove_empty_track(pm):
    occupation_rate = []

    beats = pm.get_beats()
    if len(beats) < 20:
        return None

    fs = 4 / (beats[1] - beats[0])

    for instrument in pm.instruments:
        piano_roll = instrument.get_piano_roll(fs=fs)
        if piano_roll.shape[1] == 0:
            occupation_rate.append(0)
        else:
            occupation_rate.append(np.count_nonzero(np.any(piano_roll, 0)) / piano_roll.shape[1])


    for index,rate in enumerate(occupation_rate[::-1]):
        if rate < 0.3:
            pm.instruments.pop(len(occupation_rate) - 1 - index)
    return pm




def get_beat_time(pm, beat_division=4):
    beats = pm.get_beats()

    divided_beats = []
    for i in range(len(beats) - 1):
        for j in range(beat_division):
            divided_beats.append((beats[i + 1] - beats[i]) / beat_division * j + beats[i])
    divided_beats.append(beats[-1])
    divided_beats = np.unique(divided_beats, axis=0)

    beat_indices = []
    for beat in beats:
        beat_indices.append(np.argwhere(divided_beats == beat)[0][0])

    down_beats = pm.get_downbeats()
    down_beats = np.unique(down_beats, axis=0)

    if divided_beats[-1] > down_beats[-1]:
        down_beats = np.append(down_beats, down_beats[-1] - down_beats[-2] + down_beats[-1])

    down_beat_indices = []
    for down_beat in down_beats:
        down_beat_indices.append(np.argmin(np.abs(down_beat - divided_beats)))

    return np.array(divided_beats), np.array(beats), np.array(down_beats), beat_indices, down_beat_indices


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


# files = walk('../dataset/lmd/lmd_separated_melody_bass')
#
# pm = pretty_midi.PrettyMIDI(files[100])



def time2durations(note_duration, duration_time_to_name, duration_times):

    duration_index = np.argmin(np.abs(note_duration - duration_times))
    duration_name = duration_time_to_name[duration_times[duration_index]]
    if duration_name == 'zero':
        return []

    duration_elements = duration_name.split('_')
    return duration_elements


def note_to_event_name(note,duration_time_to_name, duration_times):
    duration_event = time2durations(note.end-note.start, duration_time_to_name, duration_times)

    pitch_event = f'p_{note.pitch}'

    return pitch_event, duration_event


def bar_notes_to_event(notes, bar_time, next_bar_time, beat_times, duration_time_to_name, duration_times,minimum_difference, grid_division=4,is_grid=True):
    bar_event_list = []
    continue_note_dict = {}
    chord_list = []
    in_continue = False
    if len(notes) > 0:
        if is_grid:
            grid_notes(beat_times, notes,minimum_difference,grid_division=grid_division)
            notes.sort(key=lambda x:(x.start,x.end,x.pitch))
        rest_to_bar_start = time2durations(notes[0].start - bar_time, duration_time_to_name, duration_times)

    else:
        rest_to_bar_start = time2durations(next_bar_time - bar_time, duration_time_to_name, duration_times)

    if len(rest_to_bar_start) > 0:
        bar_event_list.append('rest')
        bar_event_list.extend(rest_to_bar_start)



    for note_idx, note in enumerate(notes):
        # if note_idx == 15:
        #     print(note_idx)

        if len(chord_list) == 0:
            chord_list.append(note)
        else:
            if note.end > next_bar_time and abs(note.start - chord_list[-1].start) < minimum_difference and abs(next_bar_time - chord_list[-1].end) < minimum_difference:
                chord_list.append(note)
            elif abs(note.start - chord_list[-1].start) < minimum_difference and abs(
                        note.end - chord_list[-1].end) < minimum_difference:
                    chord_list.append(note)

            else:
                temp_pitch_list = []



                # remove duplicate notes
                continue_list = []
                other_list = []
                for chord_list_note in chord_list:
                    if chord_list_note.velocity == -1:
                        continue_list.append(chord_list_note)
                    else:
                        other_list.append(chord_list_note)
                if len(continue_list) > 0:
                    continue_list.sort(key=lambda x: x.pitch)
                if len(other_list) > 0:
                    other_list.sort(key=lambda x: x.pitch)
                chord_list = continue_list + other_list


                remove_pos = []
                for pos in range(len(chord_list)-1):
                    if chord_list[pos].pitch == chord_list[pos+1].pitch:
                        remove_pos.append(pos)
                for pos in remove_pos[::-1]:
                    chord_list.pop(pos)

                # clear previous notes in chord_list
                for chord_list_note in chord_list:
                    if chord_list_note.velocity == -1:
                        if not in_continue:
                            temp_pitch_list.append('continue')
                            in_continue = True
                    else:
                        if in_continue:
                            bar_event_list.extend(temp_pitch_list)
                            bar_event_list.extend(duration_event)
                            bar_event_list.append('sep')
                            in_continue = False
                            temp_pitch_list = []

                    if chord_list_note.end > next_bar_time:
                        continue_note_for_next_bar = pretty_midi.Note(pitch=chord_list_note.pitch,
                                                                      start=next_bar_time,
                                                                     end=chord_list_note.end,
                                                                      velocity=-1)
                        continue_note_dict[chord_list_note.pitch] = continue_note_for_next_bar

                        note_for_this_bar = pretty_midi.Note(pitch=chord_list_note.pitch,
                                                             start=chord_list_note.start,
                                                             end=next_bar_time,
                                                             velocity=chord_list_note.velocity)

                        pitch_event, duration_event = note_to_event_name(note_for_this_bar,duration_time_to_name, duration_times)

                    else:
                        pitch_event, duration_event = note_to_event_name(chord_list_note,duration_time_to_name, duration_times)

                    temp_pitch_list.append(pitch_event)


                bar_event_list.extend(temp_pitch_list)
                bar_event_list.extend(duration_event)
                in_continue = False

                if note.start >= chord_list[-1].end:
                    # rest relative to previous end
                    rest_duration = time2durations(note.start - chord_list[-1].end, duration_time_to_name,
                                                   duration_times)
                    if len(rest_duration) > 0:
                        bar_event_list.append('rest')
                        bar_event_list.extend(rest_duration)

                else:

                    # rest relative to previous start
                    rest_duration = time2durations(note.start - chord_list[-1].start, duration_time_to_name,
                                                   duration_times)
                    bar_event_list.append('sep')
                    bar_event_list.extend(rest_duration)
                chord_list = []
                chord_list.append(note)

    else:
        temp_pitch_list = []

        # remove duplicate notes
        continue_list = []
        other_list = []
        for chord_list_note in chord_list:
            if chord_list_note.velocity == -1:
                continue_list.append(chord_list_note)
            else:
                other_list.append(chord_list_note)
        if len(continue_list) > 0:
            continue_list.sort(key=lambda x: x.pitch)
        if len(other_list) > 0:
            other_list.sort(key=lambda x: x.pitch)
        chord_list = continue_list + other_list

        # remove dupliate notes
        chord_list.sort(key=lambda x: x.pitch)
        remove_pos = []
        for pos in range(len(chord_list) - 1):
            if chord_list[pos].pitch == chord_list[pos + 1].pitch:
                remove_pos.append(pos)
        for pos in remove_pos[::-1]:
            chord_list.pop(pos)

        for chord_list_note in chord_list:
            if chord_list_note.velocity == -1:
                if not in_continue:
                    temp_pitch_list.append('continue')
                    in_continue = True
            else:
                if in_continue:
                    bar_event_list.extend(temp_pitch_list)
                    bar_event_list.extend(duration_event)
                    bar_event_list.append('sep')
                    in_continue = False
                    temp_pitch_list = []
            if chord_list_note.end > next_bar_time:
                continue_note_for_next_bar = pretty_midi.Note(pitch=chord_list_note.pitch,
                                                              start=next_bar_time,
                                                              end=chord_list_note.end,
                                                              velocity=-1)

                continue_note_dict[chord_list_note.pitch] = continue_note_for_next_bar

                note_for_this_bar = pretty_midi.Note(pitch=chord_list_note.pitch,
                                                     start=chord_list_note.start, end=next_bar_time,
                                                     velocity=chord_list_note.velocity)

                pitch_event, duration_event = note_to_event_name(note_for_this_bar,duration_time_to_name, duration_times)

            else:
                pitch_event, duration_event = note_to_event_name(chord_list_note,duration_time_to_name, duration_times)

            temp_pitch_list.append(pitch_event)


        if len(temp_pitch_list) > 0:
            bar_event_list.extend(temp_pitch_list)
            bar_event_list.extend(duration_event)

        if chord_list:
            if chord_list_note.end < next_bar_time:
                rest_to_bar_end = time2durations(next_bar_time - chord_list_note.end, duration_time_to_name, duration_times)
                if len(rest_to_bar_end) > 0:
                    bar_event_list.append('rest')
                    bar_event_list.extend(rest_to_bar_end)



    return bar_event_list, continue_note_dict

#
# def change_bar_event_list(bar_note_pitch_list, bar_note_duration_list, bar_duration):
#     if len(bar_note_pitch_list) == 0:
#         duration_event = time2durations(bar_duration, duration_time_to_name, duration_times, is_rest=True)


def grid_notes(beat_times, notes,minimum_difference, grid_division=4):
    divided_beats = []
    for i in range(len(beat_times) - 1):
        for j in range(grid_division):
            divided_beats.append((beat_times[i + 1] - beat_times[i]) / grid_division * j + beat_times[i])
    divided_beats.append(beat_times[-1])

    for note in notes:
        start_grid = np.argmin(np.abs(note.start - divided_beats))

        # maximum note length is two bars
        if note.velocity == -1:
            if note.end > divided_beats[-1]:
                note.end = divided_beats[-1]

        if note.end < divided_beats[-1]+minimum_difference:
            end_grid = np.argmin(np.abs(note.end - divided_beats))
            if start_grid == end_grid:


                if end_grid != len(divided_beats)-1:
                    end_grid += 1
                else:
                    if start_grid != 0:
                        start_grid -= 1
                    else:
                        note.start = -1
                        note.end = -1
                        continue

            note.start = divided_beats[start_grid]
            note.end = divided_beats[end_grid]

        else:
            note.start = divided_beats[start_grid]

    return


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

        quarter_note_duration = beat_duration / 3 * 2
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
        # quarter_triplets_duration = half_note_duration / 3
        # eighth_triplets_duration = quarter_note_duration / 3
        # sixteenth_triplets_duration = eighth_note_duration / 3

        bar_duration = curr_time_signature[0] * eighth_note_duration

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

def midi_2event(file_name,track_info=None):

    pm = pretty_midi.PrettyMIDI(file_name)
    if track_info:
        track_names = list(track_info[file_name].keys())
    else:
        track_names = None
    pm = remove_drum_track(pm,track_names)
    if track_names:
        if 'drum' in track_names:
            track_names.pop()
        if 'chord' in track_names and 'accompaniment' in track_names:
            if np.random.rand() > .5:
                chord_pos = np.where('chord' == np.array(track_names))[0][0]
                del pm.instruments[chord_pos]
                del track_names[chord_pos]
            else:
                accompaniment_pos = np.where('accompaniment' == np.array(track_names))[0][0]
                del pm.instruments[accompaniment_pos]
                del track_names[accompaniment_pos]

        remove_keys = []
        for key in track_info[file_name].keys():
            if key not in track_names:
                remove_keys.append(key)
        for key in remove_keys:
            track_info[file_name].pop(key)


    if len(pm.instruments) == 0:
        print('empty track')
        return None

    tempo_change_times, tempi = pm.get_tempo_changes()
    if tempo_change_times[0] != 0:
        print(f'tempo change time not at start, omit {file_name}')
        return None

    if len(tempo_change_times) > TEMPO_MAX_CHANGE:
        print(f'more than {TEMPO_MAX_CHANGE} tempo changes, omit {file_name}')
        return None


    signature_change_time = np.array([signature.time for signature in pm.time_signature_changes])

    if len(signature_change_time) == 0 or signature_change_time[0] != 0:
        print(f'signature change time not at start, omit {file_name}')
        return None

    if len(pm.time_signature_changes) > TIME_SIGNATURE_MAX_CHANGE:
        print(f'more than {TIME_SIGNATURE_MAX_CHANGE} time signature changes, omit {file_name}')
        return None

    signatures = []
    for signature in pm.time_signature_changes:
        if signature.numerator == 1 and signature.denominator == 4:
            signature.numerator = 4
            signatures.append((signature.numerator, signature.denominator))
        else:
            signatures.append((signature.numerator, signature.denominator))

    beats = np.unique(pm.get_beats(), axis=0)

    down_beats = np.unique(pm.get_downbeats(), axis=0)
    if len(down_beats) < 2:
        return None
    if beats[-1] > down_beats[-1]:
        down_beats = np.append(down_beats, down_beats[-1] + down_beats[-1] - down_beats[-2])
    if not math.isclose(down_beats[-1] - beats[-1], 0):
        beats = np.append(beats, (beats[-1] + beats[-1] - beats[-2]))
    down_beat_to_beat_indices = []
    for down_beat in down_beats:
        down_beat_to_beat_indices.append(np.argmin(np.abs(beats - down_beat)))

    # remove_index = []
    # for i in range(len(signatures[:-1])):
    #     if signatures[i+1] == signatures[i]:
    #         remove_index.append(i)
    # for i in remove_index[::-1]:
    #     signatures.remove(i)
    #     pm.time_signature_changes.remove(i)

    ## todo make sure the time signature and tempo change at the start of the bar

    for signature in signatures:
        if signature not in [(4,4),(2,4),(3,4),(6,8)]:
            print(f'not supported signature {signature}, omit {file_name}')
            return None


    if signatures[0] == (6,8):
        grid_division = 6
    else:
        grid_division = 4

    # tempo_change_bars = np.argmin(np.abs(np.repeat(tempo_change_times,down_beats.shape[0]).reshape(-1,down_beats.shape[0]) - down_beats),axis=1)

    # logger.info(f'working on {file_name}')
    event_list = []
    # duration_name_to_time = {}

    # track_num = len(pm.instruments)

    # track_0_index = track_1_index = track_2_index = 0






    track_num = len(pm.instruments)
    track_num = track_num if track_num < MAX_TRACK else MAX_TRACK
    for num in range(track_num):
        pm.instruments[num].notes.sort(key=lambda note: note.start)

    # previous_bar = -1
    # bar_note_pitch_list = []
    # bar_note_duration_list = []

    continue_dict_list = []

    for _ in range(track_num):
        continue_dict_list.append({})

    curr_time_signature = signatures[0]
    event_list.append(f'{curr_time_signature[0]}/{curr_time_signature[1]}')

    event_list.append(f'{tempi[0]}')

    for instrument in pm.instruments[:track_num]:
        event_list.append(f'i_{instrument.program}')

    for bar, bar_time in enumerate(down_beats[:-1]):
        event_list.append('bar')

        # if bar == 75:
        #     print(bar)


        beat_position = down_beat_to_beat_indices[bar]
        beat_duration = beats[beat_position + 1] - beats[beat_position]

        duration_name_to_time, duration_time_to_name,duration_times, bar_duration = get_note_duration_dict(beat_duration, curr_time_signature)
        minimum_difference = duration_name_to_time['sixteenth'] / 2

        next_bar_time = down_beats[bar + 1]

        # if len(event_list) > 1900:
        #     print(bar)
        for track in range(track_num):
            track_name = track_names[track]
            if track_name == 'melody':
                event_list.append('track_0')
            if track_name == 'bass':
                event_list.append('track_1')
            if track_name == 'accompaniment' or track_name == 'chord':
                event_list.append('track_2')


            # pitch, duration for the next bar if continue
            continue_note_dict = continue_dict_list[track]
            # if len(continue_note_dict.keys()) > 0:

                # continue_event_list, _ = bar_notes_to_event(list(continue_note_dict.values()), bar_time,
                #                                             next_bar_time, beat_in_this_bar, duration_time_to_name,
                #                                             duration_times, minimum_difference,
                #                                             is_continue=True)
                #
                # event_list.extend(continue_event_list)



            note_in_this_bar = [note for note in pm.instruments[track].notes if
                                note.start >= bar_time - minimum_difference and note.start < next_bar_time-minimum_difference]

            for note in note_in_this_bar:
                if note.pitch > TRACK_0_RANGE[1] or note.pitch < TRACK_0_RANGE[0]:
                    print(f"note pitch {note.pitch} out of range, skip this file")
                    return None

            # continue_flag.extend([0] * len(note_in_this_bar))
            beat_in_this_bar = beats[down_beat_to_beat_indices[bar]:down_beat_to_beat_indices[bar+1]+1]
            if len(continue_note_dict.keys()) > 0:
                note_in_this_bar = list(continue_note_dict.values()) + note_in_this_bar

            # if len(note_in_this_bar) > 0:
            #     logger.info(note_in_this_bar)


            bar_event_list, continue_note_dict = bar_notes_to_event(note_in_this_bar, bar_time, next_bar_time,beat_in_this_bar,
                                                                    duration_time_to_name, duration_times, minimum_difference,grid_division=grid_division)

            event_list.extend(bar_event_list)
            continue_dict_list[track] = continue_note_dict
    return event_list,pm


from collections import Counter
import re

def filter_empty_bars(events):
    bar_num = 0
    filled_bar = 0
    first_track_num = 0

    for pos,event in enumerate(events):
        if event == 'bar':
            bar_num += 1
            bar_pos = pos
        if event == 'track_0':
            if first_track_num == 0:
                first_track_num = pos
        if event[0] == 'p':
            filled_bar = bar_num

            break

    if filled_bar != 1:
        meta_events = events[:first_track_num]

        return meta_events + events[bar_pos+1:]
    else:
        return events



def remove_control_event(file_events, control_token):
    new_file_events = copy.copy(file_events)
    for token in new_file_events[::-1]:
        if token in control_token:
            new_file_events.remove(token)
    return new_file_events

def event_2midi(event_list):
    try:

        event_list = remove_control_event(event_list, control_tokens)

        if event_list[1][0] == 't':
            # print(event_list)
            tempo_category = int(event_list[1][2])
            if tempo_category == len(tempo_bins) -1:
                tempo = tempo_bins[tempo_category]
            else:
                tempo = (tempo_bins[tempo_category] + tempo_bins[tempo_category+1]) / 2
        else:
            tempo = float(event_list[1])
        pm_new = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        numerator = int(event_list[0].split('/')[0])
        denominator = int(event_list[0].split('/')[1])
        time_signature = pretty_midi.TimeSignature(numerator, denominator, 0)
        pm_new.time_signature_changes = [time_signature]



        r = re.compile('i_\d')

        programs = list(filter(r.match, event_list))

        r = re.compile('track_\d')

        track_match = list(set(filter(r.match, event_list)))
        track_match.sort()

        track_idx_dict = {}
        for track_idx,track_name in enumerate(track_match):
            track_idx_dict[track_name[-1]] = track_idx



        # program_start_pos = np.where(track_0_program == np.array(event_list))[0][0]
        #
        bar_start_pos = np.where('bar' == np.array(event_list))[0][0]
        # programs = event_list[program_start_pos:start_track_pos]



        for idx,track in enumerate(programs):
            track = pretty_midi.Instrument(program=int(track.split('_')[-1]))
            # if track_match[idx] == 'track_3':
            #     track.is_drum = True
            pm_new.instruments.append(track)



        # add a fake note for duration dict calculation
        pm_new.instruments[0].notes.append(pretty_midi.Note(
            velocity=100, pitch=30, start=0, end=10))
        beats = pm_new.get_beats()
        pm_new.instruments[0].notes.pop()
        duration_name_to_time,duration_time_to_name,duration_times,bar_duration = get_note_duration_dict(beats[1]-beats[0],(time_signature.numerator,time_signature.denominator))

        curr_time = 0
        previous_bar_start_time = 0
        previous_duration = 0

        in_duration_event = False
        is_sep = False
        is_continue = False

        pitch_list = []
        duration_list = []

        bar_num = 0
        track = 0

        bar_poses = np.where(np.array(event_list) == 'bar')[0]


        sta_dict_list = []
        track_bar_length = []
        track_bar_pitch_length = []
        for _ in range(3):
            sta_dict_list.append({'duration_token_length':[],'bar_length':[], 'pitch_token_length':[]})
            track_bar_length.append(0)
            track_bar_pitch_length.append(0)


        def total_duration(duration_list):
            total = 0
            if duration_list:

                for duration in duration_list:
                    total += duration_name_to_time[duration]
            return total

        def clear_pitch_duration_event(pm_new,
                                        track_idx,
                                       curr_time,
                                       previous_duration,
                                       is_sep,
                                       is_continue,
                                       pitch_list,
                                       duration_list):
            if is_sep:
                duration = total_duration(duration_list)
                curr_time -= previous_duration

            else:
                duration = total_duration(duration_list)


            for pitch in pitch_list:
                if is_continue:
                    # look for the previous note, and change the end time of it
                    for note in pm_new.instruments[track_idx].notes[::-1]:
                        if math.isclose(note.end,curr_time) and note.pitch == pitch:
                            note.end += duration
                            break

                else:
                    if track == 0:
                        velocity = V0
                    elif track == 1:
                        velocity = V1
                    else:
                        velocity = V2
                    note = pretty_midi.Note(velocity=velocity,pitch=pitch,start=curr_time,
                        end=curr_time + duration)
                    pm_new.instruments[track_idx].notes.append(note)

            curr_time += duration
            previous_duration = duration

            return curr_time,previous_duration

        current_bar_event = []
        for i, event in enumerate(event_list[bar_start_pos:]):


            current_bar_event.append(event)
            if event in control_tokens:
                continue

            if event in duration_name_to_time.keys():
                duration_list.append(event)
                in_duration_event = True

                track_bar_length[track] += 1

                continue

            if in_duration_event:

                sta_dict_list[track]['duration_token_length'].append(len(duration_list))

                curr_time, previous_duration = clear_pitch_duration_event(pm_new,
                                                                          track_idx,
                                                                          curr_time,
                                                                          previous_duration,
                                                                          is_sep,
                                                                          is_continue,
                                                                          pitch_list,
                                                                          duration_list)


                pitch_list = []
                duration_list = []

                in_duration_event = False
                is_sep = False
                is_continue = False


            pitch_match = re.search(r'p_(\d+)', event)
            if pitch_match:

                track_bar_pitch_length[track] += 1

                pitch = int(pitch_match.group(1))
                pitch_list.append(pitch)

            if event == 'sep':
                is_sep = True

            if event == 'continue' and i > bar_poses[1]:
                is_continue = True


            if event == 'bar':
                bar_start_time = bar_num * bar_duration
                bar_num += 1
                # if bar_num == 65:
                #     print(bar_num)

                if bar_num != 1:

                    for i in range(3):
                        sta_dict_list[i]['bar_length'].append(track_bar_length[i])
                        sta_dict_list[i]['pitch_token_length'].append(track_bar_pitch_length[i])
                        track_bar_length[i] = track_bar_pitch_length[i] = 0

                # if bar_num == 8:
                #     logger.info(bar_num)

                # validate previous bar total time

                # if not math.isclose(bar_start_time,curr_time):
                #     if curr_time != 0:
                #         print(f'in bar {bar_num} the total duration does not equal bar duration')
                #         print(current_bar_event)

                    # exit(1)
                current_bar_event = []
                continue



            track_match = re.search(r'track_(\d)', event)

            if track_match:
                curr_time = bar_start_time
                previous_duration = 0
                track = track_match.group(1)
                track_idx = track_idx_dict[track]
                track = int(track)


            track_bar_length[track] += 1

        else:

            if in_duration_event:

                sta_dict_list[track]['duration_token_length'].append(len(duration_list))

                curr_time, previous_duration = clear_pitch_duration_event(pm_new,
                                                                          track_idx,
                                                                          curr_time,
                                                                          previous_duration,
                                                                          is_sep,
                                                                          is_continue,
                                                                          pitch_list,
                                                                          duration_list)
                pitch_list = []
                duration_list = []

                in_duration_event = False
                is_sep = False
                is_continue = False


            for i in range(3):
                sta_dict_list[i]['bar_length'].append(track_bar_length[i])
                sta_dict_list[i]['pitch_token_length'].append(track_bar_pitch_length[i])
                track_bar_length[i] = track_bar_pitch_length[i] = 0
                # if pm_new.instruments[i].notes[-1].end != curr_time:
                # pm_new.instruments[i].notes.append(pretty_midi.Note(
                #     velocity=0, pitch=10, start=0, end=curr_time))

                # pm_new.instruments[i].notes.append(pretty_midi.Note(
                #     velocity=0, pitch=30, start=0, end=curr_time))

        return pm_new, sta_dict_list
    except Exception as e:
        logger.warning(e)
        return None

#

# track_0_sta = {}
# track_1_sta = {}
# track_2_sta = {}
# total_sta = {}
#
#
# def event_statistics(events):


def cal_separate_file(all_names,i,input_folder,output_folder):
    try:

        file_name = all_names[i]
        print(file_name)
        result = midi_2event(file_name)
        if result is None:
            return None
        event_list, pm = result

        if input_folder[-1] != '/':
            input_folder += '/'
        name_with_sub_folder = file_name.replace(input_folder, "")

        output_name = os.path.join(output_folder, name_with_sub_folder)

        new_output_folder = os.path.dirname(output_name)

        if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)

        pm_new, sta_dict_list = event_2midi(event_list)

        # tensiles, diameters, key, changed_key_name, key_change_beat = cal_tension(pm_new)
        base_name = os.path.basename(file_name).split('.')
        output_name = os.path.join(new_output_folder, base_name[0])
        pm_new.write(output_name + '.mid')
        result = midi_2event(output_name + '.mid')
        event_list, pm = result


        pickle.dump(event_list, open(os.path.join(new_output_folder,
                                                  base_name[0] + '_event'), 'wb'))
        # pickle.dump(diameters, open(os.path.join(new_output_folder,
        #                                           base_name[0] + '_diameter'), 'wb'))

        # pm_new.write(output_name + '.mid')
        return file_name,event_list, sta_dict_list#, key, changed_key_name, key_change_beat
    #
    except Exception as e:
        logger.warning(e)
        return None



# result = midi_2event('/home/ruiguo/infilling_server/default/imagine.mid')[0]
# new_result  = data_convert.rest_multi_step_single(result)
# print(len(result))
# midi2
if __name__== "__main__":
    args = get_args()

    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True)



    logger = logging.getLogger(__name__)

    logger.handlers = []
    logfile = args.output_folder + '/preprocessing.log'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S', filename=logfile,filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)

    coloredlogs.install(level='INFO', logger=logger, isatty=True)


    if len(args.file_name) > 0:
        all_names = [args.file_name]
        args.input_folder = os.path.dirname(args.file_name)
    else:
        all_names = walk(args.input_folder)

    input_folder = args.input_folder

    sta_all_song_tracks = []
    total_event_length = []
    key_dict = {}
    for _ in range(3):
        sta_all_song_tracks.append({'duration_token_length': [], 'bar_length': [], 'pitch_token_length': []})
    with open('/home/data/guorui/dataset/lmd/full_output/program_result.json') as json_file:
        track_info = json.load(json_file)

    for file_idx,file_name in enumerate(all_names):
        logger.info(f'working on {file_idx}th file {file_name}')
        result = midi_2event(file_name,track_info)
        if result is None:
            continue
        event_list, pm = result

        if args.input_folder[-1] != '/':
            args.input_folder += '/'
        name_with_sub_folder = file_name.replace(args.input_folder, "")

        output_name = os.path.join(output_folder, name_with_sub_folder)

        new_output_folder = os.path.dirname(output_name)

        if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)


        pm_new,sta_dict_list = event_2midi(event_list)
        base_name = os.path.basename(file_name).split('.')
        output_name = os.path.join(new_output_folder, base_name[0])


        pm_new.write(output_name + '.mid')

        track_info[output_name + '.mid'] = track_info[file_name]
        track_info.pop(file_name)
        result = midi_2event(output_name + '.mid',track_info)
        if result is None:
            os.remove(output_name + '.mid')
            continue


        event_list, pm = result

        pickle.dump(event_list, open(os.path.join(new_output_folder,
                                                  base_name[0] + '_event'), 'wb'))



        for i in range(len(sta_dict_list)):

            sta_all_song_tracks[i]['duration_token_length'].append(np.mean(sta_dict_list[i]['duration_token_length']))
            sta_all_song_tracks[i]['bar_length'].append(np.mean(sta_dict_list[i]['bar_length']))
            sta_all_song_tracks[i]['pitch_token_length'].append(np.mean(sta_dict_list[i]['pitch_token_length']))


        total_event_length.append(len(event_list))


    logger.info(f'total files are {len(total_event_length)}')
    logger.info(f'average event length  {np.mean(total_event_length)}')
    for i in range(3):
        for key in sta_all_song_tracks[i].keys():
            logger.info(f'for track {i}, the average {key} is {np.mean(sta_all_song_tracks[i][key])}')


        with open(os.path.join(args.output_folder, f'track_result_{i}.json'), 'w') as fp:
            json.dump(sta_all_song_tracks[i], fp)

    # return_list = Parallel(n_jobs=6)(delayed(cal_separate_file)(all_names, i,input_folder,output_folder) for i in range(len(all_names)))
    #
    # for items in return_list:
    #     if items:
    #         #file_name, event_list, sta_dict_list,key,changed_key_name, key_change_beat = items
    #         file_name, event_list, sta_dict_list = items
    #     # event_2midi(event_list)
    #         if sta_dict_list is not None:
    #             for i in range(len(sta_dict_list)):
    #                 sta_all_song_tracks[i]['duration_token_length'].append(np.mean(sta_dict_list[i]['duration_token_length']))
    #                 sta_all_song_tracks[i]['bar_length'].append(np.mean(sta_dict_list[i]['bar_length']))
    #                 sta_all_song_tracks[i]['pitch_token_length'].append(np.mean(sta_dict_list[i]['pitch_token_length']))
    #
    #             total_event_length.append(len(event_list))
    #
    #         #key_dict[file_name[len(input_folder):]] = [key,changed_key_name, key_change_beat]
    # logger.info(f'total files are {len(total_event_length)}')
    # logger.info(f'average event length  {np.mean(total_event_length)}')
    # for i in range(3):
    #     for key in sta_all_song_tracks[i].keys():
    #         logger.info(f'for track {i}, the average {key} is {np.mean(sta_all_song_tracks[i][key])}')
    #
    #     with open(os.path.join(args.output_folder, f'track_result_{i}.json'), 'w') as fp:
    #         json.dump(sta_all_song_tracks[i], fp)

    # with open(os.path.join(args.output_folder, f'keys.json'), 'w') as fp:
    #     json.dump(key_dict, fp)



