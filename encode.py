import pretty_midi
from collections import Counter
import re

import tension_calculation
import music21
import math, itertools
from vocab_control import *
import copy
import numpy as np


def bar_track_density(track_events, track_length):
    total_track_num = 0
    bar_track_note_num = 0

    for track_event in track_events:
        for event_index in range(len(track_event) - 1):
            if track_event[event_index][0] == 'p' and track_event[event_index + 1][0] != 'p':
                total_track_num += 1
                bar_track_note_num += 1
    bar_track_density = bar_track_note_num / track_length

    return bar_track_density


def note_density(track_events, track_length, total_track_length):
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
    return total_track_densities, bar_track_densities


def cal_tension(pm, key_names=None):
    result = tension_calculation.extract_notes(pm, len(pm.instruments))

    if result:

        pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result
    else:
        return None

    if key_names is None:
        key_names = tension_calculation.all_key_names

    result = tension_calculation.cal_tension(
        piano_roll, beat_time, beat_indices, down_beat_time,
        down_beat_indices, -1, key_names, sixteenth_time, pm)

    if result:
        tensiles, diameters, key_name, \
        changed_key_name, key_change_beat = result
    else:
        return None

    tensile_category = to_category(tensiles, tensile_bins)
    diameter_category = to_category(diameters, diameter_bins)

    # print(f'key is {key_name}')

    return tensile_category, diameter_category, key_name


def note_midi(data,start_bar,total_tracks=5):
    """


    :param data: a dict
    :return: midi
    """

    piano_notes = []
    voice_notes = []

    tempo = data['tempo']
    numerator = data['numerator']
    denominator = data['denominator']
    bar_time = 4 * 60 / tempo * numerator / denominator
    shift_time = (start_bar - 1) * bar_time
    beat_time = 60 / tempo


    pm_new = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    time_signature = pretty_midi.TimeSignature(numerator, denominator, 0)
    pm_new.time_signature_changes = [time_signature]

    # for note_idx in range(0,len(melody_notes),6):
    #     this_note = melody_notes[note_idx:note_idx+6]
    #     print(this_note)

    for track_num in range(total_tracks):
        track_name = f'track_{track_num}'

        if track_name in data.keys() and data[track_name + '_program'] > 0:
            if track_num == 4:
                is_drum = True
            else:
                is_drum = False
            program = data[track_name + '_program'] - 1
            track = pretty_midi.Instrument(program=program, is_drum=is_drum)
            pm_new.instruments.append(track)
            track_notes = data[track_name]
            for this_note in track_notes:
                if len(this_note) == 3:
                    track.notes.append(pretty_midi.Note(velocity=100,
                                                        pitch=this_note[0],
                                                        start=this_note[1] * beat_time - shift_time,
                                                        end=(this_note[1] * beat_time + this_note[2] * beat_time - shift_time)))
            track.notes.sort(key=lambda x: (x.start, x.end, x.pitch))

    if len(pm_new.instruments) == 0:
        return None
    return pm_new


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
            bar_polyphony_rate = np.count_nonzero(np.count_nonzero(piano_roll, 0) > 1) / np.count_nonzero(
                np.any(piano_roll, 0))

        return bar_occupation_rate, bar_polyphony_rate
    except:
        return -1, -1


def occupation_polyphony_rate(pm, bar_sixteenth_note_number, sixteenth_notes_time, bar_num):
    occupation_rate = []
    polyphony_rate = []
    bar_occupation_rate = {}
    bar_polyphony_rate = {}


    total_bar_number = bar_num

    for inst_idx, instrument in enumerate(pm.instruments):
        if instrument.is_drum:
            instrument = copy.deepcopy(instrument)
            instrument.is_drum = False
        piano_roll = instrument.get_piano_roll(fs=1 / sixteenth_notes_time)
        if piano_roll.shape[1] == 0:
            occupation_rate.append(0)
        else:
            occupation_rate.append(
                np.count_nonzero(np.any(piano_roll, 0)) / (total_bar_number * bar_sixteenth_note_number))
        if np.count_nonzero(np.any(piano_roll, 0)) == 0:
            polyphony_rate.append(0)
        else:
            polyphony_rate.append(
                np.count_nonzero(np.count_nonzero(piano_roll, 0) > 1) / np.count_nonzero(np.any(piano_roll, 0)))

        bar_occupation_rate[inst_idx] = []
        bar_polyphony_rate[inst_idx] = []

        for bar_idx in range(total_bar_number):
            if piano_roll.shape[1] < bar_idx * bar_sixteenth_note_number:
                bar_occupation_rate[inst_idx].append(0)
                bar_polyphony_rate[inst_idx].append(0)
            else:
                this_bar_track_roll = piano_roll[:,
                                      bar_idx * bar_sixteenth_note_number:bar_idx * bar_sixteenth_note_number + bar_sixteenth_note_number]

                if np.count_nonzero(np.any(this_bar_track_roll, 0)) == 0:
                    bar_polyphony_rate[inst_idx].append(0)

                    bar_occupation_rate[inst_idx].append(0)
                else:
                    bar_occupation_rate[inst_idx].append(
                        np.count_nonzero(np.any(this_bar_track_roll, 0)) / bar_sixteenth_note_number)

                    bar_polyphony_rate[inst_idx].append(
                        np.count_nonzero(np.count_nonzero(this_bar_track_roll, 0) > 1) / np.count_nonzero(
                            np.any(this_bar_track_roll, 0)))

    return occupation_rate, polyphony_rate, bar_occupation_rate, bar_polyphony_rate


def to_category(array, bins):
    result = []
    for item in array:
        result.append(int(np.where((item - bins) >= 0)[0][-1]))
    return result


def get_note_duration_dict(beat_duration, curr_time_signature):
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

    for name1, name2 in name_pairs:
        duration_name_to_time[name1 + '_' + name2] = duration_name_to_time[name1] + duration_name_to_time[name2]

    for name1, name2, name3 in name_triple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3] = duration_name_to_time[name1] + duration_name_to_time[
            name2] + duration_name_to_time[name3]

    for name1, name2, name3, name4 in name_quadruple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3 + '_' + name4] = duration_name_to_time[name1] + \
                                                                                 duration_name_to_time[
                                                                                     name2] + duration_name_to_time[
                                                                                     name3] + duration_name_to_time[
                                                                                     name4]

    duration_name_to_time['zero'] = 0

    # duration_name_to_time['quarter_triplets'] = quarter_triplets_duration
    # duration_name_to_time['eighth_triplets'] = eighth_triplets_duration
    # duration_name_to_time['sixteenth_triplets'] = sixteenth_triplets_duration

    if curr_time_signature[0] >= 4 and curr_time_signature[1] == 4:
        duration_name_to_time['whole'] = whole_note_duration

    duration_time_to_name = {v: k for k, v in duration_name_to_time.items()}

    duration_times = np.sort(np.array(list(duration_time_to_name.keys())))
    return duration_name_to_time, duration_time_to_name, duration_times, bar_duration


def remove_control_event(file_events, control_token):
    new_file_events = copy.copy(file_events)
    for token in new_file_events[::-1]:
        if token in control_token:
            new_file_events.remove(token)
    return new_file_events


def total_duration(duration_list, duration_name_to_time):
    total = 0
    if duration_list:

        for duration in duration_list:
            total += duration_name_to_time[duration]
    return total


def event_2midi(event_list, tempo=None):
    try:

        event_list = remove_control_event(event_list, control_tokens)
        if not tempo:
            if event_list[1][0] == 't':
                # print(event_list)
                tempo_category = int(event_list[1][2])
                if tempo_category == len(tempo_bins) - 1:
                    tempo = tempo_bins[tempo_category]
                else:
                    tempo = (tempo_bins[tempo_category] + tempo_bins[tempo_category + 1]) / 2
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

        track_names = list(set(filter(r.match, event_list)))
        track_names.sort()
        tracks_name_indices = {name:index for index,name in enumerate(track_names)}

        # program_start_pos = np.where(track_0_program == np.array(event_list))[0][0]
        #
        bar_start_pos = np.where('bar' == np.array(event_list))[0][0]
        # programs = event_list[program_start_pos:start_track_pos]

        for index,track in enumerate(programs):
            track = pretty_midi.Instrument(program=int(track.split('_')[-1]))
            if track_names[index] == 'track_4':
                track.is_drum = True
            pm_new.instruments.append(track)

        # add a fake note for duration dict calculation
        for index, instrument in enumerate(pm_new.instruments):
            instrument.notes.append(pretty_midi.Note(
                velocity=100, pitch=1, start=0, end=10))

            if index == 0:
                beats = pm_new.get_beats()
            instrument.notes.pop()

            instrument.notes.append(pretty_midi.Note(
                velocity=100, pitch=1, start=0, end=0.01))

        # pm_new.instruments[0].notes.append(pretty_midi.Note(
        #     velocity=100, pitch=1, start=0, end=10))
        # beats = pm_new.get_beats()
        # pm_new.instruments[0].notes.pop()
        #
        # pm_new.instruments[0].notes.append(pretty_midi.Note(
        #     velocity=100, pitch=1, start=0, end=0.01))
        #
        # pm_new.instruments[1].notes.append(pretty_midi.Note(
        #     velocity=100, pitch=1, start=0, end=0.01))

        duration_name_to_time, duration_time_to_name, duration_times, bar_duration = get_note_duration_dict(
            beats[1] - beats[0], (time_signature.numerator, time_signature.denominator))

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

        bar_nums = len(bar_poses)
        end_time = bar_nums * bar_duration
        lyrics = pretty_midi.Lyric('test', end_time)
        pm_new.lyrics = [lyrics]

        sta_dict_list = []
        track_bar_length = []
        track_bar_pitch_length = []
        for _ in programs:
            sta_dict_list.append({'duration_token_length': [], 'bar_length': [], 'pitch_token_length': []})
            track_bar_length.append(0)
            track_bar_pitch_length.append(0)

        def clear_pitch_duration_event(pm_new,
                                       track,
                                       curr_time,
                                       previous_duration,
                                       is_sep,
                                       is_continue,
                                       pitch_list,
                                       duration_list,
                                       duration_name_to_time):
            if is_sep:
                duration = total_duration(duration_list, duration_name_to_time)
                curr_time -= previous_duration

            else:
                duration = total_duration(duration_list, duration_name_to_time)

            for pitch in pitch_list:
                if is_continue:
                    # look for the previous note, and change the end time of it
                    for note in pm_new.instruments[track].notes[::-1]:
                        if math.isclose(note.end, curr_time) and note.pitch == pitch:
                            note.end += duration
                            break

                else:
                    if track == 0:
                        velocity = V0
                    else:

                        velocity = V1

                    note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=curr_time,
                                            end=curr_time + duration)
                    pm_new.instruments[track].notes.append(note)

            curr_time += duration
            previous_duration = duration

            return curr_time, previous_duration

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
                                                                          track,
                                                                          curr_time,
                                                                          previous_duration,
                                                                          is_sep,
                                                                          is_continue,
                                                                          pitch_list,
                                                                          duration_list,
                                                                          duration_name_to_time)

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

                if bar_num != 1:

                    for i in range(len(programs)):
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
                track = tracks_name_indices[event]

            track_bar_length[track] += 1

        else:

            if in_duration_event:
                sta_dict_list[track]['duration_token_length'].append(len(duration_list))

                curr_time, previous_duration = clear_pitch_duration_event(pm_new,
                                                                          track,
                                                                          curr_time,
                                                                          previous_duration,
                                                                          is_sep,
                                                                          is_continue,
                                                                          pitch_list,
                                                                          duration_list,
                                                                          duration_name_to_time)

        return pm_new
    except:
        print(i, event)
        return None


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

    for index, rate in enumerate(occupation_rate[::-1]):
        if rate < 0.3:
            pm.instruments.pop(len(occupation_rate) - 1 - index)
    return pm


def remove_continue_add_control_event(file_events, header_events, key, tensiles, local_pm):
    num_of_tracks = len(header_events[2:])

    bar_pos = np.where(file_events == 'bar')[0]
    new_file_events = []

    for idx, event in enumerate(file_events):
        if event == 'continue' and idx < bar_pos[1]:
            continue
        else:
            new_file_events.append(event)

    for event in header_events[::-1]:
        new_file_events = np.insert(new_file_events, 0, event)

    # pm = pretty_midi.PrettyMIDI(midi_name)

    pm = local_pm

    all_controls = {}
    all_controls['time_signature'] = new_file_events[0]
    all_controls['tempo'] = new_file_events[1][-1]
    all_controls['key'] = key

    if '_' not in new_file_events[1]:
        tempo = float(new_file_events[1])
        tempo_category = int(np.where((tempo - tempo_bins) >= 0)[0][-1])
        new_file_events[1] = f't_{tempo_category}'

    bar_pos = np.where(new_file_events == 'bar')[0]

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
    total_track_densities, bar_track_densities = note_density(track_events, bar_sixteenth_notes_number,
                                                              total_sixteenth_notes_number)

    # densities = note_density(track_events, track_length,total_track_length)
    total_density_category = to_category(total_track_densities, control_bins)
    for track_name in bar_track_densities.keys():
        bar_track_densities[track_name] = to_category(bar_track_densities[track_name], control_bins)

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

    occupation_rate, polyphony_rate, bar_occupation_rate, bar_polyphony_rate = occupation_polyphony_rate(pm,
                                                                                                         bar_sixteenth_notes_number,
                                                                                                         sixteenth_notes_time,
                                                                                                         len(bar_pos))

    if len(list(bar_track_densities.values())[0]) != len(bar_pos) or len(
            list(bar_occupation_rate.values())[0]) != len(bar_pos) or len(
        list(bar_polyphony_rate.values())[0]) != len(bar_pos):
        # print('invalid')
        return None

    total_occupation_category = to_category(occupation_rate, control_bins)
    total_polyphony_category = to_category(polyphony_rate, control_bins)
    # pitch_register_category = pitch_register(track_events)

    if len(total_density_category) != len(track_names) or len(total_occupation_category) != len(track_names) or len(
            total_polyphony_category) != len(track_names):
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
    all_controls['bar_density'] = {}
    all_controls['bar_occupation'] = {}
    all_controls['bar_polyphony'] = {}

    for track_name in track_names:
        all_controls['bar_density'][track_name] = []
        all_controls['bar_occupation'][track_name] = []
        all_controls['bar_polyphony'][track_name] = []
        all_controls[track_name] = {'instrument': 10, 'density': 10, 'polyphony': 10, 'occupation': 10}

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
                all_controls['bar_density'][track_names[track_idx]].append(0)
            else:
                new_file_events.insert(pos + total_insert, f'd_{bar_track_densities[track_name][i]}')
                all_controls['bar_density'][track_names[track_idx]].append(bar_track_densities[track_name][i])
            total_insert += 1
            if i >= len(this_track_bar_occupation):
                new_file_events.insert(pos + total_insert, 'o_0')
                all_controls['bar_occupation'][track_names[track_idx]].append(0)
            else:
                new_file_events.insert(pos + total_insert, f'o_{this_track_bar_occupation[i]}')
                all_controls['bar_occupation'][track_names[track_idx]].append(this_track_bar_occupation[i])
            total_insert += 1
            if i >= len(this_track_bar_polyphony):
                new_file_events.insert(pos + total_insert, 'y_0')
                all_controls['bar_polyphony'][track_names[track_idx]].append(0)
            else:
                new_file_events.insert(pos + total_insert, f'y_{this_track_bar_polyphony[i]}')
                all_controls['bar_polyphony'][track_names[track_idx]].append(this_track_bar_polyphony[i])
            total_insert += 1

    # bar_track_0_density = bar_track_densities
    # for i, pos in enumerate(bar_track_0_pos):

    all_controls['track_nums'] = num_of_tracks

    # track_names = ['melody','bass']
    #
    # all_controls['melody'] = {'instrument':10,'density':10,'polyphony':10,'occupation':10}
    # all_controls['bass'] = {'instrument':10,'density':10,'polyphony':10,'occupation':10}
    # all_controls['harmony'] = {'instrument': 0, 'density': 0, 'polyphony': 0, 'occupation': 0}
    for track_idx, track_program_num in enumerate(header_events[2:]):
        track_program_name = pretty_midi.program_to_instrument_name(int(track_program_num[2:]))
        all_controls[track_names[track_idx]]['instrument'] = track_program_name
        all_controls[track_names[track_idx]]['density'] = int(density_token[track_idx][-1])
        all_controls[track_names[track_idx]]['polyphony'] = int(polyphony_token[track_idx][-1])
        all_controls[track_names[track_idx]]['occupation'] = int(occupation_token[track_idx][-1])

    all_controls['tensile'] = tensiles

    all_controls['bar_nums'] = len(tensiles)

    return new_file_events, all_controls


def remove_drum_track(pm):
    instrument_idx = []
    for idx in range(len(pm.instruments)):
        if pm.instruments[idx].is_drum:
            instrument_idx.append(idx)
    for idx in instrument_idx[::-1]:
        del pm.instruments[idx]
    return pm


def file_info(midi_name):
    pm = pretty_midi.PrettyMIDI(midi_name)


    track_num = len(pm.instruments)
    down_beats = np.unique(pm.get_downbeats(), axis=0)
    bar_num = len(down_beats)

    tempo_change_times, tempi = pm.get_tempo_changes()

    tempo = tempi[0]

    result = cal_tension(pm)
    result_list = []
    if result:
        tensiles, diameters, first_key = result

        result_list.append(first_key)

    s = music21.converter.parse('no_drum.mid')

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
    result_key = count_result.most_common()[0][0]


    # print(f'result key is {result_key}')

    return {
        'key': result_key,
        'tempo': tempo,
        'track_num': track_num,
        'bar_num': bar_num}


def grid_notes(beat_times, notes, minimum_difference, grid_division=4):
    divided_beats = []
    for i in range(len(beat_times) - 1):
        for j in range(grid_division):
            divided_beats.append((beat_times[i + 1] - beat_times[i]) / grid_division * j + beat_times[i])
    divided_beats.append(beat_times[-1])
    divided_beats = np.array(divided_beats)

    for note in notes:
        start_grid = np.argmin(np.abs(note.start - divided_beats))

        # maximum note length is two bars
        if note.velocity == -1:
            if note.end > divided_beats[-1]:
                note.end = divided_beats[-1]

        if note.end < divided_beats[-1] + minimum_difference:
            end_grid = np.argmin(np.abs(note.end - divided_beats))
            if start_grid == end_grid:

                if end_grid != len(divided_beats) - 1:
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


def note_to_event_name(note, duration_time_to_name, duration_times):
    duration_event = time2durations(note.end - note.start, duration_time_to_name, duration_times)

    pitch_event = f'p_{note.pitch}'

    return pitch_event, duration_event


def time2durations(note_duration, duration_time_to_name, duration_times):
    duration_index = np.argmin(np.abs(note_duration - duration_times))
    duration_name = duration_time_to_name[duration_times[duration_index]]
    if duration_name == 'zero':
        return []

    duration_elements = duration_name.split('_')
    return duration_elements


def bar_notes_to_event(notes, bar_time, next_bar_time, beat_times, duration_time_to_name, duration_times,
                       minimum_difference, grid_division=4, is_grid=True):
    bar_event_list = []
    continue_note_dict = {}
    chord_list = []
    in_continue = False
    if len(notes) > 0:
        if is_grid:
            grid_notes(beat_times, notes, minimum_difference, grid_division=grid_division)
            notes.sort(key=lambda x: (x.start, x.end, x.pitch))
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
            if note.end > next_bar_time and abs(note.start - chord_list[-1].start) < minimum_difference and abs(
                    next_bar_time - chord_list[-1].end) < minimum_difference:
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
                for pos in range(len(chord_list) - 1):
                    if chord_list[pos].pitch == chord_list[pos + 1].pitch:
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

                        pitch_event, duration_event = note_to_event_name(note_for_this_bar, duration_time_to_name,
                                                                         duration_times)

                    else:
                        pitch_event, duration_event = note_to_event_name(chord_list_note, duration_time_to_name,
                                                                         duration_times)

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

                pitch_event, duration_event = note_to_event_name(note_for_this_bar, duration_time_to_name,
                                                                 duration_times)

            else:
                pitch_event, duration_event = note_to_event_name(chord_list_note, duration_time_to_name, duration_times)

            temp_pitch_list.append(pitch_event)

        if len(temp_pitch_list) > 0:
            bar_event_list.extend(temp_pitch_list)
            bar_event_list.extend(duration_event)

        if chord_list:
            if chord_list_note.end < next_bar_time:
                rest_to_bar_end = time2durations(next_bar_time - chord_list_note.end, duration_time_to_name,
                                                 duration_times)
                if len(rest_to_bar_end) > 0:
                    bar_event_list.append('rest')
                    bar_event_list.extend(rest_to_bar_end)

    return bar_event_list, continue_note_dict


def midi_2event(pm,track_names=[]):
    # start_bar = controls['start_bar'] - 1
    # end_bar = controls['start_bar'] + 15
    # print(start_bar)
    # print(end_bar)
    # pm = pretty_midi.PrettyMIDI(file_name)
    # pm = remove_drum_track(pm)

    beats = np.unique(pm.get_beats(), axis=0)
    numerator = pm.time_signature_changes[0].numerator
    denominator = pm.time_signature_changes[0].denominator
    tempo = pm.get_tempo_changes()[1][0]
    down_beats = np.unique(pm.get_downbeats(), axis=0)
    beat_in_bar = int(4 * numerator / denominator)
    if len(down_beats) == 1:
        down_beats_time = 4 * tempo / 60 * denominator / numerator
        down_beats = np.array([0.0,down_beats_time])
    if beats[-1] >= down_beats[-1]:
        down_beats = np.append(down_beats, down_beats[-1] + down_beats[-1] - down_beats[-2])
    while not abs(down_beats[-1] - beats[-1]) < 0.0001:
        beats = np.append(beats, (beats[-1] + beats[-1] - beats[-2]))
    down_beat_to_beat_indices = []
    down_beats = down_beats[:16]
    for down_beat in down_beats:
        down_beat_to_beat_indices.append(np.argmin(np.abs(beats - down_beat)))

    signature_change_time = np.array([signature.time for signature in pm.time_signature_changes])

    if len(signature_change_time) == 0 or signature_change_time[0] != 0:
        print(f'signature change time not at start, omit')
        return None

    if len(pm.time_signature_changes) > TIME_SIGNATURE_MAX_CHANGE:
        print(f'more than {TIME_SIGNATURE_MAX_CHANGE} time signature changes, omit ')
        return None

    signatures = []
    for signature in pm.time_signature_changes:
        # if signature.numerator == 1 and signature.denominator == 4:
        #     signatures.append((4,4))
        # else:
        signatures.append((signature.numerator, signature.denominator))

    ## todo make sure the time signature and tempo change at the start of the bar

    for signature in signatures:
        if signature not in [(4, 4), (2, 4), (3, 4), (6, 8)]:
            print(f'not supported signature {signature}, omit ')
            return None

    tempo_change_times, tempi = pm.get_tempo_changes()

    # if tempo_change_times[0] != 0:
    #     print(f'tempo change time not at start, omit {file_name}')
    #     return None

    # if len(tempo_change_times) > TEMPO_MAX_CHANGE:
    #     print(f'more than {TEMPO_MAX_CHANGE} tempo changes, omit {file_name}')
    #     return None

    if signatures[0] == (6, 8):
        grid_division = 6
    else:
        grid_division = 4

    # tempo_change_bars = np.argmin(np.abs(np.repeat(tempo_change_times,down_beats.shape[0]).reshape(-1,down_beats.shape[0]) - down_beats),axis=1)

    # logger.info(f'working on {file_name}')
    event_list = []

    track_num = len(pm.instruments)

    for num in range(track_num):
        pm.instruments[num].notes.sort(key=lambda note: note.start)

    continue_dict_list = []

    for _ in range(track_num):
        continue_dict_list.append({})

    curr_time_signature = signatures[0]
    event_list.append(f'{curr_time_signature[0]}/{curr_time_signature[1]}')

    event_list.append(f'{tempi[0]}')
    tempo = tempi[0]

    for instrument_index, instrument in enumerate(pm.instruments[:track_num]):
        # if program_names[instrument_index] != 0:
        event_list.append(f'i_{instrument.program}')

    for bar, bar_time in enumerate(down_beats):
        # print('bar', bar)
        # if bar < start_bar:
        #     continue
        # if bar >= end_bar:
        #     break
        event_list.append('bar')

        beat_position = down_beat_to_beat_indices[bar]
        if beat_position + 1 < len(beats):
            beat_duration = beats[beat_position + 1] - beats[beat_position]

        duration_name_to_time, duration_time_to_name, duration_times, bar_duration = get_note_duration_dict(
            beat_duration, curr_time_signature)
        minimum_difference = duration_name_to_time['sixteenth'] / 2

        if bar + 1 < len(down_beats):
            next_bar_time = down_beats[bar + 1]
        else:
            next_bar_time = down_beats[bar] + bar_duration

        # if len(event_list) > 1900:
        #     print(bar)
        for track in range(track_num):
            # if program_names[track] == 0:
            #     continue

            event_list.append(track_names[track])

            # pitch, duration for the next bar if continue
            continue_note_dict = continue_dict_list[track]

            note_in_this_bar = [note for note in pm.instruments[track].notes if
                                note.start >= bar_time - minimum_difference and note.start < next_bar_time - minimum_difference]

            remove_note_list = []
            for note_idx, note in enumerate(note_in_this_bar):
                if note.pitch > TRACK_0_RANGE[1] or note.pitch < TRACK_0_RANGE[0]:
                    remove_note_list.append(note_idx)
            for note_idx in remove_note_list[::-1]:
                del note_in_this_bar[note_idx]
            if len(note_in_this_bar) == 0:
                event_list.append('rest')
                bar_duration_events = time2durations(bar_duration, duration_time_to_name, duration_times)
                event_list.extend(bar_duration_events)
                continue
            # continue_flag.extend([0] * len(note_in_this_bar))
            if bar == 15:
                beat_in_this_bar = beats[down_beat_to_beat_indices[bar]: down_beat_to_beat_indices[bar] + beat_in_bar + 1]
            else:
                beat_in_this_bar = beats[down_beat_to_beat_indices[bar]:down_beat_to_beat_indices[bar + 1] + 1]
            if len(continue_note_dict.keys()) > 0:
                note_in_this_bar = list(continue_note_dict.values()) + note_in_this_bar

            # if len(note_in_this_bar) > 0:
            #     logger.info(note_in_this_bar)

            bar_event_list, continue_note_dict = bar_notes_to_event(note_in_this_bar, bar_time, next_bar_time,
                                                                    beat_in_this_bar,
                                                                    duration_time_to_name, duration_times,
                                                                    minimum_difference, grid_division=grid_division)

            event_list.extend(bar_event_list)
            continue_dict_list[track] = continue_note_dict
    bar += 1
    end_bar = 16
    if bar < end_bar:
        for _ in enumerate(range(end_bar - bar)):
            event_list.append('bar')
            event_list.append('unk')
            for track in range(track_num):
                event_list.append(f'track_{track}')
                event_list.append('rest')
                bar_duration_events = time2durations(bar_duration, duration_time_to_name, duration_times)
                event_list.extend(bar_duration_events)

    if len(np.where(np.array(event_list) == 'bar')[0]) > 16:
        print('what')

    # print(event_list)
    return event_list, pm, tempo


def midi2notes(pm, tempo, track_names,controls):
    total_track_notes = {}
    for name in track_names:
        total_track_notes[name] = []

    start_bar = controls['start_bar']
    s_bar = controls['s_bar'] - start_bar
    e_bar = controls['e_bar'] - start_bar + 1
    numerator = pm.time_signature_changes[0].numerator
    denominator = pm.time_signature_changes[0].denominator
    bar_beat = numerator * 4 / denominator
    shift_beat = bar_beat * (start_bar - 1)

    beat_time = 60 / tempo
    for track_num, track in enumerate(pm.instruments):
        track_name = track_names[track_num]
        if controls[track_name] == 0:
            for note in track.notes:
                start_beat = note.start / beat_time
                if start_beat / bar_beat + 0.01 > s_bar and start_beat / bar_beat < e_bar:
                    if note.pitch == 1 and note.duration < 0.02:
                        continue
                    new_note = {'pitch': note.pitch, 'start_time': note.start / beat_time + shift_beat,
                                'duration': note.duration / beat_time}
                    total_track_notes[track_name].append(new_note)


    return total_track_notes


def merge_pm(total_pm, partial_pm, controls, numerator, denominator, tempo):
    beat_time = 60 / tempo

    start_fill_time = beat_time * numerator * (controls['s_bar'] - 1)
    end_fill_time = beat_time * numerator * (controls['e_bar'])

    partial_shift_time = (controls['start_bar'] - 1) * beat_time * numerator

    for track_num, track in enumerate(total_pm.instruments):
        note_remove_indices = []
        for note_idx, note in enumerate(track.notes):
            if note.pitch == 1:
                note_remove_indices.append(note_idx)
            else:
                if note.start > (start_fill_time - 0.01) and note.start < end_fill_time:
                    note_remove_indices.append(note_idx)
        if note_remove_indices:
            track.notes = track.notes[0:note_remove_indices[0]] + track.notes[note_remove_indices[-1] + 1:]
        partial_notes = partial_pm.instruments[track_num].notes
        for note in partial_notes:
            note.start += partial_shift_time
            note.end += partial_shift_time
            if note.pitch != 1 and note.start >= start_fill_time and note.start < end_fill_time:
                track.notes.append(note)
        track.notes.sort(key=lambda notes: notes.start)

    return total_pm


def encode_midi(pm,controls=None, infill=False,track_names=[]):
    # print(f'file {midi_name}')

    # local midi, at least 16 bars
    events, pm, tempo = midi_2event(pm,track_names=track_names)

    # local_events, total_pm, tempo = midi_2event(pm, controls)

    pm = event_2midi(events,tempo)

    # local_pm.write('local.mid')
    file_events = np.array(events)
    key = controls['key']

    if key and key != "Not Set":

        if not infill:

            key_names = [key]
            result = cal_tension(pm, key_names=key_names)
            if result:
                tensiles, diameters, first_key = result
            else:
                tensiles, diameters, key = '', '', ''

        else:
            tensiles = controls['tensile']
    else:

        key_names = None
        result = cal_tension(pm, key_names=key_names)
        if result:
            tensiles, diameters, first_key = result
            # print(f'first cal key is {first_key}')
            result_list = []
            result_list.append(first_key)

            s = music21.converter.parse('no_drum.mid')

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



            key = count_result.most_common()[0][0]

        else:
            print('error calculating tension')
            return None

    r = re.compile('i_\d')
    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)

    if num_of_tracks < 1:
        print(f'omit file with no track')
        return None

    tempo_category = int(np.where((float(file_events[1]) - tempo_bins) >= 0)[0][-1])
    file_events[1] = f't_{tempo_category}'

    header_events = file_events[:2 + num_of_tracks]

    bar_pos = np.where(file_events == 'bar')[0]

    total_bars = min(len(tensiles), len(bar_pos))
    if total_bars > 16:
        total_bars = 16
        file_events = file_events[:bar_pos[total_bars]]
        bar_pos = bar_pos[:total_bars]

    if total_bars < 16:
        file_events = file_events[:bar_pos[total_bars + 1]]
        bar_pos = bar_pos[:total_bars]

    tension_pos = 0

    events, controls = remove_continue_add_control_event(file_events[bar_pos[0]:],
                                                         header_events, key,
                                                         tensiles[tension_pos:tension_pos + total_bars], pm)

    return events, controls


