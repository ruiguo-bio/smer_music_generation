
import json
import multiprocessing
import os
import subprocess

from tqdm import tqdm
import vocab_2023

import pretty_midi
from collections import defaultdict, OrderedDict
import itertools
import math
#
# import logging
# # Configure the logging settings
# logging.basicConfig(
#     level=logging.WARNING,
#     format='%(asctime)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',  # Format for the timestamp
#     handlers=[
#         logging.StreamHandler(),  # Output to console
#         logging.FileHandler('preprocessing.log')  # Output to log file
#     ]
# )


threshold = 0.001
# find all the file in a path with a specific extension
def find_all_files(path, extension):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                all_files.append(os.path.join(root, file))
    return all_files
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
def quantise_note_to_grid(note, bar_start_time, bar_end_time, original_duration_ratio, all_durations):
    '''
    Quantise note to the grid
    Parameters
    ----------
    note: pretty_midi.Note
    bar_start_time: float in seconds
    bar_length: float in seconds
    original_duration_ratio: the duration ratio of the note to the bar length
    all_durations: all the note durations ratios of different notes to the bar length
    -------
    '''
    bar_length = bar_end_time - bar_start_time
    original_duration_name = convert_duration_ratio_name(original_duration_ratio, all_durations)
    original_note_duration_time = original_duration_ratio * bar_length

    if 'triplet' in convert_duration_ratio_name(original_duration_ratio, all_durations):
        # Use triplet grid
        smallest_duration = all_durations['16th_triplet']
        # print('triplet grid')
    else:
        # Use normal grid
        smallest_duration = all_durations['16th']
        # print('normal grid')

    smallest_duration_time = bar_length * smallest_duration
    all_grid_times = []
    current_time = bar_start_time
    while current_time < bar_start_time + bar_length * 2:
        all_grid_times.append(current_time)
        current_time += smallest_duration_time

    closest_start_grid_time = min(all_grid_times, key=lambda x: abs(x - note.start))
    closest_end_grid_time = min(all_grid_times, key=lambda x: abs(x - note.end))

    while closest_start_grid_time >= closest_end_grid_time:
        if closest_start_grid_time - smallest_duration_time >= bar_start_time:
            closest_start_grid_time -= smallest_duration_time
        else:
            closest_end_grid_time += smallest_duration_time

    start_grid_direction = 1 if note.start > closest_start_grid_time else -1
    end_grid_direction = 1 if note.end > closest_end_grid_time else -1

    new_note_duration_time = closest_end_grid_time - closest_start_grid_time
    new_duration_name = convert_duration_ratio_name(new_note_duration_time / bar_length, all_durations)

    if not math.isclose(original_note_duration_time, new_note_duration_time, rel_tol=0.05):
        if original_note_duration_time > new_note_duration_time:
            if original_note_duration_time < 2:
                if end_grid_direction == 1:
                    closest_end_grid_time += smallest_duration_time
                else:
                    closest_start_grid_time -= smallest_duration_time
        elif original_note_duration_time < new_note_duration_time:
            if end_grid_direction == -1:
                closest_end_grid_time -= smallest_duration_time
            else:
                closest_start_grid_time += smallest_duration_time

        new_note_duration_time = closest_end_grid_time - closest_start_grid_time

    new_ratio = round(new_note_duration_time / bar_length, 4)
    closest_duration = min(
        all_durations.values(),
        key=lambda x: abs(x - new_ratio)
    )
    new_duration_name = convert_duration_ratio_name(closest_duration, all_durations)

    # if original_duration_name != new_duration_name:
        # print(f"Original duration: {original_duration_name}, new duration: {new_duration_name}")
    note.start = closest_start_grid_time
    note.end = closest_end_grid_time
    note.ratio = new_ratio


# convert abc to midi
def abc_to_midi(abc_file, output_path):

    command = f"abc2midi {abc_file} -o {output_path} -silent -quiet"
    devnull = open(os.devnull, 'w')
    result = subprocess.run(command, shell=True, check=False,stdout=devnull, stderr=devnull)
    if result.returncode != 0:
        print(f"Failed to convert {abc_file} to midi")
        return None
    return output_path


# convert midi to abc

def midi_to_abc(midi_file, output_dir):

    output_filename = os.path.splitext(os.path.basename(midi_file))[0] + ".abc"
    output_path = os.path.join(output_dir, output_filename)

    command = f"midi2abc {midi_file} -o {output_path} -nogr -obpl"
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode != 0:
        logging.error(f"Failed to convert {midi_file} to abc")
        return None
    # if successfully convert midi to abc, read the abc file

    with open(output_path, 'r') as f:
        lines = f.read()
        lines = lines.split('\n')

            # remove the comments in abc by finding the first single %
            # only remove comment line and keep all the other tokens
            # don't look for %% field
            # Filter out the comment lines and special-purpose fields
        for line in lines:
            if line.startswith("%") and not line.startswith("%%"):
                lines.remove(line)

        lines = [line for line in lines if not line.startswith("T")]

    return lines






def midi_to_event(midi_file, track_dict=None,threshold=0.001,remove_track_with_few_note=True,bar_limit=None,track_num=None):
    file_durations = defaultdict(lambda: defaultdict(int))
    desired_order = ['melody', 'bass', 'chord', 'accompaniment', 'drum']

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        midi_name = os.path.basename(midi_file)

        # limit the total number of midi tracks
        if track_num:
            midi_data.instruments = midi_data.instruments[:track_num]

        # get program number from the midi_data
        midi_programs = [f'program_{midi_data.instruments[i].program}' for i in range(len(midi_data.instruments))]
        if track_dict is None:
            midi_track_types = desired_order[:len(midi_programs)]
        else:
            midi_track_types = track_dict[midi_name].keys()
            midi_track_types = [t for t in desired_order if t in midi_track_types]


    except (IOError, TypeError, ValueError, IndexError, EOFError, OverflowError, KeyError) as e:
        print(f"Error loading MIDI file: {midi_file}")
        return None

    duration_counts = defaultdict(int)


    for signature in midi_data.time_signature_changes:
        formated_signature = (signature.numerator, signature.denominator)
        if signature.denominator not in [4,8] or signature.numerator >= 10:
            print(f'not supported signature {formated_signature}, omit {midi_name}')
            return None

    bar_beats = midi_data.get_downbeats()
    remove_track_nums = []
    for track_num, instrument in enumerate(midi_data.instruments):
        ignore_track = False
        notes = instrument.notes
        total_ignore_notes = 0

        # sort notes by pitch number and start time
        notes.sort(key=lambda x: (x.pitch, x.start))

        # merge the adjacent notes with same pitch and overlapping time
        for i, note in enumerate(notes):

            if i < len(notes) - 1:
                next_note = notes[i + 1]
                if note.pitch == next_note.pitch and note.end > next_note.start:
                    note.end = next_note.end
                    notes.remove(next_note)

        # sort notes by start time and pitch number
        notes.sort(key=lambda x: (x.start, x.pitch))


        for i, note in enumerate(notes):
            if ignore_track:
                break
            duration = note.duration

            if duration <= 0:
                continue

            # Find the time signature at the start of the note
            time_signature = None
            for ts in reversed(midi_data.time_signature_changes):
                if ts.time < note.start+0.1:
                    time_signature = ts
                    break

            all_durations, all_durations_without_triplet = get_all_durations(time_signature)

            # find the tempo at the start of the note
            tempos = midi_data.get_tempo_changes()
            # tempos[0] is a nparrry of time, and tempos[1] is a nparray of tempo
            # zip the two arrays and sort by time
            tempos = sorted(zip(tempos[0], tempos[1]), key=lambda x: x[0])



            for tempo_time, tempo_value in reversed(tempos):
                if tempo_time < note.start+0.1:
                    tempo = tempo_value
                    break

            # validate the bar length by the downbeat
            for beat_index,beat_time in enumerate(reversed(bar_beats)):
                if beat_time < note.start+0.1:
                    bar_start_time = beat_time
                    bar_end_time = bar_beats[-beat_index]
                    if bar_start_time > bar_end_time:
                        continue
                    bar_length_from_beat = bar_end_time - bar_start_time
                    break

            # Get the beat length based on the time signature and tempo
            bar_length = (60 / tempo) * (4 / time_signature.denominator) * time_signature.numerator
            # beat_length = (60 / tempo) * (4 / time_signature.denominator)
            #                 print(bar_length, beat_length)
            if not math.isclose(round(bar_length,4), round(bar_length_from_beat,4), abs_tol=0.01):
                print(f'bar length not match at {bar_start_time}, with tempo {tempo} and {ts},  {bar_length} vs {bar_length_from_beat}')
                return None
            # Calculate the duration in relation to the beat length
            duration_ratio = duration / bar_length
            smallest_duration_ratio = min(all_durations.values())

            if duration_ratio < smallest_duration_ratio/2:
                # print(f"Duration ratio too small: {duration_ratio}")
                note.ratio = 0
                total_ignore_notes += 1
                if total_ignore_notes > 0.1*len(notes) and remove_track_with_few_note:
                    # print(f"Too many ignored notes: {total_ignore_notes}")
                    ignore_track = True
                    remove_track_nums.append(track_num)
                continue
            closest_duration = min(
                all_durations.values(),
                key=lambda x: abs(x - duration_ratio)
            )


            # Thresholding: Consider duration as triplet only if it's close to the exact triplet duration
            if 'triplet' in convert_duration_ratio_name(closest_duration, all_durations):
                # Adjust this threshold as needed
                if abs(closest_duration - duration_ratio) > threshold:

                    closest_duration = min(
                        all_durations_without_triplet.values(),
                        key=lambda x: abs(x - duration_ratio)
                    )
                else:

                    is_triplet = check_triplet_neighboring_notes(note, notes, bar_length, all_durations)
                    #                         print(f'is triplet {is_triplet}')
                    if not is_triplet:
                        closest_duration = min(
                            all_durations_without_triplet.values(),
                            key=lambda x: abs(x - duration_ratio)
                        )
            # Convert duration names for keys
            duration_name = convert_duration_ratio_name(closest_duration, all_durations)
            note.ratio = closest_duration
            # if 'triplet' in duration_name:
            #     print(
            #         f'pitch:{note.pitch},start:{round(note.start,2)},duration:{round(duration,2)},ratio:{round(duration_ratio,2)},closest:{round(closest_duration,2)}')
            #     print(duration_name)

            duration_counts[duration_name] += 1

            # quantise note to grid
            quantise_note_to_grid(note, bar_start_time, bar_end_time, closest_duration, all_durations)
            if not note.ratio:
                print(note)
        #                 if 'triplet' in duration_name:
        #                     print(f'pitch:{note.pitch},start:{note.start},duration:{duration},name:{duration_name},ratio:{duration_ratio},closest:{closest_duration}')
        # print pitch name(not number) and duration
        # for i, note in enumerate(notes):
        #     print(
        #         f'pitch:{pretty_midi.note_number_to_name(note.pitch)},start:{round(note.start, 2)},duration:{convert_duration_ratio_name(note.velocity, all_durations)}')

        # remove note with velocity 0
        if not ignore_track:
            notes = [note for note in notes if note.ratio != 0]
            # sort notes in the order of start time, velocity, pitch
            notes = sorted(notes, key=lambda x: (x.start, x.ratio, x.pitch))

            instrument.notes = notes

    # remove track with too many ignored notes
    for track_num in sorted(remove_track_nums, reverse=True):
        # print(f'remove {midi_track_types[track_num]} track')
        midi_data.instruments.pop(track_num)
        midi_track_types.pop(track_num)
        midi_programs.pop(track_num)
        # print(f'remove track {track_num}')

    if len(midi_data.instruments) == 0:
        print(f'no instrument left in {midi_file}')
        return None

    # note to event
    total_event = []
    if bar_limit:
        bar_beats = bar_beats[:bar_limit + 1]


    for bar_index, bar_beat in enumerate(bar_beats):
        bar_event = []
        bar_event.append('bar')
        #print(f'bar {bar_index + 1} start at {bar_beat}')

        bar_length = bar_beats[bar_index + 1] - bar_beat if bar_index < len(bar_beats) - 1 else bar_beats[bar_index] - bar_beats[bar_index - 1]
        bar_start_time = bar_beat
        bar_end_time = bar_beats[bar_index + 1] if bar_index < len(bar_beats) - 1 else bar_beats[bar_index] + bar_length

        # time signature change in this bar
        ts_count = 0
        for ts in midi_data.time_signature_changes:
            # at most one time signature change in one bar, if more than two skip this file
            if ts.time >= bar_beat and ts.time < bar_beat + bar_length:
                ts_count += 1
                if ts_count > 1:
                    print(f'time signature change more than one in one bar, omit {midi_name}')
                    return None
                # append time signature token in the format such as 4/4
                bar_event.append(f'{ts.numerator}/{ts.denominator}')

        # tempo change in this bar
        tempo_count = 0
        for tempo_time, tempo_value in tempos:
            if tempo_time >= bar_beat and tempo_time < bar_beat + bar_length:
                tempo_count += 1
                if tempo_count > 1:
                    print(f'tempo change more than one in one bar, omit {midi_name}')
                    return None
                # append tempo token
                bar_event.append(str(tempo_value))

        # append note in each track in this bar
        for track_num, instrument in enumerate(midi_data.instruments):

            notes = instrument.notes



            # append note event in this bar
            # the basic format for tokens is p_{pitch} and d_{duration}
            # for example, p_60, d_16th
            # If several notes start and end at the same time
            # only append the duration at the end of the last note
            # between two adjacent notes, there is a rest if they don't overlap
            # the rest format is rest, d_{duration}
            # if there is partial overlap between two adjacent notes
            # use a special duration token d_{duration}_{sep}
            # that means the next note start time is set to the first note start time
            # potential rest need to be added before the next note because its start time is set to the first note start time
            # the duration of the note can be retrieved by convert_duration_ratio_name(note.velocity, all_durations)
            # when processing one note, always check the next note to decide the duration token and rest token after this note
            # also add rest tokens to make sure all the durations in this bar sum up to the bar length
            # if the last note end time is less than the bar end time, add rest token to fill the bar
            bar_track_notes = []
            for i, note in enumerate(notes):
                # if the note start time is less than the bar start time, skip this note
                if note.start < bar_start_time-0.01:
                    continue
                # if the note start time is larger than the bar end time, skip this note
                if note.start >= bar_end_time-0.01:
                    break
                bar_track_notes.append(note)

            # find the time signature in this bar
            ts = None
            for ts in reversed(midi_data.time_signature_changes):
                if ts.time < bar_beat + bar_length:
                    break
            all_durations,_ = get_all_durations(ts)

            bar_track_events = note_to_event(bar_track_notes, bar_start_time, bar_end_time, bar_length, all_durations)
            # append track type token
            bar_track_events.insert(0, midi_track_types[track_num])
            bar_event.append(bar_track_events)
            # print each bar track event
            # for event in bar_track_events:
            #     print(event)

        total_event.append(bar_event)

    # if there is no pitch event in the beginning or ending bars, remove them
    bar_start_num = 0
    bar_end_num = len(total_event)-1
    beginning_tempo = []
    beginning_ts = []
    found_start_bar = False
    # find the first bar with pitch event
    for bar_num,bar_event in enumerate(total_event):
        if bar_start_num != 0 or found_start_bar:
            break
        for events in bar_event:
            # test if events is a list
            if isinstance(events, list):
                flattened_events = flatten(events)
                for event in flattened_events:
                    if 'p_' in event:
                        bar_start_num = bar_num
                        found_start_bar = True
                        break
            else:
                # keep the time signature and tempo change in the beginning bars
                if '/' in events:
                    beginning_ts.append(events)
                elif events.isdigit() or '.' in events:
                    beginning_tempo.append(events)
                else:
                    pass

    # add the latest time signature and tempo change to the first bar with pitch event
    if beginning_ts:
        # need to check if there is already time signature change in the first bar with pitch event
        if '/' not in total_event[bar_start_num]:
            total_event[bar_start_num].insert(1, beginning_ts[-1])

        else:
            # need to replace the time signature change in the first bar with pitch event
            # find the position of the time signature change in the first bar with pitch event
            for event_num,event in enumerate(total_event[bar_start_num]):
                if '/' in event:
                    total_event[bar_start_num][event_num] = beginning_ts[-1]
                    break
    # the same logic for tempo change
    if beginning_tempo:
        if beginning_tempo[-1] not in total_event[bar_start_num]:
            total_event[bar_start_num].insert(1, beginning_tempo[-1])
        else:
            for event_num,event in enumerate(total_event[bar_start_num]):
                # test if event is list

                if event.isdigit() or '.' in event:
                    total_event[bar_start_num][event_num] = beginning_tempo[-1]
                    break


    # add the midi program change event to the first bar with pitch event

    total_event[bar_start_num].insert(1, midi_programs)
    found_end_bar = False

    # find the last bar with pitch event
    for bar_num,bar_event in enumerate(total_event[::-1]):
        if bar_end_num != len(total_event)-1 or found_end_bar:
            break
        for events in bar_event:
            # test if events is a list
            if isinstance(events, list):
                flattened_events = flatten(events)
                for event in flattened_events:
                    if 'p_' in event:
                        bar_end_num = len(total_event)-1-bar_num
                        found_end_bar = True
                        break

    # remove the bars without pitch event in the beginning and ending
    total_event = total_event[bar_start_num:bar_end_num+1]

    # print(total_event)
    # Add the duration counts to the file's record
     # Extract file name from path
    file_durations[midi_name] = duration_counts

    return file_durations, total_event

# convert note in a bar to event
def note_to_event(notes, bar_start_time, bar_end_time, bar_length, all_durations):
    note_events = []
    # if there is no note in this bar, use rest to fill the bar
    if len(notes) == 0:
        rest_duration = round(bar_end_time - bar_start_time,4)
        rest_token = ['rest']
        durations = convert_duration_ratio_name(rest_duration / bar_length, all_durations,separate=True)

        for duration in durations:
            rest_token.append('d_' + duration)
        note_events.append(rest_token)
        return note_events

    note_token = []
    # if the first note start time is larger than the bar start time, add rest token
    for i, note in enumerate(notes):
        try:
            if i == 0 and not math.isclose(note.start, bar_start_time, abs_tol=0.01):
                rest_duration = round(note.start - bar_start_time,4)
                rest_token = ['rest']
                durations = convert_duration_ratio_name(rest_duration / bar_length, all_durations,separate=True)

                for duration in durations:
                    rest_token.append('d_' + duration)

                note_events.append(rest_token)


            note_token.append('p_'+str(note.pitch))

            if i < len(notes) - 1:
                next_note = notes[i+1]
                # handle partial overlapping notes
                if next_note.start < note.end-0.01 and next_note.end > note.start+0.01:
                    # if two notes have the same start and end time, skip to add the second note duration
                    if math.isclose(next_note.start, note.start, abs_tol=0.01) and math.isclose(next_note.end, note.end, abs_tol=0.01):
                        continue

                    # if two notes not totally overlap
                    # use a special duration token to set the next note start time to the first note start time
                    durations = convert_duration_ratio_name(note.ratio, all_durations,separate=True)

                    for duration in durations:
                        note_token.append('d_' + duration + '_sep')
                    note_events.append(note_token)
                    note_token = []
                    # add rest token
                    duration_diff = round(next_note.start - note.start,4)

                    durations = convert_duration_ratio_name(duration_diff / bar_length, all_durations,separate=True)
                    if len(durations) > 0:
                        rest_token = ['rest']
                        for duration in durations:
                            rest_token.append('d_' + duration)
                        note_events.append(rest_token)
                    continue
                # handle non-overlapping notes
                if next_note.start > note.end-0.01:
                    note_duration = []
                    durations = convert_duration_ratio_name(note.ratio, all_durations, separate=True)

                    for duration in durations:
                        note_duration.append('d_' + duration)
                    note_token.extend(note_duration)
                    note_events.append(note_token)
                    note_token = []
                    # add rest token
                    rest_duration = round(next_note.start - note.end,4)
                    durations = convert_duration_ratio_name(rest_duration / bar_length, all_durations,separate=True)

                    if len(durations) > 0:
                        rest_token = ['rest']
                        for duration in durations:
                            rest_token.append('d_' + duration)
                        note_events.append(rest_token)
                    continue
            # handle the last note
            if i == len(notes) - 1:
                note_duration = []
                durations = convert_duration_ratio_name(note.ratio, all_durations,separate=True)

                for duration in durations:
                    note_duration.append('d_' + duration)
                note_token.extend(note_duration)
                note_events.append(note_token)


                # Add a rest and duration if the notes do not fill to the end of the bar
                if note.end < bar_end_time-0.01:

                    rest_duration = bar_end_time - note.end
                    durations = convert_duration_ratio_name(round(rest_duration / bar_length,4), all_durations,separate=True)
                    if len(durations) > 0:
                        rest_token = ['rest']
                        for duration in durations:
                            rest_token.append('d_' + duration)
                        note_events.append(rest_token)
        # catch all exceptions
        except Exception as e:
            print (e)
    return note_events


def get_basic_durations(time_signature):
    numerator, denominator = time_signature.numerator, time_signature.denominator
    basic_durations = {
        # "32th": round(1 / (numerator * (32 / denominator)), 4),
        "16th": round(1 / (numerator * (16 / denominator)), 4),
        "8th": round(1 / (numerator * (8 / denominator)), 4),
        "quarter": round(1 / (numerator * (4 / denominator)), 4),
        "half": round(1 / (numerator * (2 / denominator)), 4),
        "whole": 1.0,
        "16th_triplet": round(1 / (numerator * (24 / denominator)), 4),
        "8th_triplet": round(1 / (numerator * (12 / denominator)), 4),
        "4th_triplet": round(1 / (numerator * (6 / denominator)), 4),
        "2th_triplet": round(1 / (numerator * (3 / denominator)), 4)
    }
    # remove whole note if time signature is not in 4/4, 2/2, 8/4, 16/4
    # check each of the time signature to the four time signatures including 4/4, 2/2, 8/4, 16/4
    if numerator == 4 and denominator == 4 or numerator == 2 and denominator == 2 or numerator == 8 and denominator == 4 or numerator == 16 and denominator == 4:
        return basic_durations
    else:
        basic_durations.pop("whole")

    return basic_durations


def remove_triplets(durations):
    return {key: value for key, value in durations.items() if "triplet" not in key}


def get_all_durations(time_signature):
    basic_durations = get_basic_durations(time_signature)
    basic_durations_without_triplet = remove_triplets(basic_durations)
    combinations_durations = {}
    for i in range(2, 3):
        for combo in itertools.combinations(basic_durations.items(), i):
            # sort combo by duration from small to large
            combo = sorted(combo, key=lambda x: x[1], reverse=True)
            combo_name = "and".join([name for name, _ in combo])
            combo_duration = sum([duration for _, duration in combo])
            combinations_durations[combo_name] = float("{:.4f}".format(combo_duration))

    for i in range(3, len(basic_durations_without_triplet) + 1):
        for combo in itertools.combinations(basic_durations_without_triplet.items(), i):
            # sort combo by duration from small to large
            combo = sorted(combo, key=lambda x: x[1], reverse=True)
            combo_name = "and".join([name for name, _ in combo])
            combo_duration = sum([duration for _, duration in combo])
            combinations_durations[combo_name] = float("{:.4f}".format(combo_duration))

    all_durations = {**basic_durations, **combinations_durations}

    sorted_durations = sorted(all_durations.items(), key=lambda x: (x[1], len(x[0])))
    sorted_keys = []

    previous_value = None
    for key, value in sorted_durations:
        if previous_value is None or value != previous_value:
            sorted_keys.append(key)
            previous_value = value

    all_durations = {k: v for k, v in all_durations.items() if k in sorted_keys}
    # sort all durations by duration from small to large
    all_durations = OrderedDict(sorted(all_durations.items(), key=lambda x: x[1]))

    all_durations_without_triplet = {
        name: duration
        for name, duration in all_durations.items()
        if "triplet" not in name
    }

    all_durations_without_triplet = OrderedDict(sorted(all_durations_without_triplet.items(), key=lambda x: x[1]))

    return all_durations, all_durations_without_triplet


def convert_duration_ratio_name(duration_ratio, all_durations,separate=False):
    smallest_duration_ratio = min(all_durations.values())
    if duration_ratio < smallest_duration_ratio / 2 :
        return []
    for name, value in all_durations.items():
        if separate:
            if math.isclose(duration_ratio,value,abs_tol=0.001):
                if 'and' in name:
                    return name.split('and')
                else:
                    return [name]
        else:
            if math.isclose(duration_ratio,value,abs_tol=0.001):
                return name
    # if not found, return the closest duration
    closest_duration = min(
        all_durations.values(),
        key=lambda x: abs(x - duration_ratio)
    )
    for name, value in all_durations.items():
        if value == closest_duration:
            if separate:
                if 'and' in name:
                    return name.split('and')
                else:
                    return [name]
            else:
                return name



def check_triplet_neighboring_notes(note, notes, bar_length, all_durations):
    time_range = 0.03
    threshold = 0.001
    prev_notes = [n for n in notes if n.end < note.start + time_range and abs(note.start - n.end) < time_range]
    next_notes = [n for n in notes if n.start > note.end - time_range and abs(n.start - note.end) < time_range]

    for prev_note in reversed(prev_notes):
        duration_ratio = (prev_note.duration) / bar_length

        closest_duration = min(
            all_durations.values(),
            key=lambda x: abs(x - duration_ratio)
        )

        # Thresholding: Consider duration as triplet only if it's close to the exact triplet duration
        if closest_duration == all_durations["16th_triplet"] or closest_duration == all_durations[
            "8th_triplet"] or closest_duration == all_durations["4th_triplet"] or closest_duration == all_durations["2th_triplet"]:
            # Adjust this threshold as needed
            if abs(closest_duration - duration_ratio) > threshold:
                return False
            else:
                return True

    for next_note in next_notes:

        duration_ratio = next_note.duration / bar_length

        closest_duration = min(
            all_durations.values(),
            key=lambda x: abs(x - duration_ratio)
        )

        # Thresholding: Consider duration as triplet only if it's close to the exact triplet duration
        if closest_duration == all_durations["16th_triplet"] or closest_duration == all_durations[
            "8th_triplet"] or closest_duration == all_durations["4th_triplet"] or closest_duration == all_durations["2th_triplet"]:
            # Adjust this threshold as needed
            if abs(closest_duration - duration_ratio) > threshold:
                return False
            else:
                return True

    return False



# convert a flattened event sequence to a midi file
# the first bar has the programs of the tracks
# the tempo change and time signature change are after the 'bar' token if there is any
# need to calculate the bar length based on the time signature and tempo change
# then calculate the duration of each note
# the duration token ratio can be retrieved from the all_durations dict
# the duration in seconds can be calculated by multiplying the duration token ratio with the bar length
# by default, the next token start time is the end time of the previous token
# the duration token ends with 'sep' means the next token start time is the same with this note
def event_to_midi(event_list, output_file,write_file=False):
    '''

    Parameters
    ----------
    event: list
    save_path: str

    Returns
    -------

    '''

    current_bar_start_time = 0.0  # This is the start time of the current bar
    current_time = 0 # This is the current time in seconds
    current_bar_length = 0 # This is the length of the current bar in seconds

    # find the first event start with 'program_'
    # end with the first event not start with 'program_'
    program_names = []
    for event in event_list:
        if event.startswith('program'):
            program_names.append(event)
        else:
            if len(program_names) > 0:
                break

    # convert tempo from t/level to a float number
    # find if there is a token start with 't_'
    # if there is, find the index to change it later
    tempo = [event for event in event_list if event.startswith('t_')]
    if len(tempo) > 0:
        tempo = tempo[0]
        tempo_index = event_list.index(tempo)
        tempo = vocab_2023.tempo_bins[int(tempo[2:])]
        event_list[tempo_index] = str(tempo)


    # remove all the tokens in the vocab_2023.track_control_tokens
    # and vocab_2023.bar_control_tokens
    event_list = [event for event in event_list if event not in vocab_2023.track_control_tokens and event not in vocab_2023.bar_control_tokens]


    all_track_names = ['melody', 'bass', 'chord', 'accompaniment', 'drum']
    track_names = [track_name for track_name in all_track_names if track_name in event_list]
    num_tracks = len(track_names)
    instruments = [None] * num_tracks


    # Create a dictionary from track names to program numbers
    # can use zip() to create a dictionary
    track_dict = dict(zip(track_names, program_names))

    # create a dictionary from track names to track number
    track_number_dict = dict(zip(track_names, range(num_tracks)))

    bar_count = 0
    default_tempo = 120
    # find the tempo in the event list
    for event in event_list:
        if event.isdigit() or '.' in event:
            default_tempo = float(event)
    midi_object = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)

    denominator = 4
    numerator = 4
    current_tempo = 120
    first_bar = True

    duration_sep = False
    pending_pitches = []  # Stores pitch tokens until we encounter a duration token
    pending_durations = []  # Stores duration tokens until we encounter a pitch / rest token

    for event in event_list:
        try:

            if not event.startswith('d_'):

                # check if the pending duration list is empty
                if pending_durations:
                    # If the current event is not a duration token
                    current_duration = sum(pending_durations)  # Sum up all pending durations

                    pending_durations = []  # Clear the list of pending durations

                    for pitch in pending_pitches:  # Create notes for all pending pitch tokens

                        current_note = pretty_midi.Note(velocity=64, pitch=pitch, start=current_time,

                                                        end=current_time + current_duration)

                        current_track.notes.append(current_note)

                    pending_pitches = []  # Clear the list of pending pitch tokens

                    if not duration_sep:  # If the last duration token didn't end with '_sep'

                        current_time += current_duration



            if event == 'bar':
                # We're starting a new bar, so reset the current time to the start time of this bar
                if first_bar:
                    first_bar = False
                else:
                    current_bar_start_time += current_bar_length
                bar_count += 1
                # print('bar', bar_count, 'start time', current_bar_start_time)
            elif event in track_dict.keys():
                # We're starting a new track, so reset the current time to the start time of this bar
                current_time = current_bar_start_time
                current_track_number = track_number_dict[event]
                current_track = instruments[current_track_number]

            elif 'program_' in event:
                # We're setting the program number for the current track
                # e.g. program_0, program_5, program_1
                # need to find the position of program tokens, e.g. program_1 is the 2nd program token
                # and program_0 is the first program token
                # the position of program token is to be the track number

                program_number = int(event.split('_')[1])  # Get the program number
                track_index = len([track for track in instruments if track is not None])  # Count non-None tracks as index
                instruments[track_index] = pretty_midi.Instrument(program=program_number)
                track_name = track_names[track_index]
                if track_name == 'drum':
                    instruments[track_index].is_drum = True

            elif '/' in event:  # This is a time signature token
                numerator, denominator = map(int, event.split('/'))
                ts = pretty_midi.TimeSignature(numerator, denominator, current_time)
                midi_object.time_signature_changes.append(ts)
                current_bar_length = (60 / current_tempo) * (4 / denominator) * numerator
                all_durations = get_all_durations(ts)[0]
                # add the time signature change message to the midi object, considering the current time.
            # find the tempo change token
            # the tempo is a float number converted to string
            # the tempo change token is like '120.00234'
            elif event.isdigit() or '.' in event:
                current_tempo = float(event)
                current_bar_length = (60 / current_tempo) * (4 / denominator) * numerator

                # add the tempo change message to the midi object, considering the current time.

            elif event.startswith('p_'):  # If this is a pitch token

                pending_pitches.append(int(event[2:]))

            elif event.startswith('d_'):  # This is a duration token

                # get the duration names

                if event.endswith('_sep'):  # If the duration token ends with _sep,

                    duration_name = event[2:-4]  # remove the _sep suffix

                    duration_sep = True

                else:

                    duration_name = event[2:]

                    duration_sep = False
                if duration_name not in all_durations.keys():
                    print('Unknown duration: {}'.format(duration_name))
                    return None
                pending_durations.append(all_durations[duration_name] * current_bar_length)
            elif event == 'rest':
                continue
            else:
                print('Unknown event: {}'.format(event))
                return None
        # catch the exception
        except ValueError as e:
            print(e)
            print('Error parsing event: {}'.format(event))
            return None

    # If there are any pending pitch / duration tokens, create notes for them
    if pending_pitches and pending_durations:
        current_duration = sum(pending_durations)  # Sum up all pending durations
        for pitch in pending_pitches:  # Create notes for all pending pitch tokens
            current_note = pretty_midi.Note(velocity=64, pitch=pitch, start=current_time,
                                            end=current_time + current_duration)
            current_track.notes.append(current_note)


    for instrument in instruments:
        midi_object.instruments.append(instrument)
    if write_file:
        midi_object.write(output_file)
    return midi_object

#
# track_dict = json.load(open('/media/data5/rg408/dataset/MMD_processed/program_result.json', 'rb'))
# for key, value in track_dict.items():
#     new_key = os.path.basename(key)
#     # change the old key name to the new key name in the dict
#     track_dict[new_key] = track_dict.pop(key)
#
#
# output_dir = '/media/data5/rg408/dataset/encoded_v1_by_v2'
# # create the output directory if it does not exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)


input_dir = ''
midi_files = find_all_files('input_dir','mid')
# midi_files = find_all_files('/media/data5/rg408/dataset/encoded_midi','mid')
# print(len(midi_files))
# midi_files = ['/media/data5/rg408/dataset/MMD_processed/7d1b48ff7d6bcdb95234a34e3d84b9c1.mid']
# for midi_file in midi_files:
#     pm = pretty_midi.PrettyMIDI(midi_file)
#     if len(pm.get_tempo_changes()[0]) > 1:
#         continue
#     # if there are multiple tempo changes or time signature changes in the midi file, process that file.
#     # if len(pm.time_signature_changes) > 1:
#     # logging.info(midi_file)
#     # result = midi_to_event(midi_file, track_dict)
#     result = midi_to_event(midi_file, None)
#     if result:
#         base_name = os.path.basename(midi_file)
#         duration_count, total_event = result
#         result_sequence = flatten(total_event)
# #         # write the result sequence to a file
#         output_sequence_name = os.path.join(output_dir, base_name + '.txt')
#         with open(output_sequence_name, 'w') as f:
#             f.write(str(result_sequence))
#
#         output_name = os.path.join(output_dir, base_name)
#         event_to_midi(result_sequence, output_name)



def process_midi_file(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)
    if len(pm.get_tempo_changes()[0]) > 1:
        return None
    # if there are multiple tempo changes or time signature changes in the midi file, process that file.
    # if len(pm.time_signature_changes) > 1:
    #     logging.info(midi_file)
    result = midi_to_event(midi_file, track_dict)
    if result:
        base_name = os.path.basename(midi_file)
        duration_count, total_event = result
        result_sequence = flatten(total_event)
        # write the result sequence to a file
        output_sequence_name = os.path.join(output_dir, base_name + '.txt')
        with open(output_sequence_name, 'w') as f:
            f.write(str(result_sequence))

        # output_name = os.path.join(output_dir, base_name)
        # event_to_midi(result_sequence, output_name)


events = midi_to_event('/its/home/rg408/m4l/melody.mid')
print(events)
events = flatten(events[1])
for event in events:
    if event == 'bar':
        print('\n')
    print(event)

for midi_file in midi_files:
    pm = pretty_midi.PrettyMIDI(midi_file)

    # if there are multiple tempo changes or time signature changes in the midi file, process that file.
    if len(pm.time_signature_changes) > 1:
        print(midi_file)

track_dict = None

# Create a multiprocessing pool with the desired number of processes
pool = multiprocessing.Pool(processes=16)
#
# # Use tqdm to create a progress bar for the MIDI file processing
with tqdm(total=len(midi_files), ncols=80) as pbar:
    def update():
        pbar.update()

    # Map the midi_files to the processing function
    for _ in pool.imap_unordered(process_midi_file, midi_files):
        update()

# Close the pool to release resources
pool.close()
pool.join()



