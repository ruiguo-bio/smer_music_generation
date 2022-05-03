

import random

import gc
import numpy as np
import copy
# import data_convert
import math
from collections import Counter
import pretty_midi
# import music21
import sys
# import data_convert
# from preprocessing import event_2midi,midi_2event
# from data_convert import remi_2midi
import pickle
# from preprocessing import remove_empty_track

import re
import logging
import coloredlogs

import os
import vocab

# import tension_calculation
from joblib import Parallel, delayed

control_bins = np.arange(0, 1, 0.1)
tensile_bins = np.arange(0, 2.1, 0.2).tolist() + [4]
diameter_bins = np.arange(0, 4.1, 0.4).tolist() + [5]

tempo_bins = np.array([0] + list(range(60, 190, 30)) + [200])
tension_bin = np.arange(0,6.5,0.5)
tension_bin[-1] = 6.5


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



#
def walk(folder_name,suffix):
    files = []
    for p, d, f in os.walk(folder_name):
        for file_name in f:

            if file_name[-len(suffix):] == suffix:
                files.append(os.path.join(p, file_name))

    return files




def stack_batches(files,max_token_length=2200,augment=False,add_control=False,rest_multi=True,test_dataset=False):


    logger.info(f'total files {len(files)}')


    logger.info(f'augment is {augment}')
    logger.info(f'add control is {add_control}')
    random.seed(99)


    #debug
    return_events = []
    total_number = 0



    for one_file in files:


        events = pickle.load(open(one_file,'rb'))
        
        for event in events:

            r = re.compile('track_\d')
    
            track_names = list(set(filter(r.match, event)))
            track_names.sort()
    
            bar_poses = np.where(np.array(event) == 'bar')[0]
    
            r = re.compile('i_\d')
    
            track_program = list(filter(r.match, event))
            track_nums = len(track_program)
    
            if track_nums != len(track_names):
                print('invalid data')

    
            if track_nums != len(track_names):
                print('invalid data')

    
            r = re.compile('d_\d')
            density_controls = set(filter(r.match, event))
            if len(density_controls) > 0:
                for density_token in event[3:3 + track_nums]:
                    if density_token not in vocab.track_note_density_token:
                        print('invalid data')

    
            r = re.compile('o_\d')
            occupation_controls = set(filter(r.match, event))
            if len(occupation_controls) > 0:
                for occupation_token in event[3 + track_nums:3 + track_nums * 2]:
                    if occupation_token not in vocab.track_occupation_rate_token:
                        print('invalid data')

    
            r = re.compile('y_\d')
            polyphony_controls = set(filter(r.match, event))
            if len(polyphony_controls) > 0:
                for polyphony_token in event[3 + track_nums * 2:3 + track_nums * 3]:
                    if polyphony_token not in vocab.track_polyphony_rate_token:
                        print('invalid data')




        return_events.append(events)
        total_number += 1
        # print(total_number)
    logger.info(f'total number is {len(return_events)}')

    if test_dataset:
        return return_events,None


    batches = []
    for file_events in return_events:
        if file_events:
            for event in file_events:
                batches.append(event)

    batches.sort(key=len)
    i = 0
    while i < len(batches) - 1:
        if np.array_equal(batches[i],batches[i + 1]):
            del batches[i + 1]
        else:
            i += 1

    batches_new = []
    this_batch_total_length = 0

    while len(batches) > 0:
        if this_batch_total_length + len(batches[0]) < max_token_length:
            if len(batches_new) > 0:
                batches_new[-1].append(batches[0])
            else:
                batches_new.append([batches[0]])
            this_batch_total_length += len(batches[0])
        else:
            if len(batches[0]) > max_token_length:
                logger.info(
                    f'the event size {len(batches[0])} is greater than {max_token_length}, skip this file, or increase the max token length')
                this_batch_total_length = 0
            else:
                batches_new.append([batches[0]])
                this_batch_total_length = len(batches[0])
        del batches[0]
    del batches
    gc.collect()
    batch_lengths = {}
    for index, item in enumerate(batches_new):
        if len(item) not in batch_lengths:
            batch_lengths[len(item)] = [index]
        else:
            batch_lengths[len(item)].append(index)


    return batches_new, batch_lengths



smer_event_folder = '/its/home/rg408/dataset/smer_bar_track'
remi_event_folder = '/its/home/rg408/dataset/remi_bar_track'



smer_files = walk(smer_event_folder,suffix='control')
remi_files = walk(remi_event_folder,suffix='control')
smer_files.sort()
remi_files.sort()
print(len(smer_files))
# for idx,smer_file in enumerate(smer_files):
#     smer_base_name = os.path.basename(smer_file)
#     remi_file_name = smer_base_name[:-13] + 'step_single_control'
#     remi_name = os.path.join(remi_event_folder,remi_file_name)
#     if not os.path.exists(remi_name):
#         os.remove(smer_file)
#
#
# for idx,remi_file in enumerate(remi_files):
#     remi_base_name = os.path.basename(remi_file)
#     smer_file_name = remi_base_name[:-19] + 'event_control'
#     smer_name = os.path.join(smer_event_folder,smer_file_name)
#     if not os.path.exists(smer_name):
#         os.remove(remi_file)
#
# smer_files = walk(smer_event_folder,suffix='control')
# remi_files = walk(remi_event_folder,suffix='control')
# smer_files.sort()
# remi_files.sort()
# print(len(smer_files))

#
# for idx,smer_file in enumerate(smer_files):
#     if idx % 1000 == 0:
#         print(idx)
#     smer_base_name = os.path.basename(smer_file)
#     remi_file_name = smer_base_name[:-13] + 'step_single_control'
#     remi_name = os.path.join(remi_event_folder, remi_file_name)
#     smer_events = pickle.load(open(smer_file,'rb'))
#     remi_events = pickle.load(open(remi_name, 'rb'))
#
#
#     if len(smer_events) < len(remi_events):
#         for _ in range(len(remi_events) - len(smer_events)):
#             remi_events.pop(-1)
#
#     for idx,smer_event in enumerate(smer_events):
#         remove_indices = []
#         if idx < len(remi_events):
#             remi_event = remi_events[idx]
#             smer_start_bar_pos = np.where(np.array(smer_event) == 'bar')[0][0]
#
#
#
#             r = re.compile('d_\d')
#             density_numbers = len(list(filter(r.match, smer_event[smer_start_bar_pos:])))
#             r = re.compile('o_\d')
#             occupation_numbers = len(list(filter(r.match, smer_event[smer_start_bar_pos:])))
#             r = re.compile('y_\d')
#             polyphony_numbers = len(list(filter(r.match, smer_event[smer_start_bar_pos:])))
#
#             if density_numbers == occupation_numbers == polyphony_numbers:
#
#                 remi_start_bar_pos = np.where(np.array(remi_event) == 'bar')[0][0]
#
#                 r = re.compile('d_\d')
#                 density_numbers = len(list(filter(r.match, remi_event[remi_start_bar_pos:])))
#                 r = re.compile('o_\d')
#                 occupation_numbers = len(list(filter(r.match, remi_event[remi_start_bar_pos:])))
#                 r = re.compile('y_\d')
#                 polyphony_numbers = len(list(filter(r.match, remi_event[remi_start_bar_pos:])))
#
#                 if density_numbers == occupation_numbers == polyphony_numbers:
#                     continue
#                 else:
#                     remove_indices.append(idx)
#             else:
#                 remove_indices.append(idx)
#
#
#         else:
#             remove_indices.append(idx)
#
#     for idx in remove_indices[::-1]:
#         if idx < len(smer_events):
#             smer_events.pop(idx)
#         if idx < len(remi_events):
#             remi_events.pop(idx)
#
#     pickle.dump(smer_events,open(smer_file,'wb'))
#     pickle.dump(remi_events, open(remi_name, 'wb'))
#

rest_multi = True


create_test = True
logger = logging.getLogger(__name__)

logger.handlers = []

add_control = True

if create_test:
    if add_control:
        if rest_multi:
            logfile = 'rest_multi_control_test.log'
        else:
            logfile = 'step_single_control_test.log'
    else:
        if rest_multi:
            logfile = 'rest_multi_test.log'
        else:
            logfile = 'step_single_test.log'
else:
    if add_control:
        if rest_multi:
            logfile = 'dataset_rest_multi_all_control_training_augment.log'
        else:
            logfile = 'dataset_step_single_all_control_training_augment.log'
    else:
        if rest_multi:
            logfile = 'dataset_rest_multi_training_augment.log'
        else:
            logfile = 'dataset_step_single_training_augment.log'


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S', filename=logfile, filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logger.addHandler(console)

coloredlogs.install(level='INFO', logger=logger, isatty=True)


#
# output_folder = '/home/data/guorui/score_transformer/sync/'
output_folder = '/its/home/rg408/dataset/events/'
if rest_multi:
    use_files = smer_files
else:
    use_files = remi_files
start_num = int(len(use_files)*.9)
end_num = int(len(use_files))
# end_num = len(files)
logger.info(f'start file num is {start_num}')
logger.info(f'end file num is {end_num}')

create_test = True
if rest_multi:
    print(smer_event_folder)
else:
    print(remi_event_folder)

logger.info(f'create test is  {create_test}')

training_all_batches, training_batch_length = stack_batches(use_files[start_num:end_num],augment=False,add_control=add_control,rest_multi=rest_multi,test_dataset=create_test)

if add_control:
    if rest_multi:
        if create_test:
            pickle.dump(training_all_batches, open(os.path.join(output_folder,'smer_bar_evaluation_batch'), 'wb'))

        else:
            pickle.dump(training_all_batches, open(os.path.join(output_folder, 'smer_test_batch'), 'wb'))
            pickle.dump(training_batch_length,
                        open(os.path.join(output_folder, 'smer_test_batch_lengths'), 'wb'))
    else:
        if create_test:

            pickle.dump(training_all_batches, open(os.path.join(output_folder, 'remi_mock_test_evaluation_batch'), 'wb'))

        else:
            pickle.dump(training_all_batches, open(os.path.join(output_folder, 'remi_bar_training_batch'), 'wb'))
            pickle.dump(training_batch_length,
                        open(os.path.join(output_folder, 'remi_bar_training_batch_lengths'), 'wb'))



else:
    if rest_multi:
        pickle.dump(training_all_batches,
                    open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_training_batches_0', 'wb'))
        pickle.dump(training_batch_length,
                    open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_training_batch_lengths_0',
                         'wb'))
    else:
        pickle.dump(training_all_batches,
                    open(f'/home/data/guorui/score_transformer/sync/step_single_augment_training_batches_0', 'wb'))
        pickle.dump(training_batch_length,
                    open(f'/home/data/guorui/score_transformer/sync/step_single_augment_training_batch_lengths_0',
                         'wb'))

