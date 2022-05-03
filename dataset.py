import random
import torch

import numpy as np
from torch.utils.data import Dataset

from einops import rearrange
import re
import copy


class ParallelLanguageDataset(Dataset):
    def __init__(self,
                 vocab,
                 batches,
                 batch_lengths,batch_size,
                 total_mask_ratio,
                 logger,
                 pretraining=True,
                 verbose=False,
                 bar_track_control=False,
                 bar_control_at_end=False
                 ):

        random.seed(99)
        self.vocab = vocab
        self.batch_size = batch_size
        self.verbose = verbose
        self.logger = logger

        self.batches = batches
        # self.batches = self.batches[0]
        self.batch_lengths = batch_lengths
        # self.batch_lengths = self.batch_lengths[0]
        self.total_mask_ratio = total_mask_ratio

        self.previous_index = 0

        self.pretraining = pretraining
        self.bar_track_control = bar_track_control
        self.bar_control_at_end = bar_control_at_end
        control_types = set(self.vocab.token_class_ranges.values())

        total_track_control_types = 0
        if 'density' in control_types:
            total_track_control_types += 1
        if 'occupation' in control_types:
            total_track_control_types += 1
        if 'polyphony' in control_types:
            total_track_control_types += 1
        self.total_track_control_types = total_track_control_types
        if 'tensile' in control_types:
            self.tension_control = True
        else:
            self.tension_control = False

        self.len = len(batches)

    def __getitem__(self, idx):

        if self.batch_lengths == 0:
            return_idx = idx
        else:
            if idx % self.batch_size == 0:
                this_idx = random.randint(0,len(self.batches)-1)
                if this_idx + self.batch_size - 1 > len(self.batches) - 1:
                    this_idx = this_idx - self.batch_size + 1
                self.previous_index = this_idx
            else:
                self.previous_index += 1
                this_idx = self.previous_index


            if this_idx > len(self.batches) - 1:
                print(f'invalid this index {this_idx}')
                print(f'idx is {idx}')
                this_idx = len(self.batches) - 1

            if self.batch_lengths != 0:
                length = len(self.batches[this_idx])
                return_idx = random.choice(self.batch_lengths[length])

        #
        event = self.batches[return_idx]


        # remove all the token not in

        for one_batch in event:
            remove_idx = []
            for idx,token in enumerate(one_batch):
                if token not in self.vocab.control_tokens and \
                        token not in self.vocab.basic_tokens:
                    remove_idx.append(idx)

            for idx in remove_idx[::-1]:
                one_batch.pop(idx)

        r = re.compile('i_\d')


        for one_batch in event:
            track_program = set(filter(r.match, one_batch))
            track_nums = len(track_program)
            r = re.compile('track_\d')

            track_names = list(set(filter(r.match, one_batch)))
            track_names.sort()

            bar_poses = np.where(np.array(one_batch) == 'bar')[0]

            track_poses = []
            for track_name in track_names:
                track_pos = np.where(track_name == np.array(one_batch))[0]
                track_poses.extend(track_pos)
            track_poses.extend(bar_poses)

            all_track_pos = list(np.sort(track_poses))
            all_track_pos.append(len(one_batch))

            if self.bar_track_control:
                if self.bar_control_at_end:
                    # if last token is control, inserted before, continue
                    if one_batch[-1] in self.vocab.control_tokens:
                        continue

                    ## copy the bar_track control from track beginning to track end
                    ## copy the tensile control from the bar beginning to the bar end


                    for back_pos in range(len(all_track_pos)-1,-1,-1):
                        if all_track_pos[back_pos] in bar_poses:
                            # print(back_pos)

                            bar_pos = all_track_pos[back_pos]
                            # print(bar_pos)
                            if back_pos + track_nums + 1 >= len(all_track_pos):
                                print(back_pos + track_nums + 1)
                            next_bar_pos = all_track_pos[back_pos + track_nums + 1]

                            # print(next_bar_pos)
                            if self.tension_control:
                                bar_control = one_batch[bar_pos+1]
                                one_batch.insert(next_bar_pos,bar_control)

                            if self.total_track_control_types > 0:
                                for track_num in range(track_nums):
                                    track_start = all_track_pos[back_pos+track_num+1] + (self.total_track_control_types) * track_num
                                    insert_pos = all_track_pos[back_pos+track_num+2] + (self.total_track_control_types) * track_num
                                    track_controls = one_batch[track_start+1:track_start+self.total_track_control_types+1]

                                    for track_control in track_controls[::-1]:
                                        one_batch.insert(insert_pos, track_control)

        if self.pretraining:
            result = self.random_word(event,self.total_mask_ratio)

        else:
            result = self.mask_bars(event)

        return result

    def __len__(self):
        return self.len

    def random_word(self,
                    events,
                    total_ratio):
        total_tokens = []
        total_decoder_in = []
        total_decoder_target = []

        span_lengths = [3,1,2]
        span_ratio_jointly = [.5, .25, .25]
        random_threshold = total_ratio / (np.dot(span_ratio_jointly, span_lengths))
        span_ratio = span_ratio_jointly

        random.shuffle(events)
        for event in events:
            if not isinstance(event,list):
                event = event.tolist()
            event = copy.copy(event)
            
            ###
            if self.bar_track_control:
                if self.bar_control_at_end:

                    r = re.compile('track_\d')

                    track_names = list(set(filter(r.match, event)))
                    track_names.sort()
        
                    bar_poses = np.where(np.array(event) == 'bar')[0]
        
                    track_poses = []
                    for track_name in track_names:
                        track_pos = np.where(track_name == np.array(event))[0]
                        track_poses.extend(track_pos)
                    track_poses.extend(bar_poses)

                    all_track_pos = list(np.sort(track_poses))
                    all_track_pos.append(len(event))


                    control_indices = []
                    start_control = False
                    for token_idx, token in enumerate(event):
                        if token in self.vocab.control_tokens:
                            if token_idx - 1 in all_track_pos:
                                control_indices.append(token_idx)
                                start_control = True
                            else:
                                if start_control:
                                    control_indices.append(token_idx)
                        else:
                            start_control = False
           
                else:
                    control_indices = []
                    for token_idx, token in enumerate(event):
                        if token in self.vocab.control_tokens:
                            control_indices.append(token_idx)
            else:
                control_indices = []
                for token_idx, token in enumerate(event):
                    if token in self.vocab.control_tokens:
                        control_indices.append(token_idx)
            # .05% corrupt track/bar control tokens
            for token_idx in control_indices:
                if random.random() < .05:
                    event[token_idx] = self.vocab.corrupt_tokens[0]

            tokens = []
            decoder_in = []
            decoder_target = []
            start_pos = 0
            total_masked_ratio = 0

            masked_num = 0

            while total_masked_ratio < total_ratio and start_pos < len(event):
                masked_token = []
                prob = random.random()

                if prob < span_ratio[0]:
                    if start_pos + span_lengths[0] <= len(event):
                        prob = random.random()
                        if prob < random_threshold * 1.5:
                            masked_token = event[start_pos:start_pos + span_lengths[0]]
                            tokens.append(self.vocab.mask_indices[masked_num])
                            total_masked_ratio += span_lengths[0] / len(event)
                            start_pos += span_lengths[0]

                elif span_ratio[0] < prob < span_ratio[1] + span_ratio[0]:
                    if start_pos + span_lengths[1] <= len(event):
                        prob = random.random()
                        if prob < random_threshold * 1.5:
                            masked_token = event[start_pos:start_pos + span_lengths[1]]
                            tokens.append(self.vocab.mask_indices[masked_num])
                            total_masked_ratio += span_lengths[1] / len(event)
                            start_pos += span_lengths[1]
                else:
                    if start_pos + span_lengths[2] <= len(event):
                        prob = random.random()
                        if prob < random_threshold * 1.5:
                            masked_token = event[start_pos:start_pos + span_lengths[2]]
                            tokens.append(self.vocab.mask_indices[masked_num])
                            total_masked_ratio += span_lengths[2] / len(event)
                            start_pos += span_lengths[2]

                if len(masked_token) > 0:
                    if not isinstance(masked_token, list):
                        masked_token = [masked_token]
                    decoder_in.append(self.vocab.mask_indices[masked_num])
                    for token in masked_token:
                        decoder_in.append(self.vocab.char2index(token))
                        decoder_target.append(self.vocab.char2index(token))
                    else:
                        decoder_target.append(self.vocab.eos_index)

                else:
                    tokens.append(self.vocab.char2index(event[start_pos]))
                    start_pos += 1


            while start_pos < len(event):
                tokens.append(self.vocab.char2index(event[start_pos]))
                start_pos += 1

            tokens = np.array(tokens)
            if len(decoder_in) > 0:
                decoder_in = np.array(decoder_in)
                decoder_target = np.array(decoder_target)
                total_tokens.append(tokens)
                total_decoder_in.append(decoder_in)
                total_decoder_target.append(decoder_target)

                # debug purpose
                # print('\n')
                # print(f'event length is {len(event)}')
                # print(f'tokens length is {len(tokens)}')
                # print(f'control tokens length is {total_control_tokens}')
                # print(f'masked ratio is {total_masked_ratio}')
                # print(f'control masked ratio is {control_masked_ratio["total"]}')
                # print(f'decoder_in length is {len(decoder_in)}')
                # print(f'decoder_out length is {len(decoder_target)}')
                # print(f'ratio is {(len(tokens) + len(decoder_in)) / len(event)}')

        # print(len(tokens) - len(np.where(output_label==2)[0]))
        # print(len(output_label) - len(np.where(output_label==2)[0])*2)
        return total_tokens, total_decoder_in, total_decoder_target


    def mask_bars(self,events):

        # mask bar token (w/wo bar control token) and try to generate bar token

        total_tokens = []
        total_decoder_in = []
        total_decoder_target = []



        token_lengths = []
        event_lengths = []
        decoder_in_lengths = []
        decoder_out_lengths = []
        total_lengths_list = []

        random.shuffle(events)

        prob = random.random()
        if prob > 0.6:
            mask_mode = 0
            #random tracks, random bars
        elif .3 < prob <= 0.6:
            mask_mode = 1
            # whole track
        else:
            mask_mode = 2
            #whole bar

        # self.logger.info(f'mask mode is {mask_mode}')
        total_lengths = 0
        for event in events:

            if not isinstance(event,list):

                event = event.tolist()
            event = copy.copy(event)



            tokens = []
            decoder_in = []
            decoder_target = []


            masked_indices_pairs = []

            ##add new code

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
                            this_bar_pairs.append((this_bar_poses[j]+1, this_bar_poses[j + 1]))

                        bar_with_track_poses.append(this_bar_pairs)


            mask_pairs = []

            if mask_mode == 0:
                #random tracks in random bar
                # bar_mask_number = np.random.randint(0,len(bar_poses))
                bar_number_weight = np.logspace(1, 2, num=len(bar_poses))[::-1]

                bar_mask_number = random.choices(range(len(bar_poses)), weights=bar_number_weight)[0] + 1
                bar_mask_poses = np.sort(np.random.choice(len(bar_poses),size=bar_mask_number,replace=False))

                for bar_mask_pos in bar_mask_poses:
                    masked_indices_pairs = []
                    # track_mask_number = np.random.randint(0, track_nums)
                    if track_nums == 1:
                        weight = [1]
                    if track_nums == 2:
                        weight = [10, 1]
                    if track_nums == 3:
                        weight = [10, 5, 1]
                    if len(range(track_nums)) != len(weight):
                        print('what')
                        print(range(track_nums))
                        print(weight)
                    track_mask_number = random.choices(range(track_nums), weights=weight)[0] + 1
                    track_mask_poses = np.sort(np.random.choice(track_nums,size=track_mask_number,replace=False))
                    for track_mask_pos in track_mask_poses:
                        track_start, track_end = bar_with_track_poses[bar_mask_pos][track_mask_pos]
                        # print(track_start, track_end)
                        if self.bar_track_control:
                            token_start = track_start + self.total_track_control_types
                            if self.bar_control_at_end:
                                if self.tension_control and event[track_end - 1] in \
                                        self.vocab.name_to_tokens['tensile']:
                                    tensile_end = 1
                                else:
                                    tensile_end = 0
                                token_end = track_end - self.total_track_control_types - tensile_end
                            else:

                                token_end = track_end
                        else:
                            token_start = track_start
                            token_end = track_end


                        masked_indices_pairs.append((token_start,token_end))
                        if self.bar_control_at_end:
                            for i in range(self.total_track_control_types + tensile_end):
                                masked_indices_pairs.append((token_end + i,token_end + 1 + i))


                        # corrupt
                        if self.bar_track_control:
                            if self.total_track_control_types == 3:
                                # 10% corrupt one of track control token
                                # 10% corrupt two of track control tokens
                                # 10% corrupt three of track control tokens
                                corrupt_prob = random.random()
                                if 0.2 < corrupt_prob < 0.3:
                                    mask_control_indices = np.sort(np.random.choice(range(3), 1, replace=False))
                                    for mask_control_idx in mask_control_indices:
                                        control_idx = track_start + mask_control_idx
                                        event[control_idx] = self.vocab.corrupt_tokens[0]
                                if 0.1 < corrupt_prob < 0.2:
                                    mask_control_indices = np.sort(np.random.choice(range(3), 2, replace=False))
                                    for mask_control_idx in mask_control_indices:
                                        control_idx = track_start + mask_control_idx
                                        event[control_idx] = self.vocab.corrupt_tokens[0]
                                if corrupt_prob < 0.1:
                                    mask_control_indices = range(3)
                                    for mask_control_idx in mask_control_indices:
                                        control_idx = track_start + mask_control_idx
                                        event[control_idx] = self.vocab.corrupt_tokens[0]


                            elif self.total_track_control_types == 1:

                                corrupt_prob = random.random()
                                if 0.2 < corrupt_prob < 0.3:
                                    # corrupt one control
                                    mask_control_indices = [0]
                                    for mask_control_idx in mask_control_indices:
                                        control_idx = track_start + mask_control_idx
                                        event[control_idx] = self.vocab.corrupt_tokens[0]
                            else:
                                # no track control to corrupt
                                pass
                    mask_pairs.extend(masked_indices_pairs)


                # print(mask_pairs)
                # for pair in mask_pairs:
                #     if pair[1] - pair[0] > 1:
                #         print(event[pair[0]-self.total_track_control_types:pair[1]])
                #     else:
                #         print(event[pair[0]:pair[1]])



            elif mask_mode == 1:
                # mask whole tracks
                #smaller track number has more weight


                if track_nums == 1:
                    weight = [1]
                if track_nums == 2:
                    weight = [10,1]
                if track_nums == 3:
                    weight = [10,2,1]
                # if len(range(track_nums)) != len(weight):
                #     print('what')
                #     print(range(track_nums))
                #     print(weight)
                track_mask_number = random.choices(range(track_nums),weights=weight)[0] + 1
                track_mask_poses = np.sort(np.random.choice(track_nums, size=track_mask_number, replace=False))
                for bar_num, tracks_in_a_bar in enumerate(bar_with_track_poses):

                    for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
                        if track_pos in track_mask_poses:
                            track_start, track_end = track_star_end_poses

                            if self.bar_track_control:
                                token_start = track_start + self.total_track_control_types
                                if self.bar_control_at_end:
                                    if self.tension_control and event[track_end - 1] in self.vocab.name_to_tokens['tensile']:
                                        tensile_end = 1
                                    else:
                                        tensile_end = 0
                                    token_end = track_end - self.total_track_control_types - tensile_end
                                else:

                                    token_end = track_end
                            else:
                                token_start = track_start
                                token_end = track_end

                            masked_indices_pairs.append((token_start, token_end))
                            if self.bar_control_at_end:
                                for i in range(self.total_track_control_types + tensile_end):
                                    masked_indices_pairs.append((token_end + i, token_end + 1 + i))

                if self.bar_track_control:

                    ## corrupt bar track control
                    # 50% all bars for this track
                    # 50% random length
                    if random.random() > 0.5:
                        bar_mask_number = len(bar_poses)
                    else:
                        bar_mask_number = np.random.randint(len(bar_poses))

                    bar_mask_poses = np.sort(np.random.choice(len(bar_poses), size=bar_mask_number, replace=False))


                    if self.total_track_control_types == 3:
                        corrupt_prob = random.random()
                        # 40% corrupt one track control
                        # 25% corrupt two track control
                        # 10% corrupt three track control
                        # 25% no corruption

                        if corrupt_prob > 0.6:
                            mask_control_indices = np.sort(np.random.choice(range(3), 1, replace=False))
                        elif .35 < corrupt_prob <= 0.6:
                            mask_control_indices = np.sort(np.random.choice(range(3), 2, replace=False))
                        elif .25 <corrupt_prob <= .35:
                            mask_control_indices = range(3)
                        else:
                            mask_control_indices = []
                    else:
                        corrupt_prob = random.random()
                        if corrupt_prob > 0.5:
                            mask_control_indices = [0]
                        else:
                            mask_control_indices = []


                    for bar_num, tracks_in_a_bar in enumerate(bar_with_track_poses):
                        if bar_num in bar_mask_poses:
                            for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
                                if track_pos in track_mask_poses:
                                    track_start, track_end = track_star_end_poses
                                    # corrupt

                                    for mask_control_idx in mask_control_indices:
                                        control_idx = track_start + mask_control_idx
                                        event[control_idx] = self.vocab.corrupt_tokens[0]



                mask_pairs = masked_indices_pairs



                # print(mask_pairs)
                # for pair in mask_pairs:
                #     if pair[1] - pair[0] > 1:
                #         print(event[pair[0] - self.total_track_control_types:pair[1]])
                #     else:
                #         print(event[pair[0]:pair[1]])

            else:
                # mask whole bars, smaller number has more weight
                bar_number_weight = np.logspace(1,2,num=len(bar_poses))[::-1]

                bar_mask_number = random.choices(range(len(bar_poses)),weights=bar_number_weight)[0] + 1
                # bar_mask_number = np.random.randint(0, len(bar_poses))


                if random.random() > .5:
                    # choose a position then continuously mask the next bars
                    start_bar_number = np.random.randint(0, len(bar_poses)-(bar_mask_number-1))
                    bar_mask_poses = range(start_bar_number,start_bar_number+bar_mask_number)
                else:
                    #randomly choose position
                    bar_mask_poses = np.sort(np.random.choice(len(bar_poses), size=bar_mask_number, replace=False))


                for bar_mask_pos in bar_mask_poses:
                    tracks_in_a_bar = bar_with_track_poses[bar_mask_pos]
                    for track_star_end_poses in tracks_in_a_bar:

                        track_start, track_end = track_star_end_poses

                        if self.bar_track_control:
                            token_start = track_start + self.total_track_control_types
                            if self.bar_control_at_end:
                                if self.tension_control and event[track_end - 1] in \
                                        self.vocab.name_to_tokens['tensile']:
                                    tensile_end = 1
                                else:
                                    tensile_end = 0
                                token_end = track_end - self.total_track_control_types - tensile_end
                            else:

                                token_end = track_end
                        else:
                            token_start = track_start
                            token_end = track_end

                        masked_indices_pairs.append((token_start, token_end))
                        if self.bar_control_at_end:
                            for i in range(self.total_track_control_types + tensile_end):
                                masked_indices_pairs.append((token_end + i, token_end + 1 + i))

                        # corrupt
                        if self.bar_track_control:
                            if self.total_track_control_types == 3:
                                # 10% corrupt one of track control token
                                # 10% corrupt two of track control tokens
                                # 10% corrupt three of track control tokens
                                corrupt_prob = random.random()
                                if 0.2 < corrupt_prob < 0.3:
                                    mask_control_indices = np.sort(np.random.choice(range(3), 1, replace=False))
                                    for mask_control_idx in mask_control_indices:
                                        control_idx = track_start + mask_control_idx
                                        event[control_idx] = self.vocab.corrupt_tokens[0]
                                if 0.1 < corrupt_prob < 0.2:
                                    mask_control_indices = np.sort(np.random.choice(range(3), 2, replace=False))
                                    for mask_control_idx in mask_control_indices:
                                        control_idx = track_start + mask_control_idx
                                        event[control_idx] = self.vocab.corrupt_tokens[0]
                                if corrupt_prob < 0.1:
                                    mask_control_indices = range(3)
                                    for mask_control_idx in mask_control_indices:
                                        control_idx = track_start + mask_control_idx
                                        event[control_idx] = self.vocab.corrupt_tokens[0]


                            elif self.total_track_control_types == 1:

                                corrupt_prob = random.random()
                                if 0.2 < corrupt_prob < 0.3:
                                    # corrupt one control
                                    mask_control_indices = [0]
                                    for mask_control_idx in mask_control_indices:
                                        control_idx = track_start + mask_control_idx
                                        event[control_idx] = self.vocab.corrupt_tokens[0]
                            else:
                                # no track control to corrupt
                                pass


                    # 10% corrupt bar control token
                    if self.tension_control and random.random() < .1:
                        event[tracks_in_a_bar[0][0]-2] = self.vocab.corrupt_tokens[0]
                        # print(event[tracks_in_a_bar[0][0]-2:tracks_in_a_bar[0][0]-2+3])
                mask_pairs = masked_indices_pairs

                # print(mask_pairs)
                # for pair in mask_pairs:
                #     if pair[1] - pair[0] > 1:
                #         print(event[pair[0] - self.total_track_control_types:pair[1]])
                #     else:
                #         print(event[pair[0]:pair[1]])

            token_events = event.copy()

            for masked_pair in mask_pairs:
                masked_token = event[masked_pair[0]:masked_pair[1]]
                # print(masked_token)
                decoder_in.append(self.vocab.mask_indices[0])


                for token_idx, token in enumerate(masked_token):
                    # if token in self.vocab.name_to_tokens['tensile']:
                    #     print('what')
                    decoder_in.append(self.vocab.char2index(token))
                    decoder_target.append(self.vocab.char2index(token))

                else:
                    decoder_target.append(self.vocab.eos_index)

            all_pairs = mask_pairs
            all_pairs.sort(key=lambda tup: tup[0])

            for pair in all_pairs[::-1]:

                for pop_time in range(pair[1] - pair[0]):
                    token_events.pop(pair[0])
                token_events.insert(pair[0],vocab.mask[0])

            for token in token_events:
                tokens.append(self.vocab.char2index(token))

            tokens = np.array(tokens)
            if len(decoder_in) > 0:
                decoder_in = np.array(decoder_in)
                decoder_target = np.array(decoder_target)

                # self.logger.info(f'event length is {len(event)}')
                # self.logger.info(f'tokens length is {len(tokens)}')
                # print(f'masked num is {masked_num}')
                # self.logger.info(f'decoder_in length is {len(decoder_in)}')
                # self.logger.info(f'decoder_out length is {len(decoder_target)}')
                this_total_length = len(tokens) + len(decoder_in) + len(decoder_target)
                # self.logger.info(f'this total lengths is {this_total_length}')
                total_lengths += this_total_length

                # print(f'ratio is {(len(tokens) + len(decoder_in)) / len(event)}')
                total_tokens.append(tokens)
                total_decoder_in.append(decoder_in)
                total_decoder_target.append(decoder_target)


                token_lengths.append(len(tokens))
                event_lengths.append(len(event))
                decoder_in_lengths.append(len(decoder_in))
                decoder_out_lengths.append(len(decoder_target))
                total_lengths_list.append(this_total_length)

        if len(total_tokens) == 0:
            print('why')
            return None
        # if total_lengths > 4300:
        #     # self.logger.info(f'one batch total length is {total_lengths}')
        #     self.logger.info(f'event lengths is {event_lengths}')
        #     # self.logger.info(f'token lengths is {token_lengths}')
        #     # self.logger.info(f'decoder in lengths is {decoder_in_lengths}')
        #     # self.logger.info(f'decoder out lengths is {decoder_out_lengths}')
        #     self.logger.info(f'total lengths is {total_lengths}')
        # #
        #     self.logger.info(f'mask mode is {mask_mode}')
            # total_tokens.pop()
            # total_decoder_in.pop()
            # total_decoder_target.pop()

        # print(len(tokens) - len(np.where(output_label==2)[0]))
        # print(len(output_label) - len(np.where(output_label==2)[0])*2)
        return total_tokens, total_decoder_in, total_decoder_target

    def shuffle_batches(self):
        self.batches = self.gen_batches(self.num_tokens, self.data_lengths)


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')

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


def collate_mlm_pretraining(batch):
    batch = list(filter(None, batch))
    if len(batch) == 0:
        return None
    max_input_len = max_target_len = 0
    for batch_dim in range(len(batch)):
        input_lens = [x.shape[0] for x in batch[batch_dim][0]]
        if len(input_lens) == 0:
            print(batch)
        max_input_len = max(max_input_len,max(input_lens))

        target_lens = [x.shape[0] for x in batch[batch_dim][1]]
        max_target_len = max(max_target_len,max(target_lens))

    input_pad_list = []
    input_pad_masks_list = []
    target_in_pad_list = []
    target_in_pad_masks_list = []
    target_out_pad_list = []

    for batch_dim in range(len(batch)):

        # input
        input_padded = [pad1d(x, max_input_len) for x in batch[batch_dim][0]]
        input_padded = np.stack(input_padded)

        input_pad_masks = input_padded == 0

        # target
        target_in_padded = [pad1d(x, max_target_len) for x in batch[batch_dim][1]]
        target_in_padded = np.stack(target_in_padded)

        target_in_pad_masks = target_in_padded == 0

        target_out_padded = [pad1d(x, max_target_len) for x in batch[batch_dim][2]]
        target_out_padded = np.stack(target_out_padded)


        input_pad_list.append(input_padded)
        input_pad_masks_list.append(input_pad_masks)
        target_in_pad_list.append(target_in_padded)
        target_in_pad_masks_list.append(target_in_pad_masks)
        target_out_pad_list.append(target_out_padded)



    input_pad = torch.tensor(np.concatenate(input_pad_list)).long()
    target_in_pad = torch.tensor(np.concatenate(target_in_pad_list)).long()
    target_out_pad = torch.tensor(np.concatenate(target_out_pad_list)).long()
    input_pad_masks = torch.tensor(np.concatenate(input_pad_masks_list)).bool()
    target_in_pad_masks = torch.tensor(np.concatenate(target_in_pad_masks_list)).bool()

    output = {"input": input_pad,
              "target_in": target_in_pad,
              "target_out": target_out_pad,
              "input_pad_mask": input_pad_masks,
              "target_pad_mask": target_in_pad_masks

              }

    return output


def collate_mlm_finetuning(batch):
    batch = list(filter(None, batch))
    if len(batch) == 0:
        return None
    max_input_len = max_target_len = 0
    for batch_dim in range(len(batch)):
        input_lens = [x.shape[0] for x in batch[batch_dim][0]]
        if len(input_lens) == 0:
            print(batch)
        max_input_len = max(max_input_len, max(input_lens))

        target_lens = [x.shape[0] for x in batch[batch_dim][1]]
        max_target_len = max(max_target_len, max(target_lens))

    input_pad_list = []
    input_pad_masks_list = []
    target_in_pad_list = []
    target_in_pad_masks_list = []
    target_out_pad_list = []


    for batch_dim in range(len(batch)):

        # input
        input_padded = [pad1d(x, max_input_len) for x in batch[batch_dim][0]]
        input_padded = np.stack(input_padded)

        input_pad_masks = input_padded == 0

        # target
        target_in_padded = [pad1d(x, max_target_len) for x in batch[batch_dim][1]]
        target_in_padded = np.stack(target_in_padded)

        target_in_pad_masks = target_in_padded == 0

        target_out_padded = [pad1d(x, max_target_len) for x in batch[batch_dim][2]]
        target_out_padded = np.stack(target_out_padded)


        input_pad_list.append(input_padded)
        input_pad_masks_list.append(input_pad_masks)
        target_in_pad_list.append(target_in_padded)
        target_in_pad_masks_list.append(target_in_pad_masks)
        target_out_pad_list.append(target_out_padded)

    input_pad = torch.tensor(np.concatenate(input_pad_list)).long()
    target_in_pad = torch.tensor(np.concatenate(target_in_pad_list)).long()
    target_out_pad = torch.tensor(np.concatenate(target_out_pad_list)).long()
    input_pad_masks = torch.tensor(np.concatenate(input_pad_masks_list)).bool()
    target_in_pad_masks = torch.tensor(np.concatenate(target_in_pad_masks_list)).bool()



    output = {"input": input_pad,
              "target_in": target_in_pad,
              "target_out": target_out_pad,
              "input_pad_mask": input_pad_masks,
              "target_pad_mask": target_in_pad_masks
              }

    return output


def note_density(track_events, track_length):
    densities = []
    tracks = track_events.keys()
    # print(tracks)
    for track_name in tracks:
        # print(track_name)
        note_num = 0
        this_track_events = track_events[track_name]
        # print(this_track_events)
        for track_event in this_track_events:
            for event_index in range(len(track_event) - 1):
                if track_event[event_index][0] == 'p' and track_event[event_index + 1][0] != 'p':
                    note_num += 1
        #         print(note_num / track_length)
        densities.append(note_num / track_length)
    return densities


def occupation_polyphony_rate(pm):
    occupation_rate = []
    polyphony_rate = []
    beats = pm.get_beats()
    fs = 4 / (beats[1] - beats[0])

    for instrument in pm.instruments:
        piano_roll = instrument.get_piano_roll(fs=fs)
        if piano_roll.shape[1] == 0:
            occupation_rate.append(0)
        else:
            occupation_rate.append(np.count_nonzero(np.any(piano_roll, 0)) / piano_roll.shape[1])
        if np.count_nonzero(np.any(piano_roll, 0)) == 0:
            polyphony_rate.append(0)
        else:
            polyphony_rate.append(
                np.count_nonzero(np.count_nonzero(piano_roll, 0) > 1) / np.count_nonzero(np.any(piano_roll, 0)))

    return occupation_rate, polyphony_rate


def to_category(array, bins):
    result = []
    for item in array:
        result.append(int(np.where((item - bins) >= 0)[0][-1]))
    return result


def pitch_register(track_events):
    registers = []
    tracks = track_events.keys()
    # print(tracks)
    for track_name in tracks:
        # print(track_name)
        register = []
        this_track_events = track_events[track_name]
        # print(this_track_events)
        for track_event in this_track_events:
            for event in track_event:
                if event[0] == 'p':
                    register.append(int(event[2:]))
        #         print(note_num / track_length)
        # print(np.mean(register))
        if len(register) == 0:
            registers.append(0)
        else:
            registers.append(int((np.mean(register) - 21) / 11))
    return registers


#
#
# def cal_tension(pm):
#
#
#     result = tension_calculation.extract_notes(pm, 3)
#
#
#     pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result
#
#     key_name = tension_calculation.all_key_names
#
#     result = tension_calculation.cal_tension(
#         piano_roll, beat_time, beat_indices, down_beat_time,
#         down_beat_indices, -1, key_name,sixteenth_time,pm)
#
#     tensiles, diameters, key_name,\
#     changed_key_name, key_change_beat = result
#
#     total_tension = np.array(tensiles) + 0.8 * np.array(diameters)
#
#     tension_category = to_category(total_tension, tension_bin)
#
#     tensile_category = to_category(tensiles,tensile_bins)
#     diameter_category = to_category(diameters, diameter_bins)
#
#     # print(f'key is {key_name}')
#
#     return tensile_category, diameter_category, tension_category,key_name
#
#
#
# def check_remi_event(file_events,header_events):
#
#     new_file_events = file_events
#     for event in header_events[::-1]:
#         new_file_events = np.insert(new_file_events, 0, event)
#
#     pm = data_convert.remi_2midi(new_file_events.tolist())
#     pm = remove_empty_track(pm)
#     if pm is None or len(pm.instruments) < 1:
#         return None
#
#     if '_' not in new_file_events[1]:
#         tempo = float(new_file_events[1])
#         tempo_category = int(np.where((tempo - vocab.tempo_bins) >= 0)[0][-1])
#         new_file_events[1] = f't_{tempo_category}'
#
#
#     return new_file_events
#
# def remove_continue_event(file_events,header_events,midi_name):
#     bar_pos = np.where(file_events == 'bar')[0]
#     new_file_events = []
#     for idx,event in enumerate(file_events):
#         if event == 'continue' and idx<bar_pos[1]:
#             continue
#         else:
#             new_file_events.append(event)
#
#
#     for event in header_events[::-1]:
#         new_file_events = np.insert(new_file_events, 0, event)
#
#     pm = pretty_midi.PrettyMIDI(midi_name)
#
#     # pm = event_2midi(new_file_events.tolist())[0]
#     pm = remove_empty_track(pm)
#     if pm is None or len(pm.instruments) < 1:
#         return None
#
#     if '_' not in new_file_events[1]:
#         tempo = float(new_file_events[1])
#         tempo_category = int(np.where((tempo - vocab.tempo_bins) >= 0)[0][-1])
#         new_file_events[1] = f't_{tempo_category}'
#
#
#     return new_file_events
#

from model import ScoreTransformer
import vocab
import yaml
import pretty_midi




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


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word





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

#
# def shift_event_keys_with_direction(event):
#     all_shifted_event = []
#     key_idx = int(event[2][2:])
#     this_key = all_key_names[key_idx]
#     # print(f'this key is {this_key}')
#     key_mode = this_key[-5:]
#     # key = this_key[:-6]

    
    

    if key_mode == 'major':
        return all_shifted_event
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
# #
# def cal_separate_file(files,i,augment=True):
#
#     return_list = []
#
#     print(f'file {i} {files[i]}')
#     # file_events = np.array(pickle.load(open('/home/ruiguo/dataset/lmd/lmd_melody_bass_event_new/A/V/M/TRAVMSO12903CF02EE/2077456af444d348c6e4c241710ff187_event', 'rb')))
#     # event, pm = midi_2event('../dataset/0519_first.mid')
#     # file_events = np.array(event)
#     file_events = np.array(pickle.load(open(files[i], 'rb')))
#     folder_name = os.path.dirname(files[i])
#     midi_name = folder_name + '/' + files[i].split('/')[-1].split('_')[0] + '.mid'
#
#     # key_file = '/' + files[i][len(event_folder):-6] + '.mid'
#     # key = keys[key_file]
#     r = re.compile('i_\d')
#     track_program = list(filter(r.match, file_events))
#     num_of_tracks = len(track_program)
#
#
#     # file_events = pickle.load(open('/home/ruiguo/dataset/chinese_event/今生今世_event', 'rb'))
#     # changed_file_events = file_events[1:3+num_of_tracks]
#     # changed_file_events.extend(['bar'])
#     # changed_file_events.extend( file_events[3+num_of_tracks:])
#     # changed_file_events = np.array(changed_file_events)
#     # file_events = np.array(pickle.load(open(files[i], 'rb')))
#
#
#     if num_of_tracks < 1:
#         logger.info(f'omit file {files[i]} with no track')
#         # return None
#
#     header_events = file_events[:2+num_of_tracks]
#
#     # time_signature = file_events[1]
#     # tempo = file_events[2]
#
#
#     bar_pos = np.where(file_events == 'bar')[0]
#
#     bar_beginning_pos = bar_pos[::8]
#
#     # meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0] - 2]
#     # meta_without_track_control = np.concatenate([meta_events[0:3],np.array(track_program)],axis=0)
#     # # < 16 bar
#     for pos in range(len(bar_beginning_pos) - 1):
#         if pos == len(bar_beginning_pos) - 2:
#
#             # detect empty_event(
#             return_events = remove_continue_event(file_events[bar_beginning_pos[pos]:], header_events,midi_name)
#         else:
#             return_events = remove_continue_event(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]],
#                                               header_events,midi_name)
#         if return_events is not None:
#
#             return_list.append(return_events)
#
#             if augment:
#                 # shift keys to all the key in same mode for 2/4, 6/8, 3/4 time
#                 if return_events[0] in ['2/4','3/4','6/8']:
#                     if random.random() > 0.8:
#                         shifted_events = shift_event_keys(return_events)
#                         return_list.extend(shifted_events)
#                         # print(f'not 4/4, shift key to all')
#                 # else:
#                 #
#                 #         # print(f'shift key to all')
#                 #     shifted_events = shift_event_keys_with_direction(return_events)
#                 #
#                 #     return_list.extend(shifted_events)
#         else:
#             print(f'skip file {i} bar pos {pos}')
#
#
#     print(f'number of data of this song is {len(return_list)}')
#     return return_list
#

#
# def cal_remi_file(files,i,augment=True):
#
#     return_list = []
#
#     logger.info(f'file {i} {files[i]}')
#     # file_events = np.array(pickle.load(open('/home/ruiguo/dataset/lmd/lmd_melody_bass_event_new/A/V/M/TRAVMSO12903CF02EE/2077456af444d348c6e4c241710ff187_event', 'rb')))
#     # event, pm = midi_2event('../dataset/0519_first.mid')
#     # file_events = np.array(event)
#     file_events = np.array(pickle.load(open(files[i], 'rb')))
#     # key_file = '/' + files[i][len(event_folder):-6] + '.mid'
#     # key = keys[key_file]
#     r = re.compile('i_\d')
#     track_program = list(filter(r.match, file_events))
#     num_of_tracks = len(track_program)
#
#
#     # file_events = pickle.load(open('/home/ruiguo/dataset/chinese_event/今生今世_event', 'rb'))
#     # changed_file_events = file_events[1:3+num_of_tracks]
#     # changed_file_events.extend(['bar'])
#     # changed_file_events.extend( file_events[3+num_of_tracks:])
#     # changed_file_events = np.array(changed_file_events)
#     # file_events = np.array(pickle.load(open(files[i], 'rb')))
#
#
#     if num_of_tracks < 1:
#         logger.info(f'omit file {files[i]} with no track')
#         # return None
#
#     header_events = file_events[:2+num_of_tracks]
#
#     # time_signature = file_events[1]
#     # tempo = file_events[2]
#
#
#     bar_pos = np.where(file_events == 'bar')[0]
#
#     bar_beginning_pos = bar_pos[::8]
#
#     # meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0] - 2]
#     # meta_without_track_control = np.concatenate([meta_events[0:3],np.array(track_program)],axis=0)
#     # # < 16 bar
#     for pos in range(len(bar_beginning_pos) - 1):
#         if pos == len(bar_beginning_pos) - 2:
#
#             # detect empty_event(
#             return_events = check_remi_event(file_events[bar_beginning_pos[pos]:], header_events)
#         else:
#             return_events = check_remi_event(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]],
#                                               header_events)
#         if return_events is not None:
#
#             return_list.append(return_events)
#
#             if augment:
#                 # shift keys to all the key in same mode for 2/4, 6/8, 3/4 time
#                 if return_events[0] in ['2/4','3/4','6/8']:
#                     if random.random() > 0.8:
#                         shifted_events = shift_event_keys(return_events)
#                         return_list.extend(shifted_events)
#                         # print(f'not 4/4, shift key to all')
#                 # else:
#                 #
#                 #         # print(f'shift key to all')
#                 #     shifted_events = shift_event_keys_with_direction(return_events)
#                 #
#                 #     return_list.extend(shifted_events)
#         else:
#             print(f'skip file {i} bar pos {pos}')
#
#
#     logger.info(f'number of data of this song is {len(return_list)}')
#     return return_list
# #
#
# def add_whole_control_event(file_events,header_events):
#     file_events = np.copy(file_events)
#     num_of_tracks = len(header_events[2:])
#
#     # if file_events[1] not in time_signature_token:
#     #     file_events = np.insert(file_events,1,time_signature)
#     #     file_events = np.insert(file_events, 2, tempo)
#     #     for i, program in enumerate(header_events[2:]):
#     #         file_events = np.insert(file_events, 3+i, program)
#
#
#
#     bar_pos = np.where(file_events == 'bar')[0]
#     pm = event_2midi(file_events.tolist())[0]
#     pm = remove_empty_track(pm)
#     if len(pm.instruments) < 1:
#         return None
#
#     tensiles,diameters,key,_,_ = cal_tension(pm)
#
#     if tensiles is not None:
#         total_bars = min(len(tensiles), len(diameters), len(bar_pos))
#         if total_bars < len(bar_pos):
#             print(f'total bars is {total_bars}. less than original {len(bar_pos)}')
#             bar_pos = bar_pos[:total_bars + 1]
#             file_events = file_events[:bar_pos[-1]]
#             bar_pos = bar_pos[:-1]
#
#         if total_bars < len(tensiles):
#             print(f'total bars is {total_bars}. less than tensile {len(tensiles)}')
#             tensiles = tensiles[:total_bars]
#             diameters = diameters[:total_bars]
#
#
#
#     #     print(f'number of bars is {len(bar_pos)}')
#     #     print(f'time signature is {file_event[1]}')
#     bar_length = int(file_events[0][0])
#
#     if bar_length != 6:
#         bar_length = bar_length * 4 * len(bar_pos)
#     else:
#         bar_length = bar_length / 2 * 4 * len(bar_pos)
#     #     print(f'bar length is {bar_length}')
#
#     track_events = {}
#
#     for i in range(num_of_tracks):
#         track_events[f'track_{i}'] = []
#     track_names = list(track_events.keys())
#     for bar_index in range(len(bar_pos) - 1):
#         bar = bar_pos[bar_index]
#         next_bar = bar_pos[bar_index + 1]
#         bar_events = file_events[bar:next_bar]
#         #         print(bar_events)
#
#         track_pos = []
#
#         for track_name in track_names:
#             track_pos.append(np.where(track_name == bar_events)[0][0])
#         #         print(track_pos)
#         track_index = 0
#         for track_index in range(len(track_names) - 1):
#             track_event = bar_events[track_pos[track_index]:track_pos[track_index + 1]]
#             track_events[track_names[track_index]].append(track_event)
#         #             print(track_event)
#         else:
#             if track_index == 0:
#                 track_event = bar_events[track_pos[track_index]:]
#                 #             print(track_event)
#                 track_events[track_names[track_index]].append(track_event)
#             else:
#                 track_index += 1
#                 track_event = bar_events[track_pos[track_index]:]
#                 #             print(track_event)
#                 track_events[track_names[track_index]].append(track_event)
#
#     densities = note_density(track_events, bar_length)
#     density_category = to_category(densities, control_bins)
#     pm, _ = event_2midi(file_events.tolist())
#     occupation_rate, polyphony_rate = occupation_polyphony_rate(pm)
#     occupation_category = to_category(occupation_rate, control_bins)
#     polyphony_category = to_category(polyphony_rate, control_bins)
#     pitch_register_category = pitch_register(track_events)
#     #     print(densities)
#     #     print(occupation_rate)
#     #     print(polyphony_rate)
#     #     print(density_category)
#     #     print(occupation_category)
#     #     print(polyphony_category)
#
#     #     key_token =  key_to_token[key]
#
#     density_token = [f'd_{category}' for category in density_category]
#     occupation_token = [f'o_{category}' for category in occupation_category]
#     polyphony_token = [f'y_{category}' for category in polyphony_category]
#     pitch_register_token = [f'r_{category}' for category in pitch_register_category]
#
#     track_control_tokens = density_token + occupation_token + polyphony_token + pitch_register_token
#
#     # print(track_control_tokens)
#
#     file_events = file_events.tolist()
#
#
#
#     key = key_to_token[key]
#     file_events.insert(2, key)
#
#
#     for token in track_control_tokens[::-1]:
#         file_events.insert(3, token)
#
#     if '_' not in file_events[1]:
#         tempo = float(file_events[1])
#         tempo_category = int(np.where((tempo - tempo_bins) >= 0)[0][-1])
#         file_events[1] = f't_{tempo_category}'
#
#     if tensiles is not None:
#
#         tension_positions = np.where(np.array(file_events) == 'track_0')[0]
#
#         total_insert = 0
#
#         for i, pos in enumerate(tension_positions):
#             file_events.insert(pos + total_insert, f's_{tensiles[i]}')
#             total_insert += 1
#             file_events.insert(pos + total_insert, f'a_{diameters[i]}')
#             total_insert += 1
#
#     return file_events
#
# def cal_whole_file(files,i,augment=False):
#     return_list = []
#     print(f'file {i} {files[i]}')
#     # file_events = np.array(pickle.load(open('/home/ruiguo/dataset/lmd/lmd_more_event/R/R/T/TRRRTLE12903CA241F/e88a04b4b6e986efac223636a14d63bb_event', 'rb')))
#     file_events = np.array(pickle.load(open(files[i], 'rb')))
#     # key_file = '/' + files[i][len(event_folder):-6] + '.mid'
#     # key = keys[key_file]
#     r = re.compile('i_\d')
#     track_program = list(filter(r.match, file_events))
#     num_of_tracks = len(track_program)
#
#     if num_of_tracks < 1:
#         print(f'omit file {files[i]} with no track')
#         # return None
#
#     header_events = file_events[:2+num_of_tracks]
#
#     return_events = add_whole_control_event(file_events, header_events)
#
#     if return_events is not None:
#         # if key[0] != all_key_names[int(return_events[2][2:])]:
#         #     print(f'whole song key is {key[0]}')
#         #     if key[2] != -1:
#         #         print(f'change key is {key[2]}')
#         #     print(f'16 bar key is {all_key_names[int(return_events[2][2:])]}')
#
#         return return_events
#     else:
#         return None
#
#

#
#
# def cal_separate_event(event):
#
#
#     file_events = np.array(event)
#     r = re.compile('i_\d')
#     track_program = list(filter(r.match, file_events))
#     num_of_tracks = len(track_program)
#     if num_of_tracks < 2:
#         print(f'omit file  with only one track')
#         return None
#
#     time_signature = file_events[1]
#     tempo = file_events[2]
#
#
#     bar_pos = np.where(file_events == 'bar')[0]
#
#     bar_beginning_pos = bar_pos[::8]
#
#     # meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0] - 2]
#     # meta_without_track_control = np.concatenate([meta_events[0:3],np.array(track_program)],axis=0)
#     # # < 16 bar
#     for pos in range(len(bar_beginning_pos) - 1):
#         if pos == len(bar_beginning_pos) - 2:
#
#             # detect empty_event(
#             return_events = add_control_event(file_events[bar_beginning_pos[pos]:], time_signature, tempo,
#                                               track_program)
#         else:
#             return_events = add_control_event(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]],
#                                               time_signature, tempo, track_program)
#         if return_events is not None:
#             return return_events
#         else:
#             print(f'skip file')
#
# def gen_whole_batches(files,event_folder,max_token_length=2200, batch_window_size=8,augment=False):
#     return_events = cal_separate_file(files, 0,  '',augment=augment)
#     pickle.dump(return_events, open(f'/home/ruiguo/dataset/0519_event', 'wb'))
#     #
#     print(f'total files {len(files)}')
#     # for i in range(0, len(files)):
#     #     cal_separate_file(files, i)
#     print(f'augment is {augment}')
#     return_events = Parallel(n_jobs=1)(delayed(cal_separate_file)(files,i,event_folder,augment=augment) for i in range(0,len(files)))
#     batches = []
#     for file_events in return_events:
#         batches.append(file_events)
#
#     return batches
#
#
# def gen_batches(files,max_token_length=2200,augment=False):
#
#
#     logger.info(f'total files {len(files)}')
#     # for i in range(0, len(files)):
#     #     cal_separate_file(files, i)
#     logger.info(f'augment is {augment}')
#
#     # cal_separate_file(files, 0, event_folder, key_model, augment=augment)
#
#     return_events = Parallel(n_jobs=8)(delayed(cal_separate_file)(files,i,augment=augment) for i in range(0,len(files)))
#
#     batches = []
#     for file_events in return_events:
#         if file_events:
#             for event in file_events:
#                 batches.append(event)
#
#     batches.sort(key=len)
#     i = 0
#     while i < len(batches) - 1:
#         if np.array_equal(batches[i],batches[i + 1]):
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
#                 logger.info(
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
#
#
# def validate_event_data(batches):
#     for batch in batches:
#         for events in batch:
#             print(f'{len(np.where(np.array(events) == "bar")[0])}')
#             midi = event_2midi(events)[0]
#             midi.write('./temp.mid')
#             new_events = midi_2event('./temp.mid')[0]
#             print(f'{len(np.where(np.array(new_events) == "bar")[0])}')
#             added_control_event = cal_separate_event(new_events)
#             print(f'{len(np.where(np.array(added_control_event) == "bar")[0])}')
#             # for i,event in enumerate(events):
#             if len(added_control_event) < len(events):
#                 print(f'added event length{len(added_control_event)} is less than { len(events)}')
#                 # else:
#                 #     if event != added_control_event[i]:
#                 #         print('not equal')


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
# files = walk(event_folder)


# whole_batches = gen_whole_batches(['../dataset/0519_first'],'')
# pickle.dump(whole_batches, open(f'/Users/ruiguo/Downloads/score_transformer/contest_0519', 'wb'))




# #
# return_list = []
# for i in range(len(files)):
#     return_list.append(cal_whole_file(files,i))
# print(len(return_list))
# pickle.dump(return_list, open(f'/home/ruiguo/dataset/jay_whole', 'wb'))
#


# event = pickle.load(open('/Users/ruiguo/Documents/mm2021/added_event','rb'))
# cal_separate_event('/Users/ruiguo/Documents/mm2021/')
event_folder = '/home/ruiguo/dataset/lmd/lmd_melody_bass_event_new/'


# # event_folder = '/home/ruiguo/dataset/valid_midi_out'
# event_folder = '/home/ruiguo/dataset/chinese_event/'
# # event_folder = '/home/ruiguo/dataset/lmd/lmd_more_event/'
# event_folder = './lmd_melody_bass_event/'
# # event_folder = '/content/drive/My Drive/score_transformer/lmd_melody_bass_event/'
#
#

#
# print(f'file {j}')
# j=2
# i = 0.6
# print(i)
# start_num = int(len(files) * (i))
# end_num = int(len(files) * (i+0.2))
#
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num], event_folder)
# pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/training_all_batches_{j}', 'wb'))
# pickle.dump(training_batch_length,
#             open(f'/home/data/guorui/score_transformer/sync/training_batch_length_{j}', 'wb'))
#
# del training_all_batches
# gc.collect()
# #
#
# i = 0.8
# print(i)
# start_num = int(len(files) * (i))
# end_num = int(len(files) * (i+0.1))
# print('valid start')
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')
# valid_all_batches, valid_batch_length = gen_batches(files[start_num:end_num], event_folder,key_model=model_prediction,augment=False)
# pickle.dump(valid_all_batches, open(f'/home/data/guorui/score_transformer/sync/new_validation_batches', 'wb'))
# pickle.dump(valid_batch_length,
#             open(f'/home/data/guorui/score_transformer/sync/new_validation_batch_length', 'wb'))
#
# i=0.9
# print(i)
# start_num = int(len(files) * (i))
# end_num = int(len(files) * (i+0.1))
# print('test start')
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')
# test_all_batches, test_batch_length = gen_batches(files[start_num:end_num], event_folder,key_model=model_prediction,augment=False)
# pickle.dump(test_all_batches, open(f'/home/data/guorui/score_transformer/sync/new_test_batches', 'wb'))
# pickle.dump(test_batch_length,
#             open(f'/home/data/guorui/score_transformer/sync/new_validation_test_length', 'wb'))

# del test_batch_length
#
# del valid_all_batches
# gc.collect()
#
#
#
# i=0.9
# print(i)
# start_num = int(len(files) * (i))
# end_num = int(len(files) * (i+0.1))
# print('test start')
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')
# test_all_batches = gen_whole_batches(files[start_num:start_num+600], event_folder)
#
# test_all_batches, test_batch_length = gen_batches(files[start_num:end_num], event_folder,augment=False)
# pickle.dump(test_all_batches, open(f'/home/data/guorui/score_transformer/sync/test_batches_whole', 'wb'))
# pickle.dump(test_batch_length,
#             open(f'/home/data/guorui/score_transformer/sync/test_batch_length_whole', 'wb'))

# del test_batch_length
# gc.collect()
#
# print(f'file {j}')
# i+= 0.15
# print(i)
# start_num = int(len(files) * (i))
# end_num =  int(len(files) * (i+0.05))
#
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num], event_folder)
# pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/validation', 'wb'))
# pickle.dump(training_batch_length,
#             open(f'/home/data/guorui/score_transformer/sync/validation', 'wb'))
#
# training_all_batches,training_batch_length = gen_batches(files[:training_lengths],event_folder)
# pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/training_all_batches_0_0','wb'))
# pickle.dump(training_batch_length, open(f'/home/data/guorui/score_transformer/sync/training_batch_length_0_0','wb'))
# #
#
# training_lengths = int(len(files) * 0.25)
# # #
# training_all_batches,training_batch_length = gen_batches(files[:training_lengths],event_folder)
# pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/training_all_batches_0_0','wb'))
# pickle.dump(training_batch_length, open(f'/home/data/guorui/score_transformer/sync/training_batch_length_0_0','wb'))
# #
#
# all_batches = pickle.load(open('/home/ruiguo/score_transformer/sync/with_melody_batch', 'rb'))
# batch_length = pickle.load(open('/home/ruiguo/score_transformer/sync/with_melody_batch_length', 'rb'))
#
# original_data_length = len(all_batches)
# test_ratio = 0.1
# valid_ratio = 0.1
# train_ratio = 0.8
#
# print(f'train_ratio is {train_ratio}')
# print(f'valid_ratio is {valid_ratio}')
# print(f'test_ratio is {test_ratio}')
#
#
# test_data_index = np.random.choice(len(all_batches), int(original_data_length * test_ratio), replace=False)
#
# test_data_index = np.sort(test_data_index)
#
# test_batches = np.array(all_batches)[test_data_index].tolist()
# test_batch_lengths = {}
#
# for index, item in enumerate(test_batches):
#     if len(item) not in test_batch_lengths:
#         test_batch_lengths[len(item)] = [index]
#     else:
#         test_batch_lengths[len(item)].append(index)
#
# for index in test_data_index[::-1]:
#     del all_batches[index]
#
# valid_data_index = np.random.choice(len(all_batches), int(original_data_length * valid_ratio), replace=False)
# valid_data_index = np.sort(valid_data_index)
#
# valid_batches = np.array(all_batches)[valid_data_index].tolist()
# valid_batch_lengths = {}
#
# for index, item in enumerate(valid_batches):
#     if len(item) not in valid_batch_lengths:
#         valid_batch_lengths[len(item)] = [index]
#     else:
#         valid_batch_lengths[len(item)].append(index)
#
# for index in valid_data_index[::-1]:
#     del all_batches[index]
#
#
# train_batches = all_batches
# train_batch_lengths = {}
#
# for index, item in enumerate(train_batches):
#     if len(item) not in train_batch_lengths:
#         train_batch_lengths[len(item)] = [index]
#     else:
#         train_batch_lengths[len(item)].append(index)


# print(f'train batch length is {len(train_batches)}')
# print(f'valid batch length is {len(valid_batches)}')
# print(f'valid batch length is {len(test_batches)}')
#
# pickle.dump(train_batches, open('chinese_batches','wb'))
# pickle.dump(valid_batches, open('melody_valid_batches','wb'))
# pickle.dump(test_batches, open('melody_test_batches','wb'))
#
# pickle.dump(train_batch_lengths, open('melody_train_batch_lengths','wb'))
# pickle.dump(valid_batch_lengths, open('melody_valid_batch_lengths','wb'))
# pickle.dump(test_batch_lengths, open('melody_test_batch_lengths','wb'))
# sys.exit()
# pickle.dump(test_batch_lengths, open('/home/data/guorui/score_transformer/melody_test_batch_lengths','wb'))




#     pickle.dump(batch_length, open(f'./sync/batch_length_{j}','wb'))
# for i in range(0,45000,15000):
#     j = str(int(i/15000))
#     all_batches,batch_length = gen_batches(files[i:i+15000])
#     pickle.dump(all_batches, open(f'./sync/all_batches_{j}','wb'))
#     pickle.dump(batch_length, open(f'./sync/batch_length_{j}','wb'))
# all_batches,batch_length = gen_batches(files[45000:])
# j = str(int(45000/15000))
# pickle.dump(all_batches, open(f'./sync/all_batches_{j}','wb'))
# pickle.dump(batch_length, open(f'./sync/batch_length_{j}','wb'))
#
#
# sys.exit()

# validate a few event data
#
# all_batches = pickle.load(open('/home/ruiguo/score_transformer/sync/all_batches_3', 'rb'))
# batch_length = pickle.load(open('/home/ruiguo/score_transformer/sync/batch_length_3', 'rb'))
#
# original_data_length = len(all_batches)
# train_4_ratio = 0.5
#
# train_4_data_index = np.random.choice(len(all_batches), int(original_data_length * train_4_ratio), replace=False)
#
# train_4_data_index = np.sort(train_4_data_index)
#
# train_4_batches = np.array(all_batches)[train_4_data_index].tolist()
# train_4_batch_lengths = {}
#
# for index, item in enumerate(train_4_batches):
#     if len(item) not in train_4_batch_lengths:
#         train_4_batch_lengths[len(item)] = [index]
#     else:
#         train_4_batch_lengths[len(item)].append(index)
#
# for index in train_4_data_index[::-1]:
#     del all_batches[index]
#
#
# valid_data_index = np.random.choice(len(all_batches), int(len(all_batches) * 0.5), replace=False)
#
# valid_data_index = np.sort(valid_data_index)
#
# valid_batches = np.array(all_batches)[valid_data_index].tolist()
# valid_batch_lengths = {}
#
#
# for index, item in enumerate(valid_batches):
#     if len(item) not in valid_batch_lengths:
#         valid_batch_lengths[len(item)] = [index]
#     else:
#         valid_batch_lengths[len(item)].append(index)
#
#
# for index in valid_data_index[::-1]:
#     del all_batches[index]
#
# test_batches = np.array(all_batches).tolist()
# test_batch_lengths = {}
#
# for index, item in enumerate(test_batches):
#     if len(item) not in test_batch_lengths:
#         test_batch_lengths[len(item)] = [index]
#     else:
#         test_batch_lengths[len(item)].append(index)
#
#
# print(f'valid batch length is {len(valid_batches)}')
# print(f'test batch length is {len(test_batches)}')
#
#
# pickle.dump(train_4_batches, open('./sync/all_batches_4','wb'))
# pickle.dump(train_4_batch_lengths, open('./sync/batch_length_4','wb'))
# pickle.dump(valid_batches, open('./sync/valid_batches_new','wb'))
# pickle.dump(test_batches, open('./sync/test_batches_new','wb'))
# pickle.dump(valid_batch_lengths, open('./sync/valid_batch_lengths_new','wb'))
# pickle.dump(test_batch_lengths, open('./sync/test_batch_lengths_new','wb'))
# #
# sys.exit()





#
# validate_event_data(all_batches)


# #

#
#
# all_batches = pickle.load(open('/home/ruiguo/score_transformer/sync/all_batches', 'rb'))
# batch_length = pickle.load(open('/home/ruiguo/score_transformer/sync/batch_length', 'rb'))
# #
# original_data_length = len(all_batches)
# test_ratio = 0.1
# valid_ratio = 0.1
# train_ratio = 0.5
# # three separate training data for generating new data in training
# separate_training_data_ratio = 0.1
#
# print(f'train_ratio is {train_ratio}')
# print(f'valid_ratio is {valid_ratio}')
# print(f'test_ratio is {test_ratio}')
#
#
#
# test_data_index = np.random.choice(len(all_batches), int(original_data_length * test_ratio), replace=False)
#
# test_data_index = np.sort(test_data_index)
#
# test_batches = np.array(all_batches)[test_data_index].tolist()
# test_batch_lengths = {}
#
# for index, item in enumerate(test_batches):
#     if len(item) not in test_batch_lengths:
#         test_batch_lengths[len(item)] = [index]
#     else:
#         test_batch_lengths[len(item)].append(index)
#
# for index in test_data_index[::-1]:
#     del all_batches[index]
#
# valid_data_index = np.random.choice(len(all_batches), int(original_data_length * valid_ratio), replace=False)
# valid_data_index = np.sort(valid_data_index)
#
# valid_batches = np.array(all_batches)[valid_data_index].tolist()
# valid_batch_lengths = {}
#
# for index, item in enumerate(valid_batches):
#     if len(item) not in valid_batch_lengths:
#         valid_batch_lengths[len(item)] = [index]
#     else:
#         valid_batch_lengths[len(item)].append(index)
#
# for index in valid_data_index[::-1]:
#     del all_batches[index]
#
# for i in range(3):
#     separate_training_data_index = np.random.choice(len(all_batches), int(original_data_length * separate_training_data_ratio), replace=False)
#     separate_training_data_index = np.sort(separate_training_data_index)
#
#     separate_training_batches = np.array(all_batches)[separate_training_data_index].tolist()
#     separate_training_batch_lengths = {}
#
#     for index, item in enumerate(separate_training_batches):
#         if len(item) not in separate_training_batch_lengths:
#             separate_training_batch_lengths[len(item)] = [index]
#         else:
#             separate_training_batch_lengths[len(item)].append(index)
#     pickle.dump(separate_training_batches, open(f'./sync/separate_training_batches_{i}','wb'))
#     pickle.dump(separate_training_batch_lengths, open(f'./sync/separate_training_batch_lengths_{i}', 'wb'))
#
#     for index in separate_training_data_index[::-1]:
#         del all_batches[index]
#
# train_batches = all_batches
# train_batch_lengths = {}
#
# for index, item in enumerate(train_batches):
#     if len(item) not in train_batch_lengths:
#         train_batch_lengths[len(item)] = [index]
#     else:
#         train_batch_lengths[len(item)].append(index)
#
#
# print(f'train batch length is {len(train_batches)}')
# print(f'valid batch length is {len(valid_batches)}')
# print(f'valid batch length is {len(test_batches)}')
#
# print(f'separate training batch length is {len(separate_training_batches)}')
# pickle.dump(train_batches, open('./sync/train_batches','wb'))
# pickle.dump(valid_batches, open('./sync/valid_batches','wb'))
# pickle.dump(test_batches, open('./sync/test_batches','wb'))
#
# pickle.dump(train_batch_lengths, open('./sync/train_batch_lengths','wb'))
# pickle.dump(valid_batch_lengths, open('./sync/valid_batch_lengths','wb'))
# pickle.dump(test_batch_lengths, open('./sync/test_batch_lengths','wb'))
# sys.exit()

# folder_prefix = '/home/ruiguo/'
# test_batches = pickle.load(open(folder_prefix + 'score_transformer/sync/test_batches_0_0_8_new_bins', 'rb'))
# test_batch_lengths = pickle.load(open(folder_prefix + 'score_transformer/sync/test_batch_lengths_0_0_8_new_bins', 'rb'))
#
# #
# span_ratio_separately_each_epoch = np.array([[1, 0, 0], [.5, .5, 0],
#                                              [.25, .75, 0], [.25, .5, .25],
#                                              [.25, .25, .5]])
#
#
# test_dataset = ParallelLanguageDataset('', '',
#                                            vocab, 0,
#                                            0,
#                                            2200,
#                                            16,
#                                            test_batches,
#                                            test_batch_lengths,
#                                            .15,
#                                            .3,
#                                            .3,
#                                            .3,
#                                            0,
#                                            .3,
#                                            3,
#                                            0.5,
#                                            span_ratio_separately_each_epoch,
#                                            True)
# data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=lambda batch: collate_mlm(batch))  # 训练语料按长度排好序的
#
# for i in data_loader:
#     print(i)
