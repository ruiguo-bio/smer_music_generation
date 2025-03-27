import numpy as np
import encode
from einops import rearrange
import torch
import re


from tqdm.autonotebook import tqdm


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


def sampling(logit, vocab, p=None, t=1.0, no_pitch=False, no_duration=False, no_rest=False, no_whole_duration=False,
             no_eos=False, no_continue=False, no_sep=False, is_density=False, is_polyphony=False, is_occupation=False,
             is_tensile=False, no_control=False):
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

    logit = np.array([
                         -100 if i in vocab.program_indices + vocab.structure_indices + vocab.time_signature_indices + vocab.tempo_indices else
                         logit[i] for i in range(vocab.vocab_size)])
    if no_control:
        logit = np.array([-100 if i in vocab.control_indices.values() else
                          logit[i] for i in range(vocab.vocab_size)])

    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


def sampling_rest_single(logit, vocab, p=None, t=1.0, no_pitch=False, no_duration=False, no_rest=False, no_eos=False):
    logit = logit.squeeze().cpu().numpy()
    if no_pitch:
        logit = np.array(
            [-100 if i in encode.pitch_indices else logit[i] for i in range(encode.vocab_size)])

    if no_duration:
        logit = np.array([-100 if i in encode.duration_only_indices else logit[i] for i in
                          range(encode.vocab_size)])

    if no_rest:
        logit = np.array(
            [-100 if i in encode.rest_indices else logit[i] for i in range(encode.vocab_size)])

    if no_eos:
        logit = np.array(
            [-100 if i == encode.eos_index else logit[i] for i in range(encode.vocab_size)])

    logit = np.array([
                         -100 if i in encode.program_indices + encode.structure_indices + encode.time_signature_indices + encode.tempo_indices else
                         logit[i] for i in range(
            encode.vocab_size)])

    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


def sampling_step_single(logit, vocab, p=None, t=1.0, no_pitch=False, no_duration=False, no_step=False):
    logit = logit.squeeze().cpu().numpy()
    if no_pitch:
        logit = np.array(
            [-100 if i in encode.pitch_indices else logit[i] for i in range(encode.vocab_size)])

    if no_duration:
        logit = np.array([-100 if i in encode.duration_only_indices else logit[i] for i in
                          range(encode.vocab_size)])

    if no_step:
        logit = np.array(
            [-100 if i in encode.step_indices else logit[i] for i in range(encode.vocab_size)])

    logit = np.array([
                         -100 if i in encode.program_indices + encode.structure_indices + encode.time_signature_indices + encode.tempo_indices else
                         logit[i] for i in range(
            encode.vocab_size)])

    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


def sampling_step_multi(logit, vocab, p=None, t=1.0, no_pitch=False, no_duration=False, no_step=False, no_eos=False,
                        no_continue=False):
    logit = logit.squeeze().cpu().numpy()
    if no_pitch:
        logit = np.array(
            [-100 if i in encode.pitch_indices else logit[i] for i in range(encode.vocab_size)])

    if no_duration:
        logit = np.array([-100 if i in encode.duration_only_indices else logit[i] for i in
                          range(encode.vocab_size)])

    if no_step:
        logit = np.array(
            [-100 if i in encode.step_indices else logit[i] for i in range(encode.vocab_size)])
    if no_eos:
        logit = np.array(
            [-100 if i == encode.eos_index else logit[i] for i in range(encode.vocab_size)])
    if no_continue:
        logit = np.array(
            [-100 if i == encode.continue_index else logit[i] for i in range(encode.vocab_size)])

    logit = np.array([
                         -100 if i in encode.program_indices + encode.structure_indices + encode.time_signature_indices + encode.tempo_indices else
                         logit[i] for i in range(
            encode.vocab_size)])

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


def model_generate(model, src, tgt, device, return_weights=False):
    try:

        src = src.clone().detach().unsqueeze(0).long().to(device)
        tgt = torch.tensor(tgt).unsqueeze(0).to(device)
        tgt_mask = gen_nopeek_mask(tgt.shape[1])
        tgt_mask = tgt_mask.clone().detach().unsqueeze(0).to(device)

        output, weights = model.forward(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None,
                                        memory_key_padding_mask=None,
                                        tgt_mask=tgt_mask)
        if return_weights:
            return output.squeeze(0).to('cpu'), weights.squeeze(0).to('cpu')
        else:
            return output.squeeze(0).to('cpu')
    except Exception as e:
        raise(e)




def fill_empty_bars(events, generate_bar_number, bar_duration, duration_time_to_name, duration_times):
    bar_duration_list = encode.time2durations(bar_duration, duration_time_to_name, duration_times)

    r = re.compile('track_\d')
    track_names = list(set(filter(r.match, events)))
    track_names.sort()
    track_nums = len(track_names)
    for bar_number in range(generate_bar_number):
        events.append('bar')
        events.append('s_2')
        events.append('a_0')
        for track_num in range(track_nums):
            events.append(f'track_{track_num}')
            events.append('rest_e')
            events.extend(bar_duration_list)
    return events


def mask_bar_and_track(event, vocab, mask_tracks, mask_bars):
    total_track_control_types = 3
    tokens = []

    decoder_target = []
    masked_indices_pairs = []
    mask_bar_names = []
    mask_track_names = []
    r = re.compile('track_\d')

    track_names = list(set(filter(r.match, event)))
    track_names.sort()

    bar_poses = np.where(np.array(event) == 'bar')[0]

    r = re.compile('track_\d')
    track_names = list(set(filter(r.match, event)))
    track_names.sort()
    track_nums = len(track_names)

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
    for bar_num in mask_bars:
        tracks_in_a_bar = bar_with_track_poses[bar_num]
        for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
            if track_pos in mask_tracks:
                track_start, track_end = track_star_end_poses

                mask_bar_names.append(bar_num)
                mask_track_names.append(track_pos)

                token_start = track_start + total_track_control_types

                if event[track_end - 1] in vocab.name_to_tokens['tensile']:
                    tensile_end = 1
                else:
                    tensile_end = 0

                token_end = track_end - total_track_control_types - tensile_end

                masked_indices_pairs.append((token_start, token_end))

                for i in range(total_track_control_types + tensile_end):
                    masked_indices_pairs.append((token_end + i, token_end + 1 + i))

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

    return tokens, mask_track_names, mask_bar_names


def check_track_total_time(events, encode, duration_name_to_time, duration_time_to_name, duration_times,
                           bar_duration):
    current_time = 0
    in_duration = False
    duration_list = []
    previous_time = 0
    in_rest_s = False
    new_events = []

    if len(events) == 2:
        last_total_time_adjusted = encode.time2durations(bar_duration, duration_time_to_name, duration_times)
        for token in last_total_time_adjusted[::-1]:
            events.insert(-1, token)
        events.insert(-1, 'rest_e')
        return False, events

    for event in events:
        new_events.append(event)

        if in_duration and event not in encode.duration_multi:
            total_time = encode.total_duration(duration_list, duration_name_to_time)
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

        if event in encode.duration_multi:
            in_duration = True
            duration_list.append(event)

        if event == 'rest_s':
            in_rest_s = True

    else:
        if duration_list:
            total_time = encode.total_duration(duration_list, duration_name_to_time)
            if in_rest_s:
                current_time = previous_time + total_time

            else:

                current_time = current_time + total_time

    while new_events[-1] not in encode.duration_multi:
        new_events.pop()
    if current_time == bar_duration:
        return True, new_events
    else:
        if current_time > bar_duration:
            difference = current_time - bar_duration
            last_total_time_adjusted = total_time - difference

        else:
            difference = bar_duration - current_time
            last_total_time_adjusted = total_time + difference

        last_duration_list = encode.time2durations(last_total_time_adjusted, duration_time_to_name,
                                                             duration_times)
        for _ in range(len(duration_list)):
            new_events.pop()

        new_events.extend(last_duration_list)

        return False, new_events


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


def generation_all(model, events, device, vocab, logger, all_controls, tracks_to_generate, bars_to_generate):
    try:
        if int(events[0][2]) == 8:
            duration_name_to_time, duration_time_to_name, duration_times, bar_duration = encode.get_note_duration_dict(
                1.5, (int(events[0][0]), int(events[0][2])))
        else:
            duration_name_to_time, duration_time_to_name, duration_times, bar_duration = encode.get_note_duration_dict(
                1, (int(events[0][0]), int(events[0][2])))

        bar_poses = np.where(np.array(events) == 'bar')[0]

        bar_nums = len(bar_poses)

        r = re.compile('track_\d')
        track_names = list(set(filter(r.match, events)))
        track_names.sort()

        mask_target = []
        tracks_to_generate = [track_names.index(f'track_{track}') for track in tracks_to_generate]

        for _ in bars_to_generate:
            for track in tracks_to_generate:
                mask_target.extend(['r', 'd', 'o', 'p'])
                if track == len(track_names) - 1:
                    mask_target.append('t')

        if bars_to_generate[-1] >= bar_nums:
            events = fill_empty_bars(events, bars_to_generate[-1] - bar_nums + 1, bar_duration, duration_time_to_name,
                                     duration_times)


        result = mask_bar_and_track(events, vocab, tracks_to_generate, bars_to_generate)
        if result is None:
            return result
        src, mask_track_names, mask_bar_names = result

        if int(events[0][0]) >= 4 and int(events[0][2]) == 4:
            no_whole_duration = False
        else:
            no_whole_duration = True

        if int(events[0][2]) == 8:
            duration_name_to_time, duration_time_to_name, duration_times, bar_duration = encode.get_note_duration_dict(
                1.5, (int(events[0][0]), int(events[0][2])))
        else:
            duration_name_to_time, duration_time_to_name, duration_times, bar_duration = encode.get_note_duration_dict(
                1, (int(events[0][0]), int(events[0][2])))

        src_masked_nums = np.sum(src == vocab.char2index('m_0'))
        tgt_inp = []
        total_generated_events = []

        if src_masked_nums == 0:
            return None

        with torch.no_grad():


            this_track_tokens = []

            for mask_idx in tqdm(range(src_masked_nums),desc='infilling blocks'):
            # while mask_idx < src_masked_nums:

                # print(f'generating {mask_idx + 1}/{src_masked_nums}')
                this_tgt_inp = []
                this_tgt_inp.append(vocab.char2index('m_0'))
                this_generated_events = []
                this_generated_events.append('m_0')
                total_grammar_correct_times = 0

                in_pitch = False
                in_rest = False
                in_sep = False
                in_continue = False
                while this_tgt_inp[-1] != vocab.char2index('<eos>') and len(this_tgt_inp) < 100:

                    output, weight = model_generate(model, torch.tensor(src), tgt_inp + this_tgt_inp, device,
                                                    return_weights=True)

                    if in_sep:

                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_eos=True,
                                         no_whole_duration=True, no_control=True)
                        while index in vocab.rest_indices or index == vocab.eos_index or index == \
                                vocab.duration_only_indices[0]:
                            index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_eos=True,
                                             no_whole_duration=True, no_control=True)

                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info("in sep failed")
                                break

                        event = vocab.index2char(index)

                    elif in_continue:
                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_duration=True,
                                         no_continue=True, no_eos=True, no_control=True)
                        while index not in vocab.pitch_indices:
                            index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_duration=True,
                                             no_continue=True,
                                             no_eos=True, no_control=True)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info('in continue failed')
                                break

                        event = vocab.index2char(index)

                    elif in_pitch:
                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_continue=True,
                                         no_whole_duration=no_whole_duration, no_eos=True, no_control=True)
                        while index not in vocab.duration_only_indices and index not in vocab.pitch_indices:
                            index = sampling(output[-1], vocab, no_rest=True, no_sep=True, no_continue=True,
                                             no_whole_duration=no_whole_duration, no_eos=True, no_control=True)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info('in pitch failed')
                                break
                        event = vocab.index2char(index)

                    elif in_rest:
                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_pitch=True, no_rest=True, no_sep=True, no_continue=True,
                                         no_whole_duration=no_whole_duration, no_eos=True, no_control=True)
                        while index not in vocab.duration_only_indices:
                            index = sampling(output[-1], vocab, no_pitch=True, no_rest=True, no_sep=True,
                                             no_continue=True,
                                             no_whole_duration=no_whole_duration, no_eos=True, no_control=True)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                logger.info('in rest failed')
                                break
                        event = vocab.index2char(index)


                    elif len(this_tgt_inp) == 1:
                        if mask_target[mask_idx] != 'r':

                            this_target_control = mask_target[mask_idx]
                            # print(this_target_control)
                            if this_target_control == 'd':

                                index = sampling(output[-1], vocab, is_density=True)
                            elif this_target_control == 'o':
                                index = sampling(output[-1], vocab, is_occupation=True)

                            elif this_target_control == 'p':
                                index = sampling(output[-1], vocab, is_polyphony=True)

                            else:

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

                        event = vocab.index2char(index)

                    else:
                        # free state
                        index = sampling(output[-1], vocab, no_whole_duration=no_whole_duration, no_control=True)

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

                    if index in all_controls:

                        this_tgt_inp.append(index)
                        this_generated_events.append(event)
                        this_tgt_inp.append(vocab.char2index('<eos>'))
                        this_generated_events.append('<eos>')
                    else:
                        this_track_tokens.append(vocab.index2char(index))

                        this_tgt_inp.append(index)
                        this_generated_events.append(event)

                # mask_idx += 1
                tgt_inp.extend(this_tgt_inp[:-1])
                total_generated_events.extend(this_generated_events[:-1])

        src_token = []

        for i, token_idx in enumerate(src):
            src_token.append(vocab.index2char(token_idx.item()))

        return restore_marked_input(src_token, total_generated_events), mask_track_names, mask_bar_names
    except Exception as e:
        print(e)

def change_controls(original_event, controls):
    r = re.compile('i_\d')
    programs = list(filter(r.match, original_event))
    r = re.compile('track_\d')
    track_names = list(set(filter(r.match, original_event)))
    track_names.sort()
    track_nums = len(track_names)
    bar_poses = np.where(np.array(original_event) == 'bar')[0]

    # original_event[0] = controls['time_signature']
    # original_event[2] = vocab_control.key_to_token[controls['key']]
    density_poses = [-1 for i in range(track_nums)]
    polyphony_poses = [-1 for i in range(track_nums)]
    occupation_poses = [-1 for i in range(track_nums)]
    program_poses = [-1 for i in range(track_nums)]

    r = re.compile('d_\d')
    densities = list(filter(r.match, original_event[:bar_poses[0]]))
    r = re.compile('y_\d')
    polyphonies = list(filter(r.match, original_event[:bar_poses[0]]))
    r = re.compile('o_\d')
    occupations = list(filter(r.match, original_event[:bar_poses[0]]))

    for t_num in range(track_nums):
        control_name = f'track_{track_names[t_num][-1]}_c'
        if t_num == 0:
            density_poses[t_num] = np.where(densities[t_num] == np.array(original_event))[0][0]
            occupation_poses[t_num] = np.where(occupations[t_num] == np.array(original_event))[0][0]
            polyphony_poses[t_num] = np.where(polyphonies[t_num] == np.array(original_event))[0][0]
            program_poses[t_num] = np.where(programs[t_num] == np.array(original_event))[0][0]

            original_event[density_poses[t_num]] = f'd_{controls[control_name]["density"]}'
            original_event[polyphony_poses[t_num]] = f'y_{controls[control_name]["polyphony"]}'
            original_event[occupation_poses[t_num]] = f'o_{controls[control_name]["occupation"]}'

        else:

            density_poses[t_num] = \
                np.where(densities[t_num] == np.array(original_event[density_poses[t_num - 1] + 1:]))[0][0] + \
                density_poses[
                    t_num - 1] + 1
            occupation_poses[t_num] = \
                np.where(occupations[t_num] == np.array(original_event[occupation_poses[t_num - 1] + 1:]))[0][0] + \
                occupation_poses[t_num - 1] + 1
            polyphony_poses[t_num] = \
                np.where(polyphonies[t_num] == np.array(original_event[polyphony_poses[t_num - 1] + 1:]))[0][0] + \
                polyphony_poses[t_num - 1] + 1
            program_poses[t_num] = \
                np.where(programs[t_num] == np.array(original_event[program_poses[t_num - 1] + 1:]))[0][0] + \
                program_poses[
                    t_num - 1] + 1

            original_event[density_poses[t_num]] = f'd_{controls[control_name]["density"]}'
            original_event[polyphony_poses[t_num]] = f'y_{controls[control_name]["polyphony"]}'
            original_event[occupation_poses[t_num]] = f'o_{controls[control_name]["occupation"]}'

    # tensile_poses = []
    # for i, token in enumerate(original_event):
    #     if token[:2] == 's_':
    #         tensile_poses.append(i)
    #
    # for pos_idx, tensile_pos in enumerate(tensile_poses):
    #     if pos_idx < len(controls['tensile']):
    #         original_event[tensile_pos] = f's_{controls["tensile"][pos_idx]}'

    r = re.compile('track_\d')
    track_names = list(set(filter(r.match, original_event)))
    track_names.sort()

    track_poses = []
    for track_name in track_names:
        track_pos = np.where(track_name == np.array(original_event))[0]
        track_poses.extend(track_pos)
    track_poses.extend(bar_poses)

    all_track_pos = list(np.sort(track_poses))
    all_track_pos.append(len(original_event))

    bar_with_track_poses = []

    names = ['melody', 'bass']
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
    # use bar track control
    if controls['bar_track'] == 0:
        for bar_num in range(len(bar_poses)):
            # print(bar_num)
            tracks_in_a_bar = bar_with_track_poses[bar_num]
            for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
                track_start, _ = track_star_end_poses

                if controls['bar_density'][track_names[track_pos]][bar_num] == 10:
                    original_event[track_start] = 'unk'
                else:
                    original_event[track_start] = f"d_{controls['bar_density'][track_names[track_pos]][bar_num]}"

                if controls['bar_occupation'][track_names[track_pos]][bar_num] == 10:

                    original_event[track_start + 1] = 'unk'
                else:
                    original_event[track_start + 1] = f"o_{controls['bar_occupation'][track_names[track_pos]][bar_num]}"

                if controls['bar_polyphony'][track_names[track_pos]][bar_num] == 10:
                    original_event[track_start + 2] = 'unk'
                else:
                    original_event[track_start + 2] = f"y_{controls['bar_polyphony'][track_names[track_pos]][bar_num]}"
    else:
        for bar_num in range(len(bar_poses)):
            if bar_num >= controls['s_bar'] and bar_num <= controls['e_bar']:
                tracks_in_a_bar = bar_with_track_poses[bar_num]
                for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
                    name = f'{track_names[track_pos]}'
                    if controls[name] == 0:
                        track_start, _ = track_star_end_poses

                        original_event[track_start] = 'unk'
                        original_event[track_start + 1] = 'unk'
                        original_event[track_start + 2] = 'unk'
                    # if track_pos == 1 and controls['bass'] == 0:
                    #     track_start, _ = track_star_end_poses
                    #
                    #     original_event[track_start] = 'unk'
                    #     original_event[track_start+1] = 'unk'
                    #     original_event[track_start+2] = 'unk'
                    #
                    # if track_pos == 2 and controls['harmony'] == 0:
                    #     track_start, _ = track_star_end_poses
                    #
                    #     original_event[track_start] = 'unk'
                    #     original_event[track_start+1] = 'unk'
                    #     original_event[track_start+2] = 'unk'

    total_track_control_types = 3
    tension_control = True

    # copy the bar track/tension token to end

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
                bar_control = original_event[bar_pos + 1]
                original_event.insert(next_bar_pos, bar_control)

            if total_track_control_types > 0:
                for track_num in range(track_nums):
                    track_start = all_track_pos[back_pos + track_num + 1] + (
                        total_track_control_types) * track_num
                    insert_pos = all_track_pos[back_pos + track_num + 2] + (
                        total_track_control_types) * track_num
                    track_controls = original_event[
                                     track_start + 1:track_start + total_track_control_types + 1]

                    for track_control in track_controls[::-1]:
                        original_event.insert(insert_pos, track_control)

    return original_event

#
#
#
# def check_controls(original_controls, gen_track_controls, gen_bar_controls, match_tracks=None, match_bars=None, match_keys=None,
#                    match_time_signature=None):
#     ori_track_controls = []
#     ori_bar_controls = []
#
#     # extract original control
#     r = re.compile('i_\d')
#     programs = list(filter(r.match, original_event))
#     track_nums = len(programs)
#
#     density_poses = [-1 for i in range(track_nums)]
#     polyphony_poses = [-1 for i in range(track_nums)]
#     occupation_poses = [-1 for i in range(track_nums)]
#     program_poses = [-1 for i in range(track_nums)]
#
#     r = re.compile('d_\d')
#     densities = list(filter(r.match, original_event))
#     r = re.compile('y_\d')
#     polyphonies = list(filter(r.match, original_event))
#     r = re.compile('o_\d')
#     occupations = list(filter(r.match, original_event))
#
#     result_diffs = [0]
#
#     for t_num in range(track_nums):
#         if t_num == 0:
#             density_poses[t_num] = np.where(densities[t_num] == np.array(original_event))[0][0]
#             occupation_poses[t_num] = np.where(occupations[t_num] == np.array(original_event))[0][0]
#             polyphony_poses[t_num] = np.where(polyphonies[t_num] == np.array(original_event))[0][0]
#             program_poses[t_num] = np.where(programs[t_num] == np.array(original_event))[0][0]
#
#         else:
#             density_poses[t_num] = \
#             np.where(densities[t_num] == np.array(original_event[density_poses[t_num - 1] + 1:]))[0][0] + density_poses[
#                 t_num - 1] + 1
#             occupation_poses[t_num] = \
#             np.where(occupations[t_num] == np.array(original_event[occupation_poses[t_num - 1] + 1:]))[0][0] + \
#             occupation_poses[t_num - 1] + 1
#             polyphony_poses[t_num] = \
#             np.where(polyphonies[t_num] == np.array(original_event[polyphony_poses[t_num - 1] + 1:]))[0][0] + \
#             polyphony_poses[t_num - 1] + 1
#             program_poses[t_num] = \
#             np.where(programs[t_num] == np.array(original_event[program_poses[t_num - 1] + 1:]))[0][0] + program_poses[
#                 t_num - 1] + 1
#     tensile_poses = []
#     for i, token in enumerate(original_event):
#         if token[:2] == 's_':
#             tensile_poses.append(i)
#
#     diameter_poses = []
#     for i, token in enumerate(original_event):
#         if token[:2] == 'a_':
#             diameter_poses.append(i)
#     gen_track_controls_filtered = []
#     if match_tracks:
#         ori_track_controls = []
#         for i in match_tracks:
#             ori_track_controls.append((int(densities[i][2:]), int(occupations[i][2:]), int(polyphonies[i][2:])))
#             gen_track_controls_filtered.append(generated_track_controls[i])
#     # gen_bar_controls_filtered = []
#     if match_bars:
#         ori_bar_controls = []
#         for i in range(len(generated_bar_controls)):
#             ori_bar_controls.append(
#                 (int(original_event[tensile_poses[i]][2:]), int(original_event[diameter_poses[i]][2:])))
#             # gen_bar_controls_filtered.append(generated_bar_controls[i])
#         # print(ori_bar_controls)
#         # print(generated_bar_controls)
#     regenerated_tracks = []
#     if match_tracks:
#         for i in range(len(ori_track_controls)):
#             ori_track_control = ori_track_controls[i]
#             gen_track_control = gen_track_controls_filtered[i]
#             print(
#                 f'track {match_tracks[i]}, original density/occupation/polyphony {ori_track_control}, generated {gen_track_control} ')
#
#             for j, item in enumerate(ori_track_control):
#                 if abs(int(ori_track_control[j]) - int(gen_track_control[j])) > track_thresh:
#                     regenerated_tracks.append(match_tracks[i])
#                     result_diffs.append(abs(int(ori_track_control[j]) - int(gen_track_control[j])))
#
#     regenerated_bars = []
#     if match_bars:
#         for i in match_bars:
#
#             ori_bar_control = ori_bar_controls[i]
#             gen_bar_control = gen_bar_controls[i]
#             print(
#                 f'bar {i}, original original tensile strain/cloud diameter {ori_bar_control}, generated {gen_bar_control} ')
#
#             for j, item in enumerate(ori_bar_control):
#                 if abs(int(ori_bar_control[j]) - int(gen_bar_control[j])) > bar_thresh:
#                     regenerated_bars.append(i)
#                     result_diffs.append(abs(int(ori_bar_control[j]) - int(gen_bar_control[j])))
#
#     if match_tracks:
#         return list(set(regenerated_tracks)), np.sum(result_diffs)
#     if match_bars:
#         ori_tensiles = []
#         ori_diameters = []
#         gen_tensiles = []
#         gen_diameters = []
#         # print(gen_bar_controls)
#         for i in range(len(tensile_poses)):
#             ori_tensiles.append(int(original_event[tensile_poses[i]][2:]))
#             ori_diameters.append(int(original_event[diameter_poses[i]][2:]))
#             gen_tensiles.append(gen_bar_controls[i][0])
#             gen_diameters.append(gen_bar_controls[i][1])
#
#         if len(regenerated_bars) == 0:
#             draw_tension_match(ori_tensiles, generated_tensile, ori_diameters, generated_diameter)
#         return list(set(regenerated_bars)), np.sum(result_diffs)