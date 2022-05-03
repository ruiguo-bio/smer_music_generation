import numpy as np
import pickle
from einops import rearrange
import argparse
import os
import logging
import log
import coloredlogs
import re
from torch.optim import Adam
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader

import vocab

from dataset import ParallelLanguageDataset
from model import ScoreTransformer

from dataset import collate_mlm_pretraining, collate_mlm_finetuning

import wandb

wandb.login()


def get_args(default='.'):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--platform', default='local', type=str,
                        help="platform to run the code")

    parser.add_argument('-e', '--num_epochs', default=10, type=int,
                        help="number of epoch")
    parser.add_argument('-d', '--is_debug', default=False, type=bool,
                        help="debug or not")
    parser.add_argument('-v', '--device', default='', type=str,
                        help="device")

    parser.add_argument('-m', '--mode', default=0, type=int,
                        help="0:rest multi, 1:step single")

    parser.add_argument('-c', '--checkpoint_dir', default="", type=str,
                        help="checkpoint dir")
    parser.add_argument('-r', '--learning_rate', default="0.0001", type=float,
                        help="learning rate")

    parser.add_argument('-i', '--run_id', default=None, type=str,
                        help="run id")
    parser.add_argument('-a', '--reset_epoch', default=False, type=bool,
                        help="if to reset epoch to 0 after loading the checkpoint")

    parser.add_argument('-n', '--fine_tuning', default=False, type=bool,
                        help="fine tuning for generation or pretraining")

    parser.add_argument('-l', '--encoder_layers', default=4, type=int,
                        help="number of encoder layers")

    parser.add_argument('-t', '--control_number', default=0, type=int,
                        help="control number")
    parser.add_argument('-w', '--control_mode', default=0, type=int,
                        help="control mode")

    parser.add_argument('-x', '--test_data', default=False, type=bool,
                        help="test mode")

    return parser.parse_args()


def phi(rt, ri, distance='medium'):
    rt = rt.type(torch.float32)
    if distance == 'small':
        return torch.abs(rt - ri)
    elif distance == 'large':
        return 2 * torch.square(rt - ri)
    else:
        return torch.square(rt - ri)


def soft_label(total_label_num, target_index_range, distance):
    target_index_length = target_index_range[1] - target_index_range[0] + 1
    output_weights = torch.zeros(total_label_num, total_label_num)
    weights = nn.functional.softmax(-phi(torch.unsqueeze(torch.arange(target_index_length), dim=1),
                                         torch.arange(target_index_length), distance), 0)
    output_weights[target_index_range[0]:target_index_range[1] + 1,
    target_index_range[0]:target_index_range[1] + 1] = weights
    return output_weights


class OrdinalLoss(nn.Module):
    def __init__(self, target_index_range, vocab_size, distance, device, **kwargs):
        super().__init__()
        # a tuple to delineate the target index
        self.weights = soft_label(vocab_size, target_index_range, distance).to(device)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, target):
        logsoftmax = self.logsoftmax(x)
        target_array = -self.weights[target]
        return torch.mean(torch.sum(torch.multiply(logsoftmax, target_array), axis=1))


def main(**kwargs):
    # device = torch.device(kwargs['device'])
    # print(f'device is {device}')
    # args = get_args()

    # event_folder = args.event_folder
    # tension_folder = args.tension_folder
    # train_ratio = args.train_ratio
    # valid_ratio = args.valid_ratio
    # test_ratio = args.test_ratio
    # num_epochs = args.num_epochs
    # platform = args.platform
    # is_debug = args.is_debug
    # checkpoint_dir = args.checkpoint_dir
    # lr = args.learning_rate
    # run_id = args.run_id
    # fine_tuning = args.fine_tuning
    # reset_epoch = args.reset_epoch
    # control_list = args.control_list

    # control_list = control_list.split()

    logger.info(f'num_epochs is {num_epochs}')
    logger.info(f'is_debug is {is_debug}')

    logger.info(f'platform is {platform}')
    logger.info(f'the control list is {control_list}')
    logger.info(f'the control mode is {control_mode}')
    logger.info(f'test mode is {is_test}')


    this_vocab = vocab.WordVocab(vocab_mode, control_list)


    if run_id:
        logger.info(f'run_id is {run_id}')
    if args.reset_epoch:
        logger.info(f'reset epoch to 0')

    logger.info(f'learning rate is {lr}')

    if checkpoint_dir:
        print(f'checkpoint dir is {checkpoint_dir}')

    # print(f'max_token_length is {max_token_length}')
    # print(f'train jointly is {train_jointly}')

    config = {"batch_size": 2,
              "eos_weight": 0.8,
              'd_model': 512,
              'lr': lr,
              'num_encoder_layers': encoder_layers,
              'epochs': num_epochs,
              # 'num_decoder_layers': tune.grid_search([4]),
              # 'dim_feedforward': tune.grid_search([2048]),
              'nhead': 8,

              }

    run(config, this_vocab, vocab_mode, fine_tuning, reset_epoch, control_mode, checkpoint_dir=checkpoint_dir,is_test=is_test,device=device)


def logging_config(output_folder, append=False):
    logger = logging.getLogger(__name__)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    logger.handlers = []
    logfile = output_folder + '/logging.log'
    print(f'log file is {logfile}')

    if append is True:
        filemode = 'a'
    else:
        filemode = 'w'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S', filename=logfile, filemode=filemode)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)

    coloredlogs.install(level='INFO', logger=logger, isatty=True)
    logger.info('create a logger file')
    return logger


def run(config, vocab, vocab_mode, fine_tuning, reset_epoch, control_mode=0, checkpoint_dir=None, run_id=None,is_test=False,device=''):
    if is_debug:
        mode = 'offline'
    else:
        mode = 'online'

    if run_id:
        resume = 'allow'
    else:
        resume = None
    if platform == 'local':
        wandb_dir = '/home/data/guorui/'
    else:
        wandb_dir = './'
        mode = 'offline'

    print(f'wandb mode is {mode}')
    if control_mode == 0:
        tags = ['track']
    else:
        tags = ['bar']
    if is_test:
        tags.append('test')
    with wandb.init(project="smer_transformer", mode=mode, dir=wandb_dir, tags=tags, config=config,
                    resume=resume, id=run_id):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        current_folder = wandb.run.dir
        logfile = current_folder + '/logging.log'
        if resume:
            filemode = 'a'
        else:
            filemode = 'w'
        logger.handlers = []
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                            datefmt='%Y-%m-%d %H:%M:%S', filename=logfile, filemode=filemode)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logger.addHandler(console)

        coloredlogs.install(level='INFO', logger=logger, isatty=False)

        for key in config.keys():
            logger.info(f'{key} is {config[key]}')

        if device == "":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



        logger.info(f'bar num is {16}')
        logger.info(f'output folder is {current_folder}')
        logger.info(f'vocab size is {vocab.vocab_size}')
        logger.info(f'platform is {platform}')
        logger.info(f'config is {config}')

        model = ScoreTransformer(vocab.vocab_size, config['d_model'], config['nhead'], config['num_encoder_layers'],
                                 config['num_encoder_layers'], 2048, 2400,
                                 0.1, 0.1)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        optim = Adam(model.parameters(), lr=config['lr'])

        if checkpoint_dir:
            logger.info(f'load check point {checkpoint_dir}')

            model_dict = torch.load(checkpoint_dir)

            model_state = model_dict['model_state_dict']
            optimizer_state = model_dict['optimizer_state_dict']
            if reset_epoch:
                start_epoch = 0
            else:
                start_epoch = model_dict['epoch'] + 1
            logger.info(f'from epoch {start_epoch}')

            # if config['lr_reset']:
            #     optim.param_groups[0]["lr"] = 0.0001
            #     print(f'reset lr to {optim.param_groups[0]["lr"]}')

            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in model_state.items():
            #     name = k[7:] # remove `module.`
            #     new_state_dict[name] = v

            # new_state_dict = model_state

            model.load_state_dict(model_state)

        else:
            start_epoch = 0

        # if torch.cuda.device_count() > 1:
        #     logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        #     model = nn.DataParallel(model)

        model.to(device)
        if checkpoint_dir:
            optim.load_state_dict(optimizer_state)
            print(f'optim loaded lr is {optim.param_groups[0]["lr"]}')

        if platform == 'local':
            folder_prefix = '/home/data/guorui/dataset/lmd/batches/'
        else:
            folder_prefix = '../dataset/batches/'

        if is_debug:
            if vocab_mode == 0:
                if control_mode == 0:
                    # track control only
                    train_batch_name = 'track_dataset/smer_mock_batch'
                    train_length_name = 'track_dataset/smer_mock_batch_lengths'
                    valid_batch_name = 'track_dataset/smer_mock_batch'
                    valid_batch_length_name = 'track_dataset/smer_mock_batch_lengths'
                    test_batch_name = 'track_dataset/smer_mock_batch'
                    test_batch_length_name = 'track_dataset/smer_mock_batch_lengths'
                else:
                    # bar control added
                    train_batch_name = 'bar_dataset/smer_bar_mock_batch'
                    train_length_name = 'bar_dataset/smer_bar_mock_batch_lengths'
                    valid_batch_name = 'bar_dataset/smer_bar_mock_batch'
                    valid_batch_length_name = 'bar_dataset/smer_bar_mock_batch_lengths'
                    test_batch_name = 'bar_dataset/smer_bar_mock_batch'
                    test_batch_length_name = 'bar_dataset/smer_bar_mock_batch_lengths'



            else:
                if control_mode == 0:
                    # track control only
                    train_batch_name = 'track_dataset/remi_mock_batch'
                    train_length_name = 'track_dataset/remi_mock_batch_lengths'
                    valid_batch_name = 'track_dataset/remi_mock_batch'
                    valid_batch_length_name = 'track_dataset/remi_mock_batch_lengths'
                else:
                    # bar control added
                    train_batch_name = 'bar_dataset/remi_mock_batch'
                    train_length_name = 'bar_dataset/remi_mock_batch_lengths'
                    valid_batch_name = 'bar_dataset/remi_mock_batch'
                    valid_batch_length_name = 'bar_dataset/remi_mock_batch_lengths'

        else:
            if vocab_mode == 0:
                if control_mode == 0:
                    # track control only
                    train_batch_name = 'track_dataset/smer_training_batch'
                    train_length_name = 'track_dataset/smer_training_batch_lengths'
                    valid_batch_name = 'track_dataset/smer_validation_batch'
                    valid_batch_length_name = 'track_dataset/smer_validation_batch_lengths'
                    test_batch_name = 'track_dataset/smer_test_batch'
                    test_batch_length_name = 'track_dataset/smer_test_batch_lengths'
                else:
                    # bar control added
                    train_batch_name = 'bar_dataset/smer_bar_training_batch'
                    train_length_name = 'bar_dataset/smer_bar_training_batch_lengths'
                    valid_batch_name = 'bar_dataset/smer_bar_validation_batch'
                    valid_batch_length_name = 'bar_dataset/smer_bar_validation_batch_lengths'



            else:
                if control_mode == 0:
                    # track control only
                    train_batch_name = 'track_dataset/remi_training_batch'
                    train_length_name = 'track_dataset/remi_training_batch_lengths'
                    valid_batch_name = 'track_dataset/remi_validation_batch'
                    valid_batch_length_name = 'track_dataset/remi_validation_batch_lengths'
                    test_batch_name = 'track_dataset/remi_test_batch'
                    test_batch_length_name = 'track_dataset/remi_test_batch_lengths'
                else:
                    # bar control added
                    train_batch_name = 'bar_dataset/remi_training_batch'
                    train_length_name = 'bar_dataset/remi_training_batch_lengths'
                    valid_batch_name = 'bar_dataset/remi_validation_batch'
                    valid_batch_length_name = 'bar_dataset/remi_validation_batch_lengths'


        if not is_test:
            logger.info(f'folder prefix is {folder_prefix}')
            logger.info(f'training batch name is {folder_prefix + train_batch_name}')

            if not os.path.exists(folder_prefix + train_batch_name):
                    logger.info('training batch not exist, exit')
                    sys.exit(1)
            train_batches = pickle.load(open(folder_prefix + train_batch_name, 'rb'))
            train_batch_lengths = pickle.load(open(folder_prefix + train_length_name, 'rb'))

            valid_batches = pickle.load(open(folder_prefix + valid_batch_name, 'rb'))
            valid_batch_lengths = pickle.load(
                open(folder_prefix + valid_batch_length_name, 'rb'))
            logger.info(f'training batch loaded')
            logger.info(f'train batches file is  {folder_prefix + train_batch_name}')
            logger.info(f'valid batches file is  {folder_prefix + valid_batch_name}')

            logger.info(f'train batch length is {len(train_batches)}')
            logger.info(f'valid batch length is {len(valid_batches)}')
        else:
            logger.info(f'folder prefix is {folder_prefix}')
            logger.info(f'test batch name is {folder_prefix + test_batch_name}')

            if not os.path.exists(folder_prefix + test_batch_name):
                logger.info('test batch not exist, exit')
                sys.exit(1)
            test_batches = pickle.load(open(folder_prefix + test_batch_name, 'rb'))
            test_batch_lengths = pickle.load(open(folder_prefix + test_batch_length_name, 'rb'))
            logger.info(f'test batch loaded')
            logger.info(f'test batch length is {len(test_batches)}')

        # for events in train_batches:
        #
        #     for event in events:
        #
        #         r = re.compile('track_\d')
        #
        #         track_names = list(set(filter(r.match, event)))
        #         track_names.sort()
        #
        #         bar_poses = np.where(np.array(event) == 'bar')[0]
        #
        #         r = re.compile('i_\d')
        #
        #         track_program = list(filter(r.match, event))
        #         track_nums = len(track_program)
        #
        #         if track_nums != len(track_names):
        #             print('invalid data')
        #
        #         if track_nums != len(track_names):
        #             print('invalid data')
        #
        #         r = re.compile('d_\d')
        #         density_controls = set(filter(r.match, event))
        #         if len(density_controls) > 0:
        #             for density_token in event[3:3 + track_nums]:
        #                 if density_token not in vocab.name_to_tokens['density']:
        #                     print('invalid data')
        #
        #         r = re.compile('o_\d')
        #         occupation_controls = set(filter(r.match, event))
        #         if len(occupation_controls) > 0:
        #             for occupation_token in event[3 + track_nums:3 + track_nums * 2]:
        #                 if occupation_token not in vocab.name_to_tokens['occupation']:
        #                     print('invalid data')
        #
        #         r = re.compile('y_\d')
        #         polyphony_controls = set(filter(r.match, event))
        #         if len(polyphony_controls) > 0:
        #             for polyphony_token in event[3 + track_nums * 2:3 + track_nums * 3]:
        #                 if polyphony_token not in vocab.name_to_tokens['polyphony']:
        #                     print('invalid data')



        # print(f'test batch length is {len(test_batches)}')

        if control_mode == 0:
            bar_track_control = False
            bar_control_at_end = False
        elif control_mode == 1:
            bar_track_control = True
            bar_control_at_end = False
        else:
            bar_track_control = True
            bar_control_at_end = True
        if not is_test:
            train_dataset_pretraining = ParallelLanguageDataset(vocab,
                                                                train_batches,
                                                                train_batch_lengths,
                                                                config['batch_size'],
                                                                total_mask_ratio=.15,
                                                                logger=logger,
                                                                pretraining=True,
                                                                bar_track_control=bar_track_control,
                                                                bar_control_at_end=bar_control_at_end
                                                                )

            valid_dataset_pretraining = ParallelLanguageDataset(vocab,
                                                                valid_batches,
                                                                valid_batch_lengths,
                                                                config['batch_size'],
                                                                total_mask_ratio=.15,
                                                                logger=logger,
                                                                pretraining=True,
                                                                bar_track_control=bar_track_control,
                                                                bar_control_at_end=bar_control_at_end)

            train_data_loader_pretraining = DataLoader(train_dataset_pretraining, batch_size=config['batch_size'],
                                                       collate_fn=lambda batch: collate_mlm_pretraining(batch),
                                                       num_workers=1, pin_memory=True)
            valid_data_loader_pretraining = DataLoader(valid_dataset_pretraining, batch_size=config['batch_size'],
                                                       collate_fn=lambda batch: collate_mlm_pretraining(batch),
                                                       num_workers=1, pin_memory=True)

            train_dataset_finetuning = ParallelLanguageDataset(vocab,
                                                               train_batches,
                                                               train_batch_lengths,
                                                               config['batch_size'],
                                                               total_mask_ratio=.15,
                                                               logger=logger,
                                                               pretraining=False,
                                                               bar_track_control=bar_track_control,
                                                               bar_control_at_end=bar_control_at_end)

            valid_dataset_finetuning = ParallelLanguageDataset(vocab,
                                                               valid_batches,
                                                               valid_batch_lengths,
                                                               config['batch_size'],
                                                               total_mask_ratio=.15,
                                                               logger=logger,
                                                               pretraining=False,
                                                               bar_track_control=bar_track_control,
                                                               bar_control_at_end=bar_control_at_end)

            train_data_loader_finetuning = DataLoader(train_dataset_finetuning, batch_size=config['batch_size'],
                                                      collate_fn=lambda batch: collate_mlm_finetuning(batch),
                                                      num_workers=1,
                                                      pin_memory=True)
            valid_data_loader_finetuning = DataLoader(valid_dataset_finetuning, batch_size=config['batch_size'],
                                                      collate_fn=lambda batch: collate_mlm_finetuning(batch),
                                                      num_workers=1,
                                                      pin_memory=True)
        else:

            test_dataset = ParallelLanguageDataset(vocab,
                                                   test_batches,
                                                   test_batch_lengths,
                                                   config['batch_size'],
                                                   total_mask_ratio=.15,
                                                   logger=logger,
                                                   pretraining=False,
                                                   bar_track_control=bar_track_control,
                                                   bar_control_at_end=bar_control_at_end)



            test_data_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                                                      collate_fn=lambda batch: collate_mlm_finetuning(batch), num_workers=1,
                                                      pin_memory=True)

        meta_weight = torch.zeros(vocab.vocab_size, device=device)

        if fine_tuning or is_test:

            meta_weight[1] = 1
        else:
            meta_weight[1] = config['eos_weight']

        meta_loss = nn.CrossEntropyLoss(ignore_index=0, weight=meta_weight, reduction='none')

        ce_weight_all = torch.ones(vocab.vocab_size, device=device)
        # pad
        ce_weight_all[0] = 0
        # mask
        ce_weight_all[2] = 0
        # unk
        ce_weight_all[-1] = 0
        if fine_tuning or is_test:
            ce_weight_all[1] = 1
        else:
            ce_weight_all[1] = config['eos_weight']

        structure_weight = torch.zeros(vocab.vocab_size, device=device)
        structure_weight[3:7] = 1
        structure_loss = nn.CrossEntropyLoss(ignore_index=0, weight=structure_weight, reduction='none')

        time_signature_weight = torch.zeros(vocab.vocab_size, device=device)
        time_signature_weight[7:11] = 1
        time_signature_loss = nn.CrossEntropyLoss(ignore_index=0, weight=time_signature_weight, reduction='none')

        tempo_weight = torch.zeros(vocab.vocab_size, device=device)
        tempo_weight[11:18] = 1
        tempo_loss = nn.CrossEntropyLoss(ignore_index=0, weight=tempo_weight, reduction='none')

        program_weight = torch.zeros(vocab.vocab_size, device=device)
        program_weight[18:146] = 1
        program_loss = nn.CrossEntropyLoss(ignore_index=0, weight=program_weight, reduction='none')

        pitch_weight = torch.zeros(vocab.vocab_size, device=device)
        pitch_weight[146:234] = 1
        pitch_loss = nn.CrossEntropyLoss(ignore_index=0, weight=pitch_weight, reduction='none')

        duration_weight = torch.zeros(vocab.vocab_size, device=device)

        duration_weight[234:234 + len(vocab.duration_indices)] = 1
        duration_loss = nn.CrossEntropyLoss(ignore_index=0, weight=duration_weight, reduction='none')

        criteria = [meta_loss, structure_loss, time_signature_loss, tempo_loss, program_loss, pitch_loss, duration_loss]

        if 'key' in vocab.control_indices.keys():
            key_weight = torch.zeros(vocab.vocab_size, device=device)

            key_weight[vocab.control_indices['key'][0]:vocab.control_indices['key'][-1] + 1] = 1
            key_loss = nn.CrossEntropyLoss(ignore_index=0, weight=key_weight, reduction='none')

            criteria.append(key_loss)

        if 'tensile' in vocab.control_indices.keys():
            tensile_weight = torch.zeros(vocab.vocab_size, device=device)

            tensile_weight[vocab.control_indices['tensile'][0]:vocab.control_indices['tensile'][-1] + 1] = 1
            tensile_loss = nn.CrossEntropyLoss(ignore_index=0, weight=tensile_weight, reduction='none')

            criteria.append(tensile_loss)

        if 'density' in vocab.control_indices.keys():
            density_weight = torch.zeros(vocab.vocab_size, device=device)

            density_weight[vocab.control_indices['density'][0]:vocab.control_indices['density'][-1] + 1] = 1
            density_loss = nn.CrossEntropyLoss(ignore_index=0, weight=density_weight, reduction='none')

            criteria.append(density_loss)

        if 'polyphony' in vocab.control_indices.keys():
            polyphony_weight = torch.zeros(vocab.vocab_size, device=device)

            polyphony_weight[vocab.control_indices['polyphony'][0]:vocab.control_indices['polyphony'][-1] + 1] = 1
            polyphony_loss = nn.CrossEntropyLoss(ignore_index=0, weight=polyphony_weight, reduction='none')

            criteria.append(polyphony_loss)

        if 'occupation' in vocab.control_indices.keys():
            occupation_weight = torch.zeros(vocab.vocab_size, device=device)

            occupation_weight[vocab.control_indices['occupation'][0]:vocab.control_indices['occupation'][-1] + 1] = 1
            occupation_loss = nn.CrossEntropyLoss(ignore_index=0, weight=occupation_weight, reduction='none')

            criteria.append(occupation_loss)

        print_every = 100

        learning_rate_adjust_interval = 5000
        if not is_test:
            model.train()

            lowest_val = 1e9
            train_losses = []

            train_accuracies = {'total': 0
                                }

            for token_type in set(vocab.token_class_ranges.values()):
                train_accuracies[token_type] = 0

            total_step = 0
            lr = 0
            wandb.watch(model, criteria, log="all", log_freq=print_every)
            log_step = 0
            scheduler_optim = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, patience=2, factor=0.5, min_lr=0.0000001, verbose=True)

            pretraining_epochs = 2

            for epoch in range(start_epoch, config.epochs):

                if epoch >= pretraining_epochs:
                    meta_weight[1] = 1
                    meta_loss = nn.CrossEntropyLoss(ignore_index=0, weight=meta_weight, reduction='none')
                    ce_weight_all[1] = 1
                    train_data_loader = train_data_loader_finetuning
                    valid_data_loader = valid_data_loader_finetuning
                    logger.info(f'fine tuning in epoch {epoch + 1}')
                else:
                    train_data_loader = train_data_loader_pretraining
                    valid_data_loader = valid_data_loader_pretraining
                    logger.info(f'pretraning in epoch {epoch + 1}')

                every_print_accuracy = {'total': 0}
                for token_type in set(vocab.token_class_ranges.values()):
                    every_print_accuracy[token_type] = 0

                example_ct = 0
                total_loss = 0
                meta_losses = 0
                time_signature_losses = 0
                program_losses = 0
                structure_losses = 0
                tempo_losses = 0
                pitch_losses = 0
                duration_losses = 0

                key_losses = 0
                tensile_losses = 0
                density_losses = 0
                occupation_losses = 0
                polyphony_losses = 0

                for step, data in enumerate(train_data_loader):
                    total_step += 1
                    example_ct += len(data['input'])
                    # Send the batches and key_padding_masks to gpu
                    src, src_key_padding_mask = data['input'].to(device), data['input_pad_mask'].to(device)
                    tgt_inp, tgt_key_padding_mask = data['target_in'].to(device), data['target_pad_mask'].to(device)
                    tgt_out = data['target_out'].to(device)

                    memory_key_padding_mask = src_key_padding_mask.clone()

                    # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)

                    tgt_mask = gen_nopeek_mask(tgt_inp.shape[1])
                    tgt_mask = torch.tensor(
                        np.repeat(np.expand_dims(tgt_mask, 0), memory_key_padding_mask.shape[0], axis=0)).float()

                    tgt_mask = tgt_mask.to(device)

                    # Forward
                    optim.zero_grad()
                    outputs, weights = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask,
                                             memory_key_padding_mask, tgt_mask)

                    loss_input_1 = rearrange(outputs, 'b t v -> (b t) v')
                    loss_input_2 = rearrange(tgt_out, 'b o -> (b o)')
                    loss1 = meta_loss(loss_input_1, loss_input_2)
                    loss2 = time_signature_loss(loss_input_1, loss_input_2)
                    loss3 = program_loss(loss_input_1, loss_input_2)
                    loss4 = tempo_loss(loss_input_1, loss_input_2)
                    loss5 = structure_loss(loss_input_1, loss_input_2)
                    loss6 = pitch_loss(loss_input_1, loss_input_2)
                    loss7 = duration_loss(loss_input_1, loss_input_2)

                    loss1 = torch.sum(loss1) / ce_weight_all[loss_input_2].sum()
                    loss2 = torch.sum(loss2) / ce_weight_all[loss_input_2].sum()
                    loss3 = torch.sum(loss3) / ce_weight_all[loss_input_2].sum()
                    loss4 = torch.sum(loss4) / ce_weight_all[loss_input_2].sum()
                    loss5 = torch.sum(loss5) / ce_weight_all[loss_input_2].sum()
                    loss6 = torch.sum(loss6) / ce_weight_all[loss_input_2].sum()
                    loss7 = torch.sum(loss7) / ce_weight_all[loss_input_2].sum()

                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

                    if 'tensile' in vocab.control_indices.keys():
                        tensile_this_loss = tensile_loss(loss_input_1, loss_input_2)

                        tensile_this_loss = torch.sum(tensile_this_loss) / ce_weight_all[loss_input_2].sum()
                        loss += tensile_this_loss

                        tensile_losses += tensile_this_loss.item()

                    if 'key' in vocab.control_indices.keys():
                        key_this_loss = key_loss(loss_input_1, loss_input_2)

                        key_this_loss = torch.sum(key_this_loss) / ce_weight_all[loss_input_2].sum()
                        loss += key_this_loss
                        key_losses += key_this_loss.item()

                    if 'density' in vocab.control_indices.keys():
                        density_this_loss = density_loss(loss_input_1, loss_input_2)

                        density_this_loss = torch.sum(density_this_loss) / ce_weight_all[loss_input_2].sum()
                        loss += density_this_loss
                        density_losses += density_this_loss.item()

                    if 'occupation' in vocab.control_indices.keys():
                        occupation_this_loss = occupation_loss(loss_input_1, loss_input_2)

                        occupation_this_loss = torch.sum(occupation_this_loss) / ce_weight_all[loss_input_2].sum()
                        loss += occupation_this_loss
                        occupation_losses += occupation_this_loss.item()

                    if 'polyphony' in vocab.control_indices.keys():
                        polyphony_this_loss = polyphony_loss(loss_input_1, loss_input_2)

                        polyphony_this_loss = torch.sum(polyphony_this_loss) / ce_weight_all[loss_input_2].sum()
                        loss += polyphony_this_loss
                        polyphony_losses += polyphony_this_loss.item()

                    # Backpropagate and update optim
                    loss.backward()

                    # optim.step_and_update_lr()
                    optim.step()

                    total_loss += loss.item()
                    meta_losses += loss1.item()
                    time_signature_losses += loss2.item()
                    program_losses += loss3.item()
                    tempo_losses += loss4.item()
                    structure_losses += loss5.item()
                    pitch_losses += loss6.item()
                    duration_losses += loss7.item()

                    train_losses.append(loss.item())

                    # pbar.update(1)
                    if step % print_every == print_every - 1:

                        log_step += 1
                        # if step % learning_rate_adjust_interval == learning_rate_adjust_interval - 1:
                        #     scheduler_optim.step(total_loss / print_every)
                        # pbar.close()
                        times = int((step / (print_every - 1)))

                        src_token = []

                        for i, output in enumerate(src[0]):
                            output_token = vocab.index2char(output.item())
                            src_token.append(output_token)

                        accuracies, generated_output, target_output = accuracy(outputs, tgt_out, vocab)

                        for token_type in accuracies.keys():
                            every_print_accuracy[token_type] += accuracies[token_type]

                        wandb.log({"epoch": epoch,
                                   "train_loss": loss,
                                   "meta_loss": loss1,
                                   "time_signature_loss": loss2,
                                   "program_loss": loss3,
                                   "tempo_loss": loss4,
                                   "structure_loss": loss5,
                                   "pitch_loss": loss6,
                                   "duration_loss": loss7,
                                   "total_accuracy": every_print_accuracy["total"] / times,
                                   "lr": optim.param_groups[0]['lr'],
                                   "real_batch_num": example_ct,
                                   }, step=log_step)

                        if 'tensile' in vocab.control_indices.keys():
                            wandb.log({"tensile": tensile_this_loss}, step=log_step)

                        if 'key' in vocab.control_indices.keys():
                            wandb.log({"key": key_this_loss}, step=log_step)
                        if 'density' in vocab.control_indices.keys():
                            wandb.log({"density": density_this_loss}, step=log_step)

                        if 'occupation' in vocab.control_indices.keys():
                            wandb.log({"occupation": occupation_this_loss}, step=log_step)
                        if 'polyphony' in vocab.control_indices.keys():
                            wandb.log({"polyphony": polyphony_this_loss}, step=log_step)

                        logger.info(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_data_loader)}] \n \
                                            train loss: {total_loss / print_every} \n \
                                            total accuracy: {every_print_accuracy["total"] / times} \n \
                                            meta loss: {meta_losses / print_every} \n \
                                            time signature loss: {time_signature_losses / print_every} \n \
                                            program loss : {program_losses / print_every} \n \
                                            tempo loss: {tempo_losses / print_every} \n \
                                            structure loss: {structure_losses / print_every} \n \
                                            pitch loss: {pitch_losses / print_every} \n \
                                            duration loss: {duration_losses / print_every} \n \
                                            ')
                        if 'tensile' in vocab.control_indices.keys() and epoch < pretraining_epochs:
                            logger.info(f'tensile loss: {tensile_losses / print_every}')

                        if 'key' in vocab.control_indices.keys() and epoch < pretraining_epochs:
                            logger.info(f'key loss: {key_losses / print_every}')
                        if 'density' in vocab.control_indices.keys() and epoch < pretraining_epochs:
                            logger.info(f'density loss: {density_losses / print_every}')

                        if 'occupation' in vocab.control_indices.keys() and epoch < pretraining_epochs:
                            logger.info(f'occupation loss: {occupation_losses / print_every}')
                        if 'polyphony' in vocab.control_indices.keys() and epoch < pretraining_epochs:
                            logger.info(f'polyphony loss: {polyphony_losses / print_every}')

                        if lr != optim.param_groups[0]['lr']:
                            lr = optim.param_groups[0]['lr']
                            logger.info(f'learning rate is {lr}')

                        for token_type in every_print_accuracy.keys():
                            logger.debug(f'{token_type} accuracy is {every_print_accuracy[token_type] / times}')

                        for token_type in every_print_accuracy.keys():
                            wandb.log({f'{token_type}_acc': every_print_accuracy[token_type] / times,
                                       'real_batch_num': example_ct,
                                       }, step=log_step)

                        total_loss = 0
                        meta_losses = 0
                        time_signature_losses = 0
                        program_losses = 0
                        tempo_losses = 0
                        structure_losses = 0
                        pitch_losses = 0
                        duration_losses = 0

                        key_losses = 0
                        tensile_losses = 0

                        density_losses = 0

                        occupation_losses = 0
                        polyphony_losses = 0

                    # logger.info(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_data_loader)}] \n '
                    #             f'Train Loss: {total_loss / print_every} \t Accuracy {accuracies["total"] / times} \n'
                    if step % (print_every * 10) == print_every * 10 - 1:
                        # wandb.log({'input_size':src.size(),
                        #            'input':src_token[:50],
                        #             'output_size': len(target_output),
                        #              'generated_output': generated_output[:50],
                        #              'target_output': target_output[:50],
                        #              })

                        logger.debug(f'input size is {src.size()} \n'
                                     f'input is : {src_token[:50]} \n'
                                     f'output size is {len(target_output)} \n'
                                     f'generated output: {generated_output[:50]} \n'
                                     f'target output: {target_output[:50]} \n'
                                     )

                        # pbar = tqdm(total=print_every, leave=False)

                logger.info(f'Epoch [{epoch + 1} / {num_epochs} end]')

                for token_type in train_accuracies.keys():
                    train_accuracies[token_type] = (every_print_accuracy[token_type] / times)
                    wandb.log({
                        f'ave_epoch_train_{token_type}_acc': train_accuracies[token_type],
                        'epoch_metrics_step': epoch}, step=log_step
                    )
                    logger.info(f'ave_epoch_train_{token_type}_acc is {train_accuracies[token_type]}')

                average_train_loss = np.mean(train_losses)
                scheduler_optim.step(average_train_loss)
                wandb.log({
                    'ave_epoch_train_loss': average_train_loss,
                    'epoch_metrics_step': epoch}, step=log_step
                )

                logger.info(
                    f'average train losses is {average_train_loss} \t'
                )

                # Validate every epoch
                # pbar.close()
                logger.info(f'valid data size is {len(valid_data_loader)}')
                val_loss, val_accuracy = validate(valid_data_loader, model, ce_weight_all, criteria,
                                                  device, vocab, logger, epoch, pretraining_epochs)

                for key in val_loss.keys():
                    wandb.log({f'val_{key}_loss': val_loss[key],
                               'epoch_metrics_step': epoch}, step=log_step)
                    logger.info(f'validation {key} loss is {val_loss[key]}')

                for key in val_accuracy.keys():
                    wandb.log({f'val_{key}_accuracy': val_accuracy[key],
                               'epoch_metrics_step': epoch}, step=log_step)
                    logger.info(f'validation {key} accuracy is {val_accuracy[key]}')

                # val_losses.append(val_loss)

                path = os.path.join(wandb.run.dir, f"checkpoint_{epoch}")
                logger.info(f'checkpoint_dir is {path}')

                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optim.state_dict(),
                            'epoch': epoch,
                            'loss': val_loss['total']}, path)

            logger.info("Finished Training")
        else:
            logger.info(f'test data size is {len(test_data_loader)}')
            test_loss, test_accuracy = test_loss_accuracy(test_data_loader, model, ce_weight_all,criteria, device, vocab, logger)
            for key in test_loss.keys():
                logger.info(f'test {key} loss is {test_loss[key]}')

            for key in test_accuracy.keys():

                logger.info(f'test {key} accuracy is {test_accuracy[key]}')



def accuracy(outputs, targets, vocab):
    # define total accuracy, structure accuracy, control accuracy,
    # duration accuracy, pitch accuracy
    with torch.no_grad():
        # print('\n')
        accuracy_result = {}
        types_number_counter = {}
        all_type_token = set(vocab.token_class_ranges.values())
        for one_type_token in all_type_token:
            accuracy_result[one_type_token] = 0
            types_number_counter[one_type_token] = 0
        accuracy_result['total'] = 0
        types_number_counter['total'] = 0

        generated_output = []
        target_output = []

        for i, output in enumerate(outputs):

            for position, token_idx in enumerate(torch.argmax(output, axis=1)):
                # output_classes = vocab.get_token_classes(token_idx)

                output_token = vocab.index2char(token_idx.item())

                target_idx = targets[i][position].item()
                target_token = vocab.index2char(target_idx)
                # log the first one to print
                if i == 0:
                    generated_output.append(output_token)
                    target_output.append(target_token)

                if target_idx == vocab.pad_index:
                    continue

                target_classes = vocab.get_token_classes(target_idx)

                accuracy_result[target_classes] += token_idx.item() == target_idx
                types_number_counter[target_classes] += 1

                accuracy_result['total'] += token_idx.item() == target_idx
                types_number_counter['total'] += 1

        for token_type in accuracy_result.keys():
            if types_number_counter[token_type] != 0:
                accuracy_result[token_type] /= types_number_counter[token_type]

        return accuracy_result, generated_output, target_output


def validate(valid_loader, model, ce_weight_all, criteria, device, vocab, logger, epoch, pretraining_epochs):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    meta_loss, time_signature_loss, program_loss, tempo_loss, \
    structure_loss, pitch_loss, duration_loss = criteria[:7]

    start_number = 7

    if 'key' in vocab.control_indices.keys():
        key_loss = criteria[start_number]
        start_number += 1

    if 'tensile' in vocab.control_indices.keys():
        tensile_loss = criteria[start_number]
        start_number += 1

    if 'density' in vocab.control_indices.keys():
        density_loss = criteria[start_number]
        start_number += 1

    if 'polyphony' in vocab.control_indices.keys():
        polyphony_loss = criteria[start_number]
        start_number += 1

    if 'occupation' in vocab.control_indices.keys():
        occupation_loss = criteria[start_number]
        start_number += 1

    total_loss = {'total': 0, 'meta': 0}

    total_steps = 0

    total_accuracy = {'total': 0}

    for token_type in set(vocab.token_class_ranges.values()):
        total_accuracy[token_type] = 0
        total_loss[token_type] = 0

    for data in iter(valid_loader):
        total_steps += 1
        if total_steps % 100 == 0:
            logger.info(f'validation steps is {total_steps} / {len(valid_loader)}')
        # Send the batches and key_padding_masks to gpu
        src, src_key_padding_mask = data['input'].to(device), data['input_pad_mask'].to(device)
        tgt_inp, tgt_key_padding_mask = data['target_in'].to(device), data['target_pad_mask'].to(device)
        tgt_out = data['target_out'].to(device)

        memory_key_padding_mask = src_key_padding_mask.clone()

        # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)

        tgt_mask = gen_nopeek_mask(tgt_inp.shape[1])
        tgt_mask = torch.tensor(
            np.repeat(np.expand_dims(tgt_mask, 0), memory_key_padding_mask.shape[0], axis=0)).float()

        tgt_mask = tgt_mask.to(device)
        with torch.no_grad():
            outputs, weights = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask,
                                     tgt_mask)

            loss_input_1 = rearrange(outputs, 'b t v -> (b t) v')
            loss_input_2 = rearrange(tgt_out, 'b o -> (b o)')

            loss1 = meta_loss(loss_input_1, loss_input_2)
            loss2 = time_signature_loss(loss_input_1, loss_input_2)
            loss3 = program_loss(loss_input_1, loss_input_2)
            loss4 = tempo_loss(loss_input_1, loss_input_2)
            loss5 = structure_loss(loss_input_1, loss_input_2)
            loss6 = pitch_loss(loss_input_1, loss_input_2)
            loss7 = duration_loss(loss_input_1, loss_input_2)

            loss1 = torch.sum(loss1) / ce_weight_all[loss_input_2].sum()
            loss2 = torch.sum(loss2) / ce_weight_all[loss_input_2].sum()
            loss3 = torch.sum(loss3) / ce_weight_all[loss_input_2].sum()
            loss4 = torch.sum(loss4) / ce_weight_all[loss_input_2].sum()
            loss5 = torch.sum(loss5) / ce_weight_all[loss_input_2].sum()
            loss6 = torch.sum(loss6) / ce_weight_all[loss_input_2].sum()
            loss7 = torch.sum(loss7) / ce_weight_all[loss_input_2].sum()

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

            total_loss['total'] += loss.item()
            total_loss['meta'] += loss1.item()
            total_loss['time_signature'] += loss2.item()
            total_loss['program'] += loss3.item()
            total_loss['tempo'] += loss4.item()
            total_loss['structure'] += loss5.item()
            total_loss['pitch'] += loss6.item()
            total_loss['duration'] += loss7.item()

            if 'tensile' in vocab.control_indices.keys():
                tensile_this_loss = tensile_loss(loss_input_1, loss_input_2)

                tensile_this_loss = torch.sum(tensile_this_loss) / ce_weight_all[loss_input_2].sum()
                loss += tensile_this_loss

                total_loss['tensile'] += tensile_this_loss.item()

            if 'key' in vocab.control_indices.keys():
                key_this_loss = key_loss(loss_input_1, loss_input_2)

                key_this_loss = torch.sum(key_this_loss) / ce_weight_all[loss_input_2].sum()
                loss += key_this_loss
                total_loss['key'] += key_this_loss.item()

            if 'density' in vocab.control_indices.keys():
                density_this_loss = density_loss(loss_input_1, loss_input_2)

                density_this_loss = torch.sum(density_this_loss) / ce_weight_all[loss_input_2].sum()
                loss += density_this_loss
                total_loss['density'] += density_this_loss.item()

            if 'occupation' in vocab.control_indices.keys():
                occupation_this_loss = occupation_loss(loss_input_1, loss_input_2)

                occupation_this_loss = torch.sum(occupation_this_loss) / ce_weight_all[loss_input_2].sum()
                loss += occupation_this_loss
                total_loss['occupation'] += occupation_this_loss.item()

            if 'polyphony' in vocab.control_indices.keys():
                polyphony_this_loss = polyphony_loss(loss_input_1, loss_input_2)

                polyphony_this_loss = torch.sum(polyphony_this_loss) / ce_weight_all[loss_input_2].sum()
                loss += polyphony_this_loss
                total_loss['polyphony'] += polyphony_this_loss.item()

            accuracies, generated_output, target_output = accuracy(outputs, tgt_out, vocab)

            for token_type in total_accuracy.keys():
                total_accuracy[token_type] += accuracies[token_type]

    for key in total_loss.keys():
        total_loss[key] /= total_steps

    for key in total_accuracy.keys():
        total_accuracy[key] /= total_steps

    src_token = []
    for i, output in enumerate(src[0]):
        output_token = vocab.index2char(output.item())
        src_token.append(output_token)

    logger.info(f'input size is {src.size()} \n'
                f'input is : {src_token[:50]} \n'
                f'output size is {len(target_output)} \n'
                f'generated output: {generated_output[:50]} \n'
                f'target output: {target_output[:50]} \n')

    # pbar = tqdm(total=print_every, leave=False)

    # pbar.update(1)

    # pbar.close()
    model.train()
    return total_loss, total_accuracy


def test_loss_accuracy(test_loader, model,ce_weight_all, criteria, device, vocab, logger):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    meta_loss, time_signature_loss, program_loss, tempo_loss, \
    structure_loss, pitch_loss, duration_loss = criteria[:7]

    start_number = 7

    if 'key' in vocab.control_indices.keys():
        key_loss = criteria[start_number]
        start_number += 1

    if 'tensile' in vocab.control_indices.keys():
        tensile_loss = criteria[start_number]
        start_number += 1

    if 'density' in vocab.control_indices.keys():
        density_loss = criteria[start_number]
        start_number += 1

    if 'polyphony' in vocab.control_indices.keys():
        polyphony_loss = criteria[start_number]
        start_number += 1

    if 'occupation' in vocab.control_indices.keys():
        occupation_loss = criteria[start_number]
        start_number += 1

    total_loss = {'total': 0, 'meta': 0}

    total_steps = 0

    total_accuracy = {'total': 0}

    for token_type in set(vocab.token_class_ranges.values()):
        total_accuracy[token_type] = 0
        total_loss[token_type] = 0

    for data in iter(test_loader):
        total_steps += 1
        if total_steps % 100 == 0:
            logger.info(f'test steps is {total_steps} / {len(test_loader)}')
        # Send the batches and key_padding_masks to gpu
        src, src_key_padding_mask = data['input'].to(device), data['input_pad_mask'].to(device)
        tgt_inp, tgt_key_padding_mask = data['target_in'].to(device), data['target_pad_mask'].to(device)
        tgt_out = data['target_out'].to(device)

        memory_key_padding_mask = src_key_padding_mask.clone()

        # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)

        tgt_mask = gen_nopeek_mask(tgt_inp.shape[1])
        tgt_mask = torch.tensor(
            np.repeat(np.expand_dims(tgt_mask, 0), memory_key_padding_mask.shape[0], axis=0)).float()

        tgt_mask = tgt_mask.to(device)
        with torch.no_grad():
            outputs, weights = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask,
                                     tgt_mask)

            loss_input_1 = rearrange(outputs, 'b t v -> (b t) v')
            loss_input_2 = rearrange(tgt_out, 'b o -> (b o)')

            loss1 = meta_loss(loss_input_1, loss_input_2)
            loss2 = time_signature_loss(loss_input_1, loss_input_2)
            loss3 = program_loss(loss_input_1, loss_input_2)
            loss4 = tempo_loss(loss_input_1, loss_input_2)
            loss5 = structure_loss(loss_input_1, loss_input_2)
            loss6 = pitch_loss(loss_input_1, loss_input_2)
            loss7 = duration_loss(loss_input_1, loss_input_2)

            loss1 = torch.sum(loss1) / ce_weight_all[loss_input_2].sum()
            loss2 = torch.sum(loss2) / ce_weight_all[loss_input_2].sum()
            loss3 = torch.sum(loss3) / ce_weight_all[loss_input_2].sum()
            loss4 = torch.sum(loss4) / ce_weight_all[loss_input_2].sum()
            loss5 = torch.sum(loss5) / ce_weight_all[loss_input_2].sum()
            loss6 = torch.sum(loss6) / ce_weight_all[loss_input_2].sum()
            loss7 = torch.sum(loss7) / ce_weight_all[loss_input_2].sum()

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

            total_loss['total'] += loss.item()
            total_loss['meta'] += loss1.item()
            total_loss['time_signature'] += loss2.item()
            total_loss['program'] += loss3.item()
            total_loss['tempo'] += loss4.item()
            total_loss['structure'] += loss5.item()
            total_loss['pitch'] += loss6.item()
            total_loss['duration'] += loss7.item()

            if 'tensile' in vocab.control_indices.keys():
                tensile_this_loss = tensile_loss(loss_input_1, loss_input_2)

                tensile_this_loss = torch.sum(tensile_this_loss) / ce_weight_all[loss_input_2].sum()
                loss += tensile_this_loss

                total_loss['tensile'] += tensile_this_loss.item()

            if 'key' in vocab.control_indices.keys():
                key_this_loss = key_loss(loss_input_1, loss_input_2)

                key_this_loss = torch.sum(key_this_loss) / ce_weight_all[loss_input_2].sum()
                loss += key_this_loss
                total_loss['key'] += key_this_loss.item()

            if 'density' in vocab.control_indices.keys():
                density_this_loss = density_loss(loss_input_1, loss_input_2)

                density_this_loss = torch.sum(density_this_loss) / ce_weight_all[loss_input_2].sum()
                loss += density_this_loss
                total_loss['density'] += density_this_loss.item()

            if 'occupation' in vocab.control_indices.keys():
                occupation_this_loss = occupation_loss(loss_input_1, loss_input_2)

                occupation_this_loss = torch.sum(occupation_this_loss) / ce_weight_all[loss_input_2].sum()
                loss += occupation_this_loss
                total_loss['occupation'] += occupation_this_loss.item()

            if 'polyphony' in vocab.control_indices.keys():
                polyphony_this_loss = polyphony_loss(loss_input_1, loss_input_2)

                polyphony_this_loss = torch.sum(polyphony_this_loss) / ce_weight_all[loss_input_2].sum()
                loss += polyphony_this_loss
                total_loss['polyphony'] += polyphony_this_loss.item()

            accuracies, generated_output, target_output = accuracy(outputs, tgt_out, vocab)

            for token_type in total_accuracy.keys():
                total_accuracy[token_type] += accuracies[token_type]

    for key in total_loss.keys():
        total_loss[key] /= total_steps

    for key in total_accuracy.keys():
        total_accuracy[key] /= total_steps

    src_token = []
    for i, output in enumerate(src[0]):
        output_token = vocab.index2char(output.item())
        src_token.append(output_token)

    logger.info(f'input size is {src.size()} \n'
                f'input is : {src_token[:50]} \n'
                f'output size is {len(target_output)} \n'
                f'generated output: {generated_output[:50]} \n'
                f'target output: {target_output[:50]} \n')

    # pbar = tqdm(total=print_every, leave=False)

    # pbar.update(1)

    # pbar.close()

    return total_loss, total_accuracy


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


if __name__ == "__main__":

    args = get_args()

    num_epochs = args.num_epochs
    platform = args.platform
    is_debug = args.is_debug
    checkpoint_dir = args.checkpoint_dir
    lr = args.learning_rate
    run_id = args.run_id
    fine_tuning = args.fine_tuning
    reset_epoch = args.reset_epoch
    vocab_mode = int(args.mode)
    encoder_layers = int(args.encoder_layers)

    control_number = args.control_number
    control_mode = args.control_mode
    device = args.device

    is_test = args.test_data

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

    # control_list = args.control_list

    # control_list = control_list.split()

    file_name = '_'.join(control_list)
    if is_test:
        file_name += '_is_test'

    if vocab_mode == 0:
        logfile = 'smer_control' + file_name + '.log'
    if vocab_mode == 1:
        logfile = 'remi_control' + file_name + '.log'
    logger = log.logger_init(logfile, 'w')
    main()
    sys.exit()
