import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR

from transformers import AutoTokenizer
from utils.data_utils import AG_NEWS_DATASET
from utils.constants import *
from utils.training import Learner

from quantization.transformer import Transformer
from quantization.binarize import binarize
from quantization.quantize import quantizer
from quantization.fully_quantize import Model as fullyQuantModel


def create_model(vocab_size, quant_type=None, quant_method=None, bit_num=None, quant_pattern=None):
    '''
    Create training model based on sepcified quant_type
    ----------
    Arguments:
    quant_type    - quant type, should be one of [None, 'quantization', 'binarization']
    quant_method  - quant method to use, if quant_type is None, it should also be None
                    For 'quantization', should be one of ['basic', 'fully']
                    For 'binarization', should be one of ['basic', 'ir']
    bit_num       - bit number for each parameter, only works when quant_type is 'quantization'
                    should be one of [8,4,2]
    quant_pattern - quantization pattern, should be one of ['MHA', 'FFN', 'CLS', 'ALL', 'ALL_QK']
    '''
    # create baseline model for further quantization
    model = Transformer(d_model=BASELINE_MODEL_DIM,
                             d_ff=BASELINE_FFN_DIM,
                             d_hidden=BASELINE_HIDDEN_DIM,
                             h=BASELINE_MODEL_NUMBER_OF_HEADS,
                             n_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
                             n_class=4,
                             vocab=vocab_size
                            )
    
    # if quant type == None, there is the baseline model without quantization
    # other arguments should not be defined
    if quant_type == None:
        assert quant_method is None, f"Quant method should not be specified in baseline model, got {quant_method}!"
        assert bit_num is None, f"Bit number should not be specified in baseline model, got {bit_num}!"
        assert quant_pattern is None, f"Quant pattern should not be specified in baseline model, got {quant_pattern}!"
        print("INFO: Creating baseline transformer with no quantization/binarization!")
        
    # if quant type == 'quantization', we will do quantization on base model
    # bit num must be specified as one of [8,4,2]
    # quant pattern should not be specified
    elif quant_type == 'quantization':
        __quant_method__ = ['basic', 'fully']
        assert quant_method in __quant_method__, f"Unimplemented quantization method, should be one of {__quant_method__}, got '{quant_method}'!"
        assert bit_num != None, f"Bit number can not be None!"
        assert quant_pattern is None, f"Quant pattern should not be specified in quantization, got {quant_pattern}!"
        
        if quant_method == 'basic':
            model = quantizer(model, bit_num, True)
            
        elif quant_method == 'fully':
            model = fullyQuantModel(4,
                                    vocab_size,
                                    BASELINE_MODEL_NUMBER_OF_LAYERS,
                                    BASELINE_MODEL_NUMBER_OF_HEADS,
                                    BASELINE_MODEL_DIM,
                                    k = bit_num)
        print(f"INFO: Creating quantized transformer with {bit_num}bit weights, using {quant_method} quant algorithm!")
    
    # if quant type == 'binarization', we will do binarization on base model
    # bit num should be None
    # quant pattern should be specified as one of [MHA, FFN, CLS, ALL, ALL_QK]
    # here all binarization method skip final layer
    elif quant_type == 'binarization':
        __quant_method__ = ['basic', 'ir']
        assert quant_method in __quant_method__, f"Unimplemented quantization method, should be one of {__quant_method__}, got '{quant_method}'!"
        assert quant_pattern != None, f"Quant pattern can not be None!"
        assert bit_num is None, f"Bit number should not be specified in binarization, got {bit_num}!"
        
        if quant_pattern == 'ALL_QK':
             binarize(model, 'ALL', quant_method, skip_final=True, kv_only=True)
        else:
            binarize(model, quant_pattern, quant_method, skip_final=True)
        
        print(f"INFO: Creating binarized transformer with using {quant_method} binarize algorithm, {quant_pattern} of whole model get binarized!")
    return model
if __name__ == '__main__':
    
    # set parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--quant_type", 
                        type=str, 
                        help="whether to conduct quantization or binarization", 
                        choices=['quantization', 'binarization'],
                        default=None)
    parser.add_argument("--quant_method", 
                        type=str, help="which specific quantization method to use", 
                        choices=['basic', 'fully', 'ir'],
                        default=None)
    parser.add_argument("--bit_num", 
                        type=int, 
                        help="bit number for each parameter", 
                        choices=[8,4,2],
                        default=None)
    parser.add_argument("--quant_pattern", 
                        type=str, 
                        help="which part of transformer to quant", 
                        choices=['MHA', 'FFN', 'CLS', 'ALL', 'ALL_QK'],
                        default=None)
    parser.add_argument("--pre_trained", 
                        help="whether to load pre-trained weight", 
                        action='store_true',
                        default=False)
    parser.add_argument("--latent", 
                        help="whether to do latent training", 
                        action='store_true',
                        default=False)
    parser.add_argument("--exp_name", type=str, help="experiment name, to specify save path", default='tmp')
    
    args = parser.parse_args()
    
    # load dataset
    print("INFO: Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dl, test_dl = AG_NEWS_DATASET(tokenizer, batch_size = BATCH_SIZE).load_data()
    print("INFO: Finished!")
    
    # create model
    model = create_model(tokenizer.vocab_size, args.quant_type, args.quant_method, args.bit_num, args.quant_pattern)
    print("INFO: Finished!")
    print("======Model Structure======")
    print(model)
    
    # if using pre-trained weights, load it
    if args.pre_trained:
        print("INFO: Loading pre-trained weights...")
        model.load_state_dict(torch.load(PRETRAINED_WEIGHT_PATH), strict=False)
        print("INFO: Finished!")
    
    
    # loss func
    loss_fn = nn.CrossEntropyLoss()

    # baseline training config 
    optim = Adam(model.parameters(), lr= 1e-4)
    scheduler = MultiStepLR(optim, milestones=[10,15], gamma=0.1)

    train_config ={'model': model,
                   'loss_fn': loss_fn,
                   'optim': optim,
                   'scheduler': scheduler,
                   'datasets': [train_dl, test_dl],
                   'epochs': 10,
                   'batch_size': BATCH_SIZE,
                   'exp_name': args.exp_name
                   }
    
    if args.latent:
        # create baseline model as original full-precision model
        org_model = Transformer(d_model=BASELINE_MODEL_DIM,
                                d_ff=BASELINE_FFN_DIM,
                                d_hidden=BASELINE_HIDDEN_DIM,
                                h=BASELINE_MODEL_NUMBER_OF_HEADS,
                                n_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
                                n_class=4,
                                vocab=tokenizer.vocab_size
                                )
        optim_original = Adam(org_model.parameters(), lr= 1e-4)
        scheduler_original = MultiStepLR(optim_original, milestones=[10,15], gamma=0.1)
        
        # specify learning setting for original model
        train_config['model_original'] = org_model
        train_config['optim_original'] = optim_original
        train_config['scheduler_original'] = scheduler_original
        print("INFO: Start Training...")
        
    else:
        print("INFO: Start Latent Training...")

    # training
    learner_ag_news = Learner(train_config, ema=(args.quant_method == 'fully'),  ir=(args.quant_method == 'ir'))
    
    
    learner_ag_news.train()
    print("INFO: Finished!")
    
