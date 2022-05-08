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

from quantization.binarize import binarize, IRLinear
from quantization.transformer import Transformer
from quantization.quantize import quantizer
from quantization.pytorch_api import ModelQuant
from quantization.fully_quantize import Model as fullyQuantModel

from utils.train_utils import change_t

def create_model(vocab_size, quant_type=None, quant_method=None, bit_num=None, quant_pattern=None):
    '''
    Create training model based on sepcified quant_type
    ----------
    Arguments:
    quant_type    - quant type, should be one of [None, 'quantization', 'binarization']
    quant_method  - quant method to use, if quant_type is None, it should also be None
                    For 'quantization', should be one of ['basic', 'pytorch', 'fully']
                    For 'binarization', should be one of ['basic', 'ir']
    bit_num       - bit number for each parameter, only works when quant_type is 'quantization'
                    should be one of [8,4,2]
    quant_pattern - quantization pattern, should be one of ['MHA', 'FFN', 'CLS', 'ALL']
    '''
    model = Transformer(d_model=BASELINE_MODEL_DIM,
                             d_ff=BASELINE_FFN_DIM,
                             d_hidden=BASELINE_HIDDEN_DIM,
                             h=BASELINE_MODEL_NUMBER_OF_HEADS,
                             n_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
                             n_class=4,
                             vocab=vocab_size
                            )
    
    __quant_type__ = [None,'quantization','binarization']
    __bit_num__ = [None,8,4,2]
    __quant_pattern__ = [None,'MHA', 'FFN', 'CLS', 'ALL']
    
    assert quant_type in __quant_type__, f"Unimplemented quantization type, should be one of {__quant_type__}, got '{quant_type}'!"
    assert bit_num in __bit_num__, f"Unimplemented bit number, should be one of {__bit_num__}, got '{bit_num}'!"
    assert quant_pattern in __quant_pattern__, f"Unimplemented quantization method, should be one of {__quant_pattern__}, got '{quant_pattern}'!"
    
    if quant_type == None:
        if quant_method is not None:
            print(f"Quant method {quant_method} will not work in baseline model!")
        if bit_num is not None:
            print(f"Bit number {bit_num} will not work in baseline model!")
        if quant_pattern is not None:
            print(f"Quant pattern {quant_pattern} will not work in baseline model!")
    
    elif quant_type == 'quantization':
        __quant_method__ = ['basic', 'pytorch', 'fully']
        
        assert quant_method in __quant_method__, f"Unimplemented quantization method, should be one of {__quant_method__}, got '{quant_method}'!"
        assert bit_num != None, f"Bit number can not be None!"
        assert quant_pattern != None, f"Quant pattern can not be None!"
        
        if quant_method == 'basic':
            if quant_pattern != 'ALL':
                print(f"Current quant method {quant_method} can only quantize the whole network, quant pattern {quant_pattern} will not work!")
            model = quantizer(model, bit_num, True)
            
        elif quant_method == 'pytorch':
            model = ModelQuant(d_model=BASELINE_MODEL_DIM,
                               d_ff=BASELINE_FFN_DIM,
                               d_hidden=BASELINE_HIDDEN_DIM,
                               h=BASELINE_MODEL_NUMBER_OF_HEADS,
                               n_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
                               n_class=4,
                               vocab=tokenizer.vocab_size,
                               quant_ffn=((quant_pattern == 'FFN')|(quant_pattern == 'ALL')),
                               quant_mha=((quant_pattern == 'MHA')|(quant_pattern == 'ALL')),
                               quant_classifier=((quant_pattern == 'CLS')|(quant_pattern == 'ALL')),
                               bit_num=bit_num)
            
        elif quant_method == 'fully':
            print("For fully_quantized model, bit number and quant pattern will not work!")
            model = fullyQuantModel(4,
                tokenizer.vocab_size,
                BASELINE_MODEL_NUMBER_OF_LAYERS,
                BASELINE_MODEL_NUMBER_OF_HEADS,
                BASELINE_MODEL_DIM)
            
    elif quant_type == 'binarization':
        __quant_method__ = ['basic', 'ir']
        assert quant_method in __quant_method__, f"Unimplemented quantization method, should be one of {__quant_method__}, got '{quant_method}'!"
        assert quant_pattern != None, f"Quant pattern can not be None!"
        print(f"For binarization model, bit num will not work!")
        
        binarize(model, quant_pattern, quant_method, skip_final=True, qk_only=True)

    
    return model

if __name__ == '__main__':
    
    # set parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--quant_type", type=str, help="whether to conduct quantization or binarization", default=None)
    parser.add_argument("--quant_method", type=str, help="which specific quantization method to use", default=None)
    parser.add_argument("--bit_num", type=int, help="bit number for each parameter", default=None)
    parser.add_argument("--quant_pattern", type=str, help="which part of transformer to quant", default=None)
    parser.add_argument("--exp_name", type=str, help="experiment name, to specify save path", default='tmp')
    
    args = parser.parse_args()
    
    # load dataset
    print("INFO: Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dl, test_dl = AG_NEWS_DATASET(tokenizer, batch_size = BATCH_SIZE).load_data()
    print("INFO: Finished!")
    
    # create model
    print("INFO: Creating model...")
    model = create_model(tokenizer.vocab_size, args.quant_type, args.quant_method, args.bit_num, args.quant_pattern)
    print("INFO: Finished!")
    print("======Model Structure======")
    print(model)
    
    # loss func
    loss_fn = nn.CrossEntropyLoss()

    # baseline training config -> do not change!
    optim = Adam(model.parameters(), lr= 1e-4)
    scheduler = MultiStepLR(optim, milestones=[10,15], gamma=0.1)

    train_config ={'model': model,
                   'loss_fn': loss_fn,
                   'optim': optim,
                   'scheduler': scheduler,
                   'datasets': [train_dl, test_dl],
                   'epochs': 10,
                   'batch_size': BATCH_SIZE
                   }

    train_config['exp_name'] = args.exp_name

    # training
    learner_ag_news = Learner(train_config, ir = (args.quant_method == 'ir'))
    
    print("INFO: Start Training...")
    learner_ag_news.train()
    print("INFO: Finished!")
    