# -*- coding: utf-8 -*-

import argparse
import os
import time
from easydict import EasyDict

from tools import yaml_utils

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt

def mimic_test_parser():
    opt= dict(model_dir='model_dir/v2vnet/',
              fusion_method='late',
              show_vis=False,
              show_sequence=False,
              save_vis=False,
              save_npy=False) 
    opt = EasyDict(opt)
    return opt 

    
if __name__ == '__main__':
    # opt = test_parser()
    opt = mimic_test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                'the results in single ' \
                                                'image mode or video mode'                                                
    print('opt.model_dir:',opt.model_dir)
    print('opt.fusion_method:',opt.fusion_method)
    print('opt.show_vis:',opt.show_vis)
    print('opt.show_sequence:',opt.show_sequence)
    print('opt.save_vis:',opt.save_vis)
    print('opt.save_npy:',opt.save_npy)
    
    hypes = yaml_utils.load_yaml(None, opt)
    
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    
    target_model_name = backbone_name.replace('_', '')
    
    
        
    
    


