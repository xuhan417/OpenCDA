# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from late_fusion_dataset import LateFusionDataset
from early_fusion_dataset import EarlyFusionDataset
from intermediate_fusion_dataset import IntermediateFusionDataset

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset
}

# the final range for evaluation
GT_RANGE = [-140, -40, -3, 140, 40, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['LateFusionDataset', 'EarlyFusionDataset',
                            'IntermediateFusionDataset'], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
