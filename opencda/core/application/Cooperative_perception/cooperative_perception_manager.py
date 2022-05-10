# -*- coding: utf-8 -*-
"""
Cooperative Perception module base.
"""

import open3d as o3d

from opencda.core.application.Cooperative_perception \
    import train_utils
from opencda.core.sensing.perception.perception_manager \
    import PerceptionManager

class CoopPerceptionManager(PerceptionManager):
    def __init__(self, vehicle, config_yaml, cav_world,
                 data_dump=False, carla_world=None, infra_id=None):

        super(CoopPerceptionManager,self).__init__(
            vehicle,
            config_yaml,
            cav_world,
            data_dump,
            carla_world,
            infra_id
        )

        self.fusion_method = config_yaml[]
        assert self.fusion_method in ['late', 'early', 'intermediate']

        # need to verify  (from 49,62-71)
        model_hypes = config_yaml[''] # need to check the 49 line in inference.py
        model = train_utils.create_model(model_hypes)
        # we assume gpu is necessary
        if torch.cuda.is_available():
            model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('Loading Model from checkpoint')
        save_path = config_yaml['model_dir']
        _, model = train_utils.load_saved_model(saved_path, model)
        self.model = model
        self.model.eval()

        self.show_sequence = config_yaml['show_sequence']

    def search(self):

    def detect(self, ego_pos):
        pass






if __name__ == '__main__':
    pass