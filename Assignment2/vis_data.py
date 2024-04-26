import os
import numpy as np
import ObjectDetection.Open3D_ML.ml3d as _ml3d
import ObjectDetection.Open3D_ML.ml3d.torch as tml3d
from ObjectDetection.Open3D_ML.ml3d.vis import Visualizer

from ObjectDetection.TaskDataset import TaskDataset


cfg_file = "/workspaces/AutomotiveVehicles/Assignment2/ObjectDetection/pointpillars_waymo.yml"
cfg =  _ml3d.utils.Config.load_from_file(cfg_file)
dataset = TaskDataset(cfg.dataset)
data = dataset.get_split("test").get_data(0)
points = data["point"]
data = [
    {
        'name': 'my_point_cloud',
        'points': points,
        'random_colors': np.random.rand(*points.shape).astype(np.float32),
        'int_attr': (points[:,0]*5).astype(np.int32),
    }
]

vis = Visualizer()
vis.visualize(data)