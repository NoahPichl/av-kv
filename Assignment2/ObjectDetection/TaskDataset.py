from ObjectDetection.Open3D_ML.ml3d.datasets.base_dataset import BaseDataset
from ObjectDetection.Open3D_ML.ml3d.datasets.utils.bev_box import BEVBox3D

from glob import glob
import os
import numpy as np
import sys
from itertools import permutations as perm
from pathlib import Path



sys.path.append("Assignment2/tools")
from tools.dataset_tools import read_frame, decode_img, decode_lidar

"""
Dataset which load files of the Task Dataset and returns always all the files.
"""

class TaskDataset(BaseDataset):
    def __init__(self, cfg, name="MyDataset", dataset_path="../Dataset/data_2"):
        super().__init__(name=name, dataset_path=dataset_path)
        self.data_path = dataset_path
        self.files =  [f for f in glob(dataset_path + "/*.pb")]
        self.cfg = cfg


    def get_split(self, split):
        return TaskDataSplit(self, split=split)

    def is_tested(self, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = os.path.join(path, self.name, name + '.npy')
        if os.path.exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        os.make_dir(path)

        pred = results['predict_labels']
        pred = np.array(self.label_to_names[pred])

        store_path = os.path.join(path, name + '.npy')
        np.save(store_path, pred)

    def get_label_to_names(self):
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'VEHICLE',
        }
        return label_to_names


    @staticmethod
    def _extend_matrix(mat):
        mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
        return mat

    
    @staticmethod
    def read_label(dataset: BaseDataset, detections: list):
        """Reads label data from the detection provided.

        Returns:
            labels: label data with BEVBox3D format.
        """
        labels = []
        mapping = dataset.get_label_to_names()
        for detection in detections:
            label = BevBox3DTask(center=detection.pos, 
                             size=[detection.scale[i] for i in [1, 2, 0]],   #l, w, h -> w, h, l
                             yaw=-detection.rot[-1], 
                             label_class=mapping[detection.type],
                             confidence=1.)
            labels.append(label)
        return labels


class TaskDataSplit():
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.split = split
        self.path_list = dataset.files
        

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        path = self.path_list[idx]
        frame = read_frame(path)
        pc = decode_lidar(frame.lidars[0])
        labels = self.dataset.read_label(self.dataset, frame.lidars[0].detections)
        return {
            'point': pc,  # reduce to points which are visible in camera
            'feat': None, 
            'calib': None, 
            'bounding_boxes': labels}

    def get_attr(self, idx):
        path = self.path_list[idx]
        name = Path(path).name.split('.')[0]
        return {'name': name, 'path': path, 'split': self.split}
    

class Track:
    
    def __init__(self, detection):
        self.x = np.asarray(detection.pos)
        self.l = detection.scale[0]
        self.w = detection.scale[1]
        self.h = detection.scale[2]
        self.yaw = detection.rot[2]
        self.state = "initialized"
        self.id = detection.id
        self.label_class = detection.id

# class to overwrite BevBox3D to reorder the size of the box
class BevBox3DTask(BEVBox3D):
    def __init__(self, center, size, yaw, label_class, confidence):
        super().__init__(center, size, yaw, label_class, confidence)
    
    def to_xyzwhlr(self):
        """Returns box in the common 7-sized vector representation: (x, y, z, w,
        l, h, a), where (x, y, z) is the bottom center of the box, (w, l, h) is
        the width, length and height of the box a is the yaw angle.

        Returns:
            box(7,)

        """
        bbox = np.zeros((7,))
        bbox[0:3] = self.center - [0, 0, self.size[1] / 2]
        bbox[3:6] = np.array(self.size)[[2, 1, 0]]  # reorder size
        bbox[6] = self.yaw
        return bbox
