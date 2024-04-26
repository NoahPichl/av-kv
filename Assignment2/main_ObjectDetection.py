import os
import ObjectDetection.Open3D_ML.ml3d as _ml3d
import ObjectDetection.Open3D_ML.ml3d.torch as tml3d

from ObjectDetection.TaskDataset import TaskDataset


cfg_file = "/workspaces/AutomotiveVehicles/Assignment2/ObjectDetection/pointpillars_waymo.yml"
cfg =  _ml3d.utils.Config.load_from_file(cfg_file)

model = tml3d.models.PointPillars(**cfg.model)
dataset = TaskDataset(cfg.dataset)
pipeline = tml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointpillars_waymo_202211200158utc_seed2_gpu16.pth"
pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_waymo_202211200158utc_seed2_gpu16.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
    os.system(cmd)

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

test_split = dataset.get_split("test")
data = test_split.get_data(0)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
result = pipeline.run_inference(data)

# evaluate performance on the test set; this will write logs to './logs'.
pipeline.run_valid()