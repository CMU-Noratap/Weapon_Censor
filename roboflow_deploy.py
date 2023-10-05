from roboflow import Roboflow

home = "C:/Users/PC_ML/Desktop/Gun_Knife_Censor/Yolo/yolov5/runs/train/exp/"

rf = Roboflow(api_key="PtNU0H508OYHCbE79WAF")
project = rf.workspace().project("gun_and_knife_detection")

project.version(1).deploy(model_type="yolov5", model_path=f"{home}")