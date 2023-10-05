import os
import glob

import numpy as np
import cv2

def txt_union_bbox(list_bbox,img):
    img_h = img.shape[0]
    img_w = img.shape[1]
    print()
    for bbox in list_bbox:
        _bbox = bbox.split(" ")
        print(_bbox)
        label = int(_bbox[0])
        x = float(_bbox[1])*img_w
        y = float(_bbox[2])*img_h
        w = float(_bbox[3])*img_w
        h = float(_bbox[4])*img_h

        x0 = int(x - w//2)
        x1 = int(x + w//2)
        y0 = int(y - h//2)
        y1 = int(y + h//2)

        print(x0,x1,y0,y1)

        img[y0:y1,x0:x1] = 1.0

    return img

def show_img(title,img):
    cv2.imshow(title, img)

def cal_iou(img_gt,img_pred):
    _img_gt = img_gt.astype(bool)
    _img_pred = img_pred.astype(bool)

    intersection = _img_gt*_img_pred # AND
    union = _img_gt+_img_pred

    iou = (np.sum(intersection) + 1e-320) / (np.sum(union) + 1e-320)

    return iou, intersection.astype(np.float32) , union.astype(np.float32)

img_h = 300
img_w = 300
varborn = False # False = not display img , True = display img

list_path_gt = glob.glob("C:/Users/PC_ML/Desktop/Gun_Knife_Censor/Dataset/guns-knives-yolo/guns-knives-yolo/test/labels/*")
#path_predict_root = "C:/Users/PC_ML/Desktop/Gun_Knife_Censor/Yolo/yolov5/runs/detect/exp8/labels/" # Yolov5
path_predict_root = "C:/Users/PC_ML/Desktop/Gun_Knife_Censor/CNN/predict/labels/" # CNN

print(f"num_tes:{len(list_path_gt)}")
list_iou = []

for path_gt in list_path_gt:
    file_name = path_gt.split("\\")[1]
    print(file_name)
    
    #Gt to img
    img_gt = np.zeros((img_h,img_w))
    with open(path_gt) as file:
        list_bbox = file.readlines()
        img = txt_union_bbox(list_bbox,img_gt)

        if varborn:
            show_img("GT",img*255)

    #Predict to img
    path_predict = path_predict_root + file_name
    img_pred = np.zeros((img_h,img_w))
    try:
        with open(path_predict) as file:
            list_bbox = file.readlines()
            img = txt_union_bbox(list_bbox,img_pred)

            if varborn:
                show_img("Predict",img*255)
    except:
        list_bbox = []
        img = txt_union_bbox(list_bbox,img_pred)

        if varborn:
            show_img("Predict",img*255)

    #Cal IOU
    iou , im_i, im_u = cal_iou(img_gt,img_pred)
    list_iou.append(iou)
    print(f"IOU:{iou}")

    if varborn:
        show_img("Intersection",im_i*255)
        show_img("Union",im_u*255)

        key = cv2.waitKey(0) 
        cv2.destroyAllWindows()
        if(key == ord('q')):
            break

print(f"mean_iou:{np.mean(list_iou)}")


