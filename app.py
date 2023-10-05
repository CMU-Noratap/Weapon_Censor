import streamlit as st
from PIL import Image
import cv2
import numpy as np

import os
import uuid

import subprocess
import sys

upload_path = "/mount/src/weapon_censor/upload_image/"
detect_path = "/mount/src/weapon_censor/yolov5/runs/detect/"
gradcam_path = "/mount/src/weapon_censor/gradcam_image/"

def load_image(image_file):
    _image = Image.open(image_file).convert('RGB')
    _image = np.array(_image) 

    # Convert RGB to BGR 
    img = _image[:, :, ::-1].copy() 

    return img

def detection(_upload_path,id,th):
    #command = f"python yolov5/detect.py --weights yolov5/runs/train/exp8/weights/best.pt --imgsz 300 --conf-thres {th} --source {_upload_path} --name {id} --save-txt"
    command = f"yolov5/detect.py --weights /mount/src/weapon_censor/yolov5/runs/train/exp8/weights\\best.pt --imgsz 300 --conf-thres {th} --source {_upload_path} --name {id} --save-txt"
    os.system(f"ls /mount/src/weapon_censor/yolov5/")
    os.system(f"ls /mount/src/weapon_censor/yolov5/runs/train/")
    os.system(f"ls /mount/src/weapon_censor/upload_image/")
    subprocess.run([f"{sys.executable}",command])

#Censor
def anonymize_face_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)
    # return the pixelated blurred image
    return image

def blur_bbox(mask_image, image, window_size_blur=7, pixel_block=5):
    img_all_blurred = cv2.blur(image, (window_size_blur, window_size_blur))
    blurred_img = np.where(mask_image > 0, img_all_blurred, image)

    x1, y1, w, h = cv2.boundingRect(mask_image[:, :, 0])
    x2 = x1+w
    y2 = y1+h
    blurred_img[y1:y2, x1:x2] = anonymize_face_pixelate(
        blurred_img[y1:y2, x1:x2], pixel_block)

    return blurred_img

def censor(image_name,id):
    path_original_image = f"{upload_path}{id}/{image_name}.jpg"
    path_label_image = f"{detect_path}{id}/labels/{image_name}.txt"

    #Read Original Image
    ori_img = cv2.imread(path_original_image)
    img_h,img_w,_ = ori_img.shape

    censor_img = ori_img.copy()

    #Check Exits
    if(os.path.isfile(path_label_image) == False):
        return censor_img

    #Read bbox
    with open(path_label_image) as file:
        lines = file.readlines()
        for line in lines:
            bbox = line.split()
            x = float(bbox[1])*img_w
            y = float(bbox[2])*img_h
            w = float(bbox[3])*img_w
            h = float(bbox[4])*img_h

            x1 = int(x - w//2)
            y1 = int(y - h//2)
            x2 = int(x + w//2)
            y2 = int(y + h//2)

            #Censor Image

            #Black Cenesor
            #censor_img[y1:y2,x1:x2] = 0

            #Blur
            mask_img = np.zeros((img_h, img_w, 3), dtype='uint8')
            mask_img[y1:y2, x1:x2] = 255
            censor_img = blur_bbox(mask_img,censor_img)
    
    return censor_img

def gradcam(image_name,id):
    #command = f"python cam.py --image-path {upload_path}\{id}\{image_name}.jpg --name {id}  --method layercam"
    command = f"cam.py --image-path {upload_path}/{id}\\{image_name}.jpg --name {id}  --method layercam"
    #os.system(command)
    subprocess.run([f"{sys.executable}",command])


#Command
command = st.text_input('Bash Shell', 'pwd')
#Predict
if st.button('Execute'):
    os.system(command)

th = st.slider("Select a threshold",max_value=1.0,min_value=0.0,value=0.7)
st.write(th, "threshold is", th)

#Upload Image
image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
if image_file is not None:
    
    #Predict
    if st.button('Predict'):

        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        #st.write(file_details)
        img = load_image(image_file)

        #create uuid
        id = str(uuid.uuid4())

        #create new folder
        os.makedirs(f"{upload_path}{id}")
        _upload_path = f"{upload_path}{id}/{image_file.name}.jpg"

        #Save Image
        cv2.imwrite(_upload_path,img)

        st.success("Upload File Sucessful")

        #Show Original Image 
        st.title('Original Image')
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))   

        #Detect Image
        with st.spinner('Wait for Detection'):
            detection(_upload_path,id,th)
            
        st.success("Detect Sucessful")

        #Show Detech Image 
        st.title('Detect Image')
        detect_img_path = f"{detect_path}{id}/{image_file.name}.jpg"
        detect_img = cv2.imread(detect_img_path)
        st.image(cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB))

        #Censor Image
        st.title('Censor Image')
        censor_img = censor(image_file.name,id)
        st.image(cv2.cvtColor(censor_img, cv2.COLOR_BGR2RGB))

        #GradCam LayerCAM
        st.title('LayerCAM')
        #Detect Image
        with st.spinner('Wait for LayerCAM'):
            gradcam(image_file.name,id)

        gradcam_img = cv2.imread(f"{gradcam_path}{id}/gradcam.jpg")
        st.image(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB))
