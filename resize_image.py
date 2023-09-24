import cv2
import glob

input_path = "C:/Users/PC_ML/Desktop/Gun_Knife_Censor/Dataset/add_without_weapon/raw/*"
output_path = "C:/Users/PC_ML/Desktop/Gun_Knife_Censor/Dataset/add_without_weapon/resize/"

list_image_file = glob.glob(input_path)

target_h = 300
target_w = 300

def crop_resize(image):
    h, w , _ = image.shape
    min_size = min(h,w)
    
    x0 = (w//2) - (min_size//2)
    x1 = (w//2) + (min_size//2)

    y0 = (h//2) - (min_size//2)
    y1 = (h//2) + (min_size//2)

    crop_img = image[y0:y1,x0:x1]
    resize_img = cv2.resize(crop_img, (target_h,target_w), interpolation = cv2.INTER_AREA)
    return resize_img

for img_file in list_image_file:
    print(img_file)
    file_name = img_file.split("\\")[1].split(".")[0]

    # Load the image
    image = cv2.imread(img_file)
    
    # Display the image
    cv2.imshow("Original", image)

    #Crop & Resize
    new_img = crop_resize(image)

    # Display the image
    cv2.imshow("New", new_img)

    #Save image
    file_target = f"{output_path}{file_name}.jpg"
    cv2.imwrite(file_target, new_img)

    key = cv2.waitKey(0) 
    cv2.destroyAllWindows()
    if(key == ord('q')):
        break