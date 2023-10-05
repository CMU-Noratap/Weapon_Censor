import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

output_path = "Dataset/add_without_weapon/webcam/"
file_name = 503

while True:
    ret, frame = cap.read()
    
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    x,y,_ = frame.shape
    x_offset = (x-480)//2
    y_offset = (y-480)//2

    x1 = x_offset
    y1 = y_offset
    x2 = x - x_offset
    y2 = y - y_offset

    #print(x1,y1,x2,y2)
    #frame = cv2.rectangle(frame, (y1,x1), (y2,x2), (0,0,255), 1)

    frame = frame[y1:y2,x1:x2]
    frame = cv2.resize(frame, None, fx=(300/480), fy=(300/480), interpolation=cv2.INTER_AREA)

    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)

    print(c)

    if c == 27: #ESC
        break
    elif c == 32: # spacebar
        #Save Image
        print(f"Save Pic at {output_path}{file_name}.jpg")
        cv2.imwrite(f"{output_path}{file_name}.jpg",frame)

        #Save Label
        with open(f"{output_path}labels/{file_name}.txt", 'w') as fp:
            pass
        file_name += 1


cap.release()
cv2.destroyAllWindows()