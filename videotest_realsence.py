import cv2
import numpy as np
from keras.models import model_from_json
import pyrealsense2 as rs
import keras.utils as image


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")



##############test#########################

print("reset start")
ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()
print("reset done")


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

#################test######################


# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture("D:\\Work\\Project AppS597\\Emotion_detection_with_CNN-main\\sad.mp4")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    # color_image = cv2.resize(frame, (1280, 720))
    # # color_image = cv2.resize(frame, (720, 1280))
    # if not ret:
    #     break

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue


    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    color_image = np.asanyarray(color_frame.get_data())
    gray_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(color_image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_img[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#########################################
        # cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        # roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        # roi_gray = cv2.resize(roi_gray, (48, 48))
        # img_pixels = image.img_to_array(roi_gray)
        # cropped_img = np.expand_dims(img_pixels, axis=0)
        # img_pixels /= 255
#################################################
        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(color_image, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    ##############test#########################

    # if labels[-1] == predicted_emotion:
    #     true = true + 1
    # else:
    #     false = false + 1

    #################test######################

    cv2.imshow('Emotion Detection', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
