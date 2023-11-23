import ultralytics
import torch
device = torch.cuda.get_device_name(0)
from ultralytics import YOLO
import time
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np
from util import get_car, read_license_plate, write_csv
import os



from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.deep_sort.sort.tracker import Tracker

deep_sort_weights = 'deep_sort/deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=25)

video_path = 'demo.mp4'
count=0
nov=set() 

class_names= {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',}


cap = cv2.VideoCapture(video_path)

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

frames = []
vehicles = [2, 3, 5, 7]
unique_track_ids = set()

i = 0
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()
# resultss = []
frame_nmr = -1
ret = True
model = YOLO("yolov8x.pt") 
license_plate_detector = YOLO('best.pt')


def resize_frame(frame,scale=0.25):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv2.resize(frame,dimensions,cv2.INTER_AREA)


while ret:
    frame_nmr += 1

    # if(frame_nmr>100):
    #     break
    ret, frame = cap.read()

    # frame = resize_frame(frame)

    if ret:
        print(frame_nmr)
        # resultss[frame_nmr] = {}


        results = model(frame, device=0, classes=[2,3,5], conf=0.8)

        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            probs = result.probs  # Class probabilities for classification outputs
            cls = boxes.cls.tolist()  # Convert tensor to list
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh  # box with xywh format, (N, 4)
            for class_index in cls:
                class_name = class_names[int(class_index)]
                #print("Class:", class_name)


        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float)

        
        tracks = tracker.update(bboxes_xywh, conf,pred_cls, frame)
        for track in tracks:
             x3, y3, x4, y4,car_id,class_id=track
             if car_id not in nov:
                 nov.add(car_id)
             Vehicle_info=str(class_names[class_id])+","+str(car_id)

             cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
             cv2.putText(frame,Vehicle_info,(x3+5,y3-10),cv2.FONT_HERSHEY_DUPLEX ,1,(255,0,0),0)
             count=len(nov)
             cv2.putText(frame,str(count),(61,146),cv2.FONT_HERSHEY_DUPLEX,5,(255,255,255),3)
        # print('tracks',tracks)

        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            # print('plate',x1, y1, x2, y2, score)
            xcar1, ycar1, xcar2, ycar2, car_id,class_id= get_car(license_plate, tracks)
            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                license_plate_text  = read_license_plate(license_plate_crop_gray)
                print('car_id :',car_id,' class_id :',class_id,' lp :',license_plate_text)
                # cv2.putText(frame,str(license_plate_text),(xcar1+75 ,ycar1-10),cv2.FONT_HERSHEY_DUPLEX ,1,(255,255,0),1)

    
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
                # if license_plate_text is not None:
                #     resultss[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                #                                   'license_plate': {'bbox': [x1, y1, x2, y2],
                #                                                     'text': license_plate_text,
                #                                                     'bbox_score': score,
                #                                                     'text_score': license_plate_text_score}}


# print(resultss)
                # print('license_plate_text',license_plate_text)

        
#         for track in tracker.tracker.tracks:
#             track_id = track.track_id
#             hits = track.hits
#             x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
#             w = x2 - x1  # Calculate width
#             h = y2 - y1  # Calculate height

#             # Set color values for red, blue, and green
#             red_color = (0, 0, 255)  # (B, G, R)
#             blue_color = (255, 0, 0)  # (B, G, R)
#             green_color = (0, 255, 0)  # (B, G, R)

#             # Determine color based on track_id
#             color_id = track_id % 3
#             if color_id == 0:
#                 color = red_color
#             elif color_id == 1:
#                 color = blue_color
#             else:
#                 color = green_color

#             cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

#             text_color = (255, 255, 255)  # Black color for text
#             cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)

#             # Add the track_id to the set of unique track IDs
#             unique_track_ids.add(track_id)

#         # Update the person count based on the number of unique track IDs
#         person_count = len(unique_track_ids)

#         # Update FPS and place on frame
#         current_time = time.perf_counter()
#         elapsed = (current_time - start_time)
#         counter += 1
#         if elapsed > 1:
#             fps = counter / elapsed
#             counter = 0
#             start_time = current_time

#         # Draw person count on frame
#         cv2.putText(og_frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         # Append the frame to the list
#         frames.append(og_frame)

#         # Write the frame to the output video file
#         out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

#         # Show the frame


cap.release()
out.release()
cv2.destroyAllWindows()

