# facebook: https://www.facebook.com/salar.brefki/
# instagram: https://www.instagram.com/salarbrefki/

import cv2
####### From Video or Camera #######
def Camera(ip):
   cam = cv2.VideoCapture(ip)

   cam.set(3, 740)
   cam.set(4, 580)

   classNames = []
   classFile = 'coco.names'

   with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')

   configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
   weightpath = 'frozen_inference_graph.pb'

   net = cv2.dnn_DetectionModel(weightpath, configPath)
   net.setInputSize(320 , 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)
   counter=0

   while True:
      success, img = cam.read()
      classIds, confs, bbox = net.detect(img, confThreshold=0.7)

      if len(classIds) !=0:
         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId==1 or classId==3:
               counter+=1
               cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
               cv2.putText(img, str(classNames[classId-1])+str(counter), (box[0] + 10, box[1] + 20), 
                           cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
               
               print(counter)
            else:
               continue
            

         print(classIds)
      counter=0
      cv2.imshow('Output', img)
      cv2.waitKey(1)==ord("q")
######################################


## Call ImgFile() Function for Image Or Camera() Function for Video and Camera
# ImgFile()
Camera(0)
