import cv2
import numpy as np
import os, io
from google.cloud import vision
from google.cloud.vision import types
import threading
import asyncio
import time

globalFrame = None
location = None
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()

def localize_objects(img):
    global location
    global globalFrame

    success, encoded_image = cv2.imencode('.png', img)
    content2 = encoded_image.tobytes()
    image = vision.types.Image(content=content2)

    objects = client.object_localization(
        image=image).localized_object_annotations

    print('Number of objects found: {}'.format(len(objects)))
    objectsPosition = []
    for object_ in objects:

        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')

        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))
            objectsPosition.append([vertex.x, vertex.y])

    location = objectsPosition
    threading.Timer(0.5, localize_objects, args=(globalFrame,)).start()



def main():
    global location
    global globalFrame

    started = False
    cap = cv2.VideoCapture("http://dl86.y2mate.com/?file=M3R4SUNiN3JsOHJ6WWQ2a3NQS1Y5ZGlxVlZIOCtyZ0hsOGMxeVJFMUNMbFp0Y1lLMmZDbE1kb0VDcUlaekptMkVNcGQrem1UWDlXY2V6K0J0NHNqQ1RiVzFKMFF0aTNCK29BbkV1TitVMTI5ek1Yb3V3SllxeVRLYU1MaU9iOVFXQzVpczNGY2duYkxuT0dhdEFXczlYdWtxMGk4ZkNVZXVtWk9iOVBKL29wY3dHdk9LcVhDM29CUjZIUzV1dDhiL09uUHVRRGl4dnc5dDlFK0V3OTNmSUpTMTV2ajF1TFp0bmdjam8wWnlGNmJyZWV3QjQwaEhiR1hkekJQTUNjZTF2L25YUWdiOENnUjZtSzMrS2dhNUc4TVRxMW01bUMzeHVUelNST2ZlUGV2Vm9HMUZibnQvcFhwclBSeDZ3ZkNxdXJDa281cHBsWDJHSkd4RjRoZS94cDI3L2JhbzhrdTBsK3IzeHNUaitWVWxSNmtJeHh2R3M0YmNERWNNWkVFRVgwPQ%3D%3D")
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1920, 1080))
        if ret == True:
            globalFrame = frame
        if location:
            for i in range(3, len(location), 4):
                if i - 1 > len(location):
                    break
                cv2.rectangle(frame, (int(location[i-3][0]*1920), int(location[i-3][1]*1080)), (int(location[i-1][0]*1920), int(location[i-1][1]*1080)), (255,0,0), 2)
        else:
            if not started:
                localize_objects(frame)
                started = True

        if ret == True:
            cv2.imshow('Video',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.0166)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
