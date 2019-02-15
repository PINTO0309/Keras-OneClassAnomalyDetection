import cv2
import time
import os
import sys
import numpy as np
import argparse
import platform
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.models import model_from_json
from keras import backend as K
from openvino.inference_engine import IENetwork, IEPlugin

def main(camera_FPS, camera_width, camera_height, inference_scale, threshold, device):

    path = "pictures/"
    if not os.path.exists(path):
        os.mkdir(path)

    model_path = "OneClassAnomalyDetection-RaspberryPi3/DOC/model/" 
    if os.path.exists(model_path):
        # LOF
        print("LOF model building...")
        x_train = np.loadtxt(model_path + "train.csv",delimiter=",")

        ms = MinMaxScaler()
        x_train = ms.fit_transform(x_train)

        # fit the LOF model
        clf = LocalOutlierFactor(n_neighbors=5)
        clf.fit(x_train)

        # DOC
        print("DOC Model loading...")
        model_xml="irmodels/tensorflow/FP32/weights.xml"
        model_bin="irmodels/tensorflow/FP32/weights.bin"
        net = IENetwork(model=model_xml, weights=model_bin)
        plugin = IEPlugin(device=device)
        if device == "CPU":
            if platform.processor() == "x86_64":
                plugin.add_cpu_extension("lib/x86_64/libcpu_extension.so")
        exec_net = plugin.load(network=net)
        input_blob = next(iter(net.inputs))
        print("loading finish")
    else:
        print("Nothing model folder")
        sys.exit(0)

    base_range = min(camera_width, camera_height)
    stretch_ratio = inference_scale / base_range
    resize_image_width = int(camera_width * stretch_ratio)
    resize_image_height = int(camera_height * stretch_ratio)

    if base_range == camera_height:
        crop_start_x = (resize_image_width - inference_scale) // 2
        crop_start_y = 0
    else:
        crop_start_x = 0
        crop_start_y = (resize_image_height - inference_scale) // 2
    crop_end_x = crop_start_x + inference_scale
    crop_end_y = crop_start_y + inference_scale

    fps = ""
    message = "Push [p] to take a picture"
    result = "Push [s] to start anomaly detection"
    flag_score = False
    picture_num = 1
    elapsedTime = 0
    score = 0
    score_mean = np.zeros(10)
    mean_NO = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, camera_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    time.sleep(1)

    while cap.isOpened():
        t1 = time.time()

        ret, image = cap.read()

        if not ret:
            break

        image_copy = image.copy()

        # prediction
        if flag_score == True:
            prepimg = cv2.resize(image, (resize_image_width, resize_image_height))
            prepimg = prepimg[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            prepimg = np.array(prepimg).reshape((1, inference_scale, inference_scale, 3))
            prepimg = prepimg / 255
            prepimg = prepimg.transpose((0, 3, 1, 2))

            exec_net.start_async(request_id = 0, inputs={input_blob: prepimg})
            exec_net.requests[0].wait(-1)
            outputs = exec_net.requests[0].outputs["Reshape_"]
            outputs = outputs.reshape((len(outputs), -1))
            outputs = ms.transform(outputs)
            score = -clf._decision_function(outputs)

        # output score
        if flag_score == False:
            cv2.putText(image, result, (camera_width - 350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            score_mean[mean_NO] = score[0]
            mean_NO += 1
            if mean_NO == len(score_mean):
                mean_NO = 0
                
            if np.mean(score_mean) > threshold: #red if score is big
                cv2.putText(image, "{:.1f} Score".format(np.mean(score_mean)),(camera_width - 230, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            else: # blue if score is small
                cv2.putText(image, "{:.1f} Score".format(np.mean(score_mean)),(camera_width - 230, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
              
        # message
        cv2.putText(image, message, (camera_width - 285, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, fps, (camera_width - 164, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0 ,0), 1, cv2.LINE_AA)

        cv2.imshow("Result", image)
            
        # FPS
        elapsedTime = time.time() - t1
        fps = "{:.0f} FPS".format(1/elapsedTime)

        # quit or calculate score or take a picture
        key = cv2.waitKey(1)&0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            cv2.imwrite(path + str(picture_num) + ".jpg", image_copy)
            picture_num += 1
        if key == ord("s"):
            flag_score = True

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfps","--camera_FPS",dest="camera_FPS",type=int,default=30,help="USB Camera FPS. (Default=30)")
    parser.add_argument("-cwd","--camera_width",dest="camera_width",type=int,default=320,help="USB Camera Width. (Default=320)")
    parser.add_argument("-cht","--camera_height",dest="camera_height",type=int,default=240,help="USB Camera Height. (Default=240)")
    parser.add_argument("-sc","--inference_scale",dest="inference_scale",type=int,default=96,help="Inference scale. (Default=96)")
    parser.add_argument("-th","--threshold",dest="threshold",type=int,default=2.0,help="Threshold. (Default=2.0)")
    parser.add_argument("-d","--device",dest="device",default="CPU",help="Device. CPU/GPU (Default=CPU) GPU=Intel HD Graphics.")
    args = parser.parse_args()
    camera_FPS = args.camera_FPS
    camera_width = args.camera_width
    camera_height = args.camera_height
    inference_scale = args.inference_scale
    threshold = args.threshold
    device = args.device

    main(camera_FPS, camera_width, camera_height, inference_scale, threshold, device)
