import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array



# LOAD YOLO MODEL
INPUT_WIDTH =  640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/models/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

model = load_model('./static/models/pest_detection_model_2.h5')

def get_detections(img,net):
    # CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_supression(input_image,detections):
    # FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    # NMS
    index = np.array(cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)).flatten()
    
    return boxes_np, confidences_np, index



def drawings(image,boxes_np,confidences_np,index, label):
    # drawings
    if label == '':
        label = 'Pest'
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        
        conf_text = f'{label}: {bb_conf*100:.0f}%'
        


        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        if (y-30) < 0:
            cv2.rectangle(image,(x,y+30),(x+w,y),(255,0,255),-1)
            cv2.putText(image,conf_text,(x,y+10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        else:
            cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
            cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        

    return image,len(index)


# predictions
def yolo_predictions(img,net, label):
    ## step-1: detections
    input_image, detections = get_detections(img,net)
    ## step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
   
    ## step-3: Drawings
    result_img, no_detection = drawings(img,boxes_np,confidences_np,index, label)

    return result_img, no_detection

def object_detection(path,filename):
    # read image
    image = cv2.imread(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    ## class prediction
    label = predict_classes(path,model) 
    result_img, no_detection = yolo_predictions(image,net,label)  
    cv2.imwrite('./static/predict/{}'.format(filename),result_img)
    return no_detection, label


def predict_classes(test_image_path,model ):

    class_labels = {
        0: 'aphids',
        1: 'armyworm',
        2: 'beetle',
        3: 'bollworm',
        4: 'grasshopper',
        5: 'mites',
        6: 'mosquito',
        7: 'sawfly',
        8: 'stem_borer'
    }

    # Image dimensions
    image_width, image_height = 150, 150

    # Number of training and validation samples
    train_samples = 2700
    validation_samples = 450


    # Load and preprocess the test image
    test_image = load_img(test_image_path, target_size=(image_width, image_height))
    test_image_array = img_to_array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    test_image_array = test_image_array / 255.0

    # Perform prediction
    predictions = model.predict(test_image_array)
    predicted_class_indices = np.argmax(predictions, axis=1)
    predicted_classes = [list(class_labels.keys())[idx] for idx in predicted_class_indices]
    confidences = predictions[np.arange(len(predictions)), predicted_class_indices] * 100


    # Loop over the predictions and draw bounding boxes
    for predicted_class, confidence in zip(predicted_classes, confidences):
        # Get the class label and confidence as text
        label = f"{class_labels[predicted_class]}"
        
        
    return label   

    
        
