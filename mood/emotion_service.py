from torchvision.transforms import ToPILImage
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision import transforms
from mood import EmotionNet
import torch.nn.functional as nnf
from mood import utils
import numpy as np
import torch
import time
import cv2
import json


model_pth = 'model/model_50-40.pth'
prototxt = 'model/deploy.prototxt'
caffemodel = 'model/res10_300x300_ssd_iter_140000_fp16.caffemodel'
confidence_thresh = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
                4: "neutral", 5: "sad", 6: "surprise"}

# initialize a list of preprocessing steps to apply on each image during runtime
data_transform = transforms.Compose([
    ToPILImage(),
    Grayscale(num_output_channels=1),
    Resize((48, 48)),
    ToTensor()
])

# load the Models
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
model = EmotionNet(num_of_channels=1, num_of_classes=len(emotion_dict))
model_weights = torch.load(model_pth)
model.load_state_dict(model_weights)
model.to(device)
model.eval()


# Helper code
def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return ((np.array(image.getdata())
            .reshape((im_height, im_width, 3)))
            .astype(np.uint8))


def get_emotions(image, threshold, container_size):
    c_width, c_height = container_size

    image = load_image_into_numpy_array(image)
    frame = utils.resize_image(image, width=c_width, height=c_height)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))

    net.setInput(blob)
    detections = net.forward()

    outputs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            face = frame[start_y:end_y, start_x:end_x]
            face = data_transform(face)
            face = face.unsqueeze(0)
            face = face.to(device)

            predictions = model(face)
            prob = nnf.softmax(predictions, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            top_p, top_class = top_p.item(), top_class.item()

            emotion_prob = [p.item() for p in prob[0]]
            emotion_value = list(emotion_dict.values())
            face_emotion = emotion_dict[top_class]

            output = {
                'emotion_prob': emotion_prob,
                'emotion_value': emotion_value,
                'face_emotion': face_emotion,
                'box': [int(start_x), int(start_y), int(end_x-start_x), int(end_y-start_y)],
                'top_p': round(top_p,2)
            }

            outputs.append(output)

    return json.dumps(outputs)

