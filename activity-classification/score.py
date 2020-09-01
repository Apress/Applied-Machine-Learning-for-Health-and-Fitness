
import torch
import torch.nn as nn
from torchvision import transforms
import json
import base64
from io import BytesIO
from PIL import Image
import os
import pickle

from azureml.core.model import Model

def transform(image_file):
    t = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225])])
    image = Image.open(image_file)
    image = t(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image

def decode_base64_to_img(base64_string):
    base64_image = base64_string.encode('utf-8')
    decoded_img = base64.b64decode(base64_image)
    return BytesIO(decoded_img)

def init():
    global model, classes
    #model_path = Model.get_model_path('activities')
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'models', 'activities.pkl')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()
    #pkl_file = open(os.path.join(model_path,'class_names.pkl'), 'rb')
    #classes = pickle.load(pkl_file)
    #pkl_file.close() 
    classes = ['surfing','tennis']

def run(input_data):
    image = decode_base64_to_img(json.loads(input_data)['data'])
    image = transform(image)

    output = model(image)

    softmax = nn.Softmax(dim=1)
    pred_probs = softmax(model(image)).detach().numpy()[0]
    index = torch.argmax(output, 1)

    result = json.dumps({"label": classes[index], "probability": str(pred_probs[index])})
    return result
