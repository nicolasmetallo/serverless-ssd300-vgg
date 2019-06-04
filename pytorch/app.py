# this import statement is needed if you want to use the AWS Lambda Layer called "pytorch-v1-py36"
# it unzips all of the pytorch & dependency packages when the script is loaded to avoid the 250 MB unpacked limit in AWS Lambda
try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import tarfile
import glob
import time
import logging
import base64
import numpy as np

import boto3
import requests
import PIL

import torch
import sys
sys.path.append('SSD')

from ssd.data.datasets import COCODataset, VOCDataset
from ssd.modeling.predictor import Predictor
from ssd.modeling.vgg_ssd import build_ssd_model

# load the S3 client when lambda execution context is created
s3 = boto3.client('s3')

# set logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# get bucket name from ENV variable
MODEL_BUCKET=os.environ.get('MODEL_BUCKET')
logger.info(f'Model Bucket is {MODEL_BUCKET}')

# get bucket prefix from ENV variable
MODEL_KEY=os.environ.get('MODEL_KEY')
logger.info(f'Model Prefix is {MODEL_KEY}')

def get_configuration(config_file):
    from ssd.config import cfg
    
    cfg.merge_from_file(config_file)
    cfg.freeze()

    logger.info(f"Loaded configuration file {config_file}")
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info(f"Running with config:\n{cfg}")
    
    return cfg
    
def load_model(config_file='SSD/configs/ssd300_voc0712.yaml',
               iou_threshold=0.5, score_threshold=0.5):
    cfg = get_configuration(config_file)
    class_names = VOCDataset.class_names
    global device 
    device = torch.device('cpu')
    model = build_ssd_model(cfg)
    logger.info('Loading model from S3')
    obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
    bytestream = io.BytesIO(obj['Body'].read())
    model.load(bytestream)
    logger.info(f'Loaded weights from {MODEL_KEY}')
    model = model.to(device)
    return Predictor(cfg=cfg,
                          model=model,
                          iou_threshold=iou_threshold,
                          score_threshold=score_threshold,
                          device=device)

def predict(input_object, model):
    """Predicts the class from an input image.

    Parameters
    ----------
    input_object: Tensor, required
        The tensor object containing the image pixels reshaped and normalized.

    Returns
    ------
    Response object: dict
        Returns the predicted class and confidence score.
    
    """
    logger.info("Calling prediction on model")   
    start_time = time.time()
    h, w = input_object.shape[:2]
    output = model.predict(input_object)
    boxes, labels, scores = [o.to(device).numpy() for o in output]

    for bbox in boxes:
        bbox[0] = 0 if bbox[0]<0 else bbox[0]
        bbox[1] = 0 if bbox[1]<0 else bbox[1]
        bbox[2] = w if bbox[2]>w else bbox[2]
        bbox[3] = h if bbox[3]>h else bbox[3]

    logger.info("--- Inference time: %s seconds ---" % (time.time() - start_time))
    response = {}

    try:
        roi = max([(a,b,c) for a,b,c in zip(labels, scores, boxes) if a == 15], default=0)
        if roi[2].size == 4:
            xmin, ymin, xmax, ymax = [roi[2][x] for x in range(4)]
            response['is_person'] = True
            response['confidence'] = float(roi[1])
            response['xmin'] = int(xmin)
            response['ymin'] = int(ymin)
            response['xmax'] = int(xmax)
            response['ymax'] = int(ymax)
        else:
            response['is_person'] = False
            response['confidence'] = 0.0
    except ValueError:
        response['is_person'] = False
        response['json'] = json.dumps(dict(
            boxes=boxes.tolist(),
            labels=labels.tolist(),
            scores=scores.tolist(),
            ))
    
    return response

# load the model when lambda execution context is created
model=load_model()
    
def input_fn(request_body):
    """Pre-processes the input data from JSON to PyTorch Tensor.

    Parameters
    ----------
    request_body: dict, required
        The request body submitted by the client. Expect an entry 'url' containing a URL of an image to classify.

    Returns
    ------
    PyTorch Tensor object: Tensor
    
    """    
    logger.info("Getting input base64 image data to np.array")
    if isinstance(request_body, str):
        request_body = json.loads(request_body)
    img = PIL.Image.open(io.BytesIO(base64.b64decode(request_body['data'])))
    return np.array(img)
    
def lambda_handler(event, context):
    """Lambda handler function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    print("Starting event")
    logger.info(event)
    print("Getting input object")
    input_object = input_fn(event['body'])
    print("Calling prediction")
    response = predict(input_object, model)
    print("Returning response")
    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }
