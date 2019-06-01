# SSD implementation in PyTorch + AWS Lambda
---
The idea of this repo is to create a Lambda function that takes binary input (image) and returns a JSON. For my use case, I'm looking to find if there's a person in the picture, and make a crop from the image out of that. If there's no person in the image, then return everything that you have found (i.e. boxes, labels, confidence scores) for the remaining 21 classes (VOC dataset).

## Pre-requisities
You should have:
- Docker CE
- PyTorch 1.0 (https://pytorch.org/) (optional if you don't want to test locally).
- Amazon AWS account
- AWS CLI + AWS SAM CLI

## Installation
The first step is to clone this repo.
```
git clone https://github.com/nicolasmetallo/serverless-ssd300-vgg/
```
And now `cd` into `pytorch` on your Terminal and run:
```
git clone https://github.com/lufficc/SSD
```
Download the pre-trained weights from that repo using [this link](https://github.com/lufficc/SSD/releases/download/v1.0.1/ssd300_voc0712_mAP77.83.pth) into your local filesystem. Don't save them to your working directory otherwise it will get packaged into your lambda function.

## AWS stuff
The first thing you need to do is install the AWS CLI with [these steps](https://docs.aws.amazon.com/cli/latest/userguide/install-macos.html) if you are on a Mac.

After you have finished that, install the AWS SAM CLI [through these instructions](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install-mac.html).

Now we need to setup your AWS credentials. Go to your AWS console and click on `My Security Credentials` and then go to `Access Keys` and generate the keys. Download these keys to your machine.

Now go to the terminal and type

```
aws configure
```

Add your Access Key, Secret Key, region (i.e. `us-east-1`) and output (i.e. `json`).

Now we need to upload the pre-trained weights to S3 as we will download them later. Open the terminal and create a new bucket:

```
aws s3api create-bucket --bucket serverless-ssd300-vgg --region us-east-1
```

And copy the weights:

```
aws s3 cp ssd300_voc0712_mAP77.83.pth s3://serverless-ssd300-vgg/ssd300_voc0712_mAP77.83.pth
```

## Packaging and deployment

Next, run the following command to package our Lambda function to S3:

```
sam package \
    --output-template-file packaged.yaml \
    --s3-bucket serverless-ssd300-vgg
```

The output is 

```
Uploading to 1cfb305546dcf4a4c711999764d577b9  46733743 / 46733743.0  (100.00%)
Successfully packaged artifacts and wrote output template to file packaged.yaml.
```

Next, the following command will create a Cloudformation Stack and deploy your SAM resources. You will need to override the default parameters for the bucket name and object key. This is done by passing the `--parameter-overrides` option to the `deploy` command as shown below. Because we have some external libraries that are not included in the previous Lambda Layer, we need to `sam build` before we deploy. We run

```
sam build -u && sam deploy \
    --template-file packaged.yaml \
    --stack-name serverless-ssd300-vgg \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides BucketName=serverless-ssd300-vgg ObjectKey=ssd300_voc0712_mAP77.83.pth
```

And the output is

```
Build Succeeded

Built Artifacts  : .aws-sam/build
Built Template   : .aws-sam/build/template.yaml

Commands you can use next
=========================
[*] Invoke Function: sam local invoke
[*] Package: sam package --s3-bucket <yourbucket>
    
Running PythonPipBuilder:ResolveDependencies
Running PythonPipBuilder:CopySource

Waiting for changeset to be created..
Waiting for stack create/update to complete
Successfully created/updated stack - serverless-ssd300-vgg
```

After deployment is complete you can run the following command to retrieve the API Gateway Endpoint URL:

```
aws cloudformation describe-stacks \
    --stack-name serverless-ssd300-vgg \
    --query 'Stacks[].Outputs[?OutputKey==`PyTorchApi`]' \
    --output table

```

The output is:

```
------------------------------------------------------------------------------------------------------------------------------------------------------------
|                                                                      DescribeStacks                                                                      |
+---------------------------------------------------------------+-------------+----------------------------------------------------------------------------+
|                          Description                          |  OutputKey  |                                OutputValue                                 |
+---------------------------------------------------------------+-------------+----------------------------------------------------------------------------+
|  API Gateway endpoint URL for Prod stage for PyTorch function |  PyTorchApi |   https://53f8w4fcua.execute-api.us-east-1.amazonaws.com/Prod/invocations/      |
+---------------------------------------------------------------+-------------+----------------------------------------------------------------------------+

```

## Cleanup (in case you messed it up)

In order to delete our Serverless Application recently deployed you can use the following AWS CLI Command:

```
aws cloudformation delete-stack --stack-name serverless-ssd300-vgg

```

## Run Inference

I usually start a jupyter lab instance on the Terminal

```
jupyter lab

```

And then run the `test.ipynb` notebook. The important bit is this:

```
now=time.time()
url = 'https://adeo3k2yxa.execute-api.us-east-1.amazonaws.com/Prod/invocations/'
payload = {'data': b64_im.decode('utf-8')}
    
headers = {'content-type': 'application/json'}

r = requests.post(url, json=payload, headers=headers)
print('proc time: {} seconds and response is: {}'.format(time.time()-now, r))

```

