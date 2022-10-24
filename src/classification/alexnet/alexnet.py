import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import AlexNet_Weights


def classify(image):
    """

    Author: https://pytorch.org/hub/pytorch_vision_alexnet/
    :param image:
    :return:
    """
    input_image = Image.open(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        alex_net.to('cuda')

    with torch.no_grad():
        output = alex_net(input_batch)

    return output.tolist()[0]


# instantiate the model
alex_net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
alex_net.classifier = nn.Sequential(*[alex_net.classifier[i] for i in range(1)])
alex_net.eval()
