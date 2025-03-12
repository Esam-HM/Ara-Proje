import io
import cv2
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from cjm_pytorch_utils.core import tensor_to_pil
from transformers import DeiTForImageClassification
from torchvision.utils import draw_segmentation_masks
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

 
## Loads mask r-cnn model.
def create_model(model_path,device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    return model

## Draw a red overlay on masked region.
def draw_overlay_mask(image_path,result):
    mask = result[0]['masks'][0] > 0.5
    image = Image.open(image_path).convert('RGB')
    annotated_tensor = draw_segmentation_masks(
    image=T.PILToTensor()(image),
    masks=mask.bool(),
    alpha=0.3,
    colors = 'red')
    return tensor_to_pil(annotated_tensor)

## Mask a mandibular in given dental x-ray image.
def mask_rcnn_model_prediction(image,model,device):
    img = Image.open(image).convert('RGB')
    tensor = T.ToTensor()(img)
    model.eval()
    with torch.no_grad():
        res = model([tensor.to(device)])
    
    return res

## Extract the mandibular from dental x-ray image.
def extract_mask(image_path,tensor):
  """
  Extract mask from given image
  Return PIL image mask
  """
  image = Image.open(image_path).convert('RGB')
  mask = tensor[0]['masks'][0,0] > 0.5
  image_arr = np.array(image,dtype=np.uint8)
  mask_arr = np.array(mask,dtype=np.uint8)
  masked_img = cv2.bitwise_and(image_arr,image_arr,mask=mask_arr)
  return Image.fromarray(masked_img)

## Converts image to bytes array object.
def getBytesObject(image):
    bytesIO = io.BytesIO()
    image.save(bytesIO,format='PNG')
    bytesIO.seek(0)
    return bytesIO

## Combine models result.
def combine(rontgen_probs,mandibula_probs):
    mand_pred = np.argmax(mandibula_probs)
    rontgen_pred = np.argmax(rontgen_probs)

    if mand_pred == rontgen_pred:
        return mand_pred
    else:
        if mandibula_probs[mand_pred] > rontgen_probs[rontgen_pred]:
            return mand_pred
        else:
            return rontgen_pred
    
## Predict function used to classify image given model.
def predict(model,device,imgPath):
    transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    res = {}
    with torch.no_grad():
        image = Image.open(imgPath).convert('RGB')
        image = transform(image)
        output = model(image.unsqueeze(0).to(device)).logits
        prediction = output.argmax(dim=1, keepdim=True).squeeze().cpu().numpy()
        prob = F.softmax(output, dim=1).cpu().numpy().squeeze()
        res = {'prediction' : prediction, 'prob' : prob, }
    return res

## To load classification models given model path.
def initModel(id2label,label2id,device,modelPath):
    model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224",
                                                         label2id=label2id,
                                                         id2label=id2label,
                                                         ignore_mismatched_sizes=False).to(device)
    model.load_state_dict(torch.load(modelPath,map_location=device))
    return model

def initModelArgs(classes):
    label2id = {}
    id2label = {}
    id = 0
    for cls in classes:
        label2id[cls] = id
        id2label[id] = cls
        id+=1
    return label2id, id2label

## Get true labels from Total_labels excel file.
def getTrueLabels(excel_path, file_name):
    labelsFile = pd.read_excel(excel_path,header=None,keep_default_na=False)
    file_name = Path(file_name).stem
    for i in range(labelsFile.shape[0]):
        if(str(labelsFile.iloc[i,0])==file_name):
            return labelsFile.iloc[i,1], labelsFile.iloc[i,2]
    return -1,-1
      
      
      



   
      
      
      
      