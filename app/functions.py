import io
import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torchvision
from PIL import Image
from cjm_pytorch_utils.core import tensor_to_pil
from torchvision.utils import draw_segmentation_masks
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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

def draw_overlay_mask(image_path,result):
    mask = result[0]['masks'][0] > 0.5
    image = Image.open(image_path).convert('RGB')
    annotated_tensor = draw_segmentation_masks(
    image=T.PILToTensor()(image),
    masks=mask.bool(),
    alpha=0.3,
    colors = 'red')
    return tensor_to_pil(annotated_tensor)


def mask_rcnn_model_prediction(image,model,device):
    img = Image.open(image).convert('RGB')
    tensor = T.ToTensor()(img)
    model.eval()
    with torch.no_grad():
        res = model([tensor.to(device)])
    
    return res


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

def getBytesObject(image):
    bytesIO = io.BytesIO()
    image.save(bytesIO,format='PNG')
    bytesIO.seek(0)
    return bytesIO