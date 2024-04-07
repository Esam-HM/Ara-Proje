import os
import cv2
import json
from tqdm import tqdm
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image,ImageDraw
from torchvision import transforms as T
from cjm_pytorch_utils.core import tensor_to_pil
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,ConfusionMatrixDisplay


## General functions.
def create_histogram(classes:list,excel_file:list):
  """
  Create histogram given class names and excel file contains labels. 
  """
  values = [0]*len(classes)
  if len(classes)>2:
      for i in range(0,len(excel_file)):
          id = excel_file.iloc[i,2]
          if id !='' and id !=0:
              values[id-1]+=1
      colors = ["#9467bd","#17becf","#e377c2","#2ca02c","#ff7f0e","#bcbd22"]
      labels = ["1","2","3","4","5","6"]                  ## isimler x eksenine sığmadı.
      title = "Hata tipleri ve sayıları"
  else:
      for i in range(0,len(excel_file)):
          values[excel_file.iloc[i,1]]+=1
      colors = ["#2ca02c","#d62728"]
      labels = ["0","1"]
      title = "Hatalı ve Hatasız görüntülerin sayısı"
  plt.figure(figsize=(7,5))
  plt.bar(labels,values,color = colors,label = classes)
  for i in range(0,len(classes)):
      plt.text(i,values[i]//2,values[i],ha='center',fontsize = 12)            ## Her sutünün görüntü sayısı yazdırma
  plt.legend(loc="upper right")
  plt.title(title)
  plt.xlabel("Sınıf")
  plt.ylabel("Görüntü sayısı")
  plt.tight_layout()
  plt.show()

def create_cm(actual_labels,predicted,classes,title="Confusion Matrix"):
  """
  Create Confusion Matrix.
  """
  cm = confusion_matrix(actual_labels, predicted)
  cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
  cm_display.plot()
  plt.title(title)
  plt.show()

def calculate_metrics(labels,predictions):
  """
  Retuns respctively calculated accuracy, precision, recall and f1 score.  
  """
  f1 = f1_score(labels,predictions)
  precision = precision_score(labels,predictions)
  recall = recall_score(labels,predictions)
  accuracy = accuracy_score(labels,predictions)
  return accuracy, precision, recall, f1

## Image preprocessing
def pad_image(image_path):
  """
  Resize image to 224 height and adjust width to protect aspect ratio.
  Pad image to make 350,350 square image.
  """
  image = Image.open(image_path).convert('RGB')
  new_img = Image.new('RGB',(350,350),color=0)
  image = T.Resize(224,max_size=320)(image)
  x = (new_img.width - image.width) // 2
  y = (new_img.height - image.height) // 2
  new_img.paste(image,(x,y))
  return new_img, new_img.size

## To manipulate data.
def sort_list(lst,isImage:bool=True):
  """
  Sort given list
  isImage used to check for images list or json list is given.
  """
  if not isImage:
    s = 5
  else:
    s = 4
  new_lst = []
  extensions = {}
  for i in range(0,len(lst)):
    id = Path(lst[i]).stem
    extensions[id] = lst[i][-s:]
    new_lst.append(int(id))

  new_lst = sorted(new_lst)

  for i in range(0,len(lst)):
    id = str(new_lst[i])
    new_lst[i]= id + extensions[id]
  return new_lst

def display_images_labels(imgs_lst,image_label_dictionary):
  """
  Return dataframe contains images and corresponding labels.
  """
  return pd.DataFrame({
      "Image" : [img for img in imgs_lst],
      "Label" : [image_label_dictionary[img] for img in imgs_lst]
  })

def create_list(dir,isImage=True,sort=False,fullPath=False):
  """
  Create list given directory.
  """
  lst = os.listdir(dir)

  if sort:
    lst = sort_list(lst,isImage)

  if fullPath:
    for i,img in enumerate(lst):
      path = os.path.join(dir,img)
      lst[i] = path

  return lst

def merge_directories(dir1,dir2,isImage=True,sort=False,fullPath=False):
  """
  Merge two directory.
  """
  list1 = create_list(dir1,isImage,sort,fullPath)
  list2 = create_list(dir2,isImage,sort,fullPath)
  total = list1 + list2
  return total

def get_selected_indexes(indexs:list,elements_list:list):
  """
  Get elements from elemnts list given their indexs. 
  """
  arr = []
  for idx in indexs:
    arr.append(elements_list[idx])
  return arr

def get_values(images,dictionary):
  """
  Given images as keys, returns values from dictionary.
  """
  arr = []
  for img in images:
    arr.append(dictionary[img])
  return arr

def create_dict(keys,values):
  """
  Return created dictionary.
  """
  dictionary = {}
  for i,k in enumerate(keys):
    dictionary[k] = values[i]
  return dictionary

def create_img_label_dict(imgs_lst,labels_lst,column_num:int):
  """
  create dictionary with images as keys and excel labels as values.
  Image list must be sorted before call this function.
  For sorting call sort_image_list function before.
  """
  _dict = {}
  for i,img in enumerate(imgs_lst):
    _dict[img] = labels_lst.iloc[i,column_num]
  return _dict

## For mask.
def get_data_from_json(json_file_path):
  """
  Returns data from given json path.
  """
  with open(json_file_path,'r') as file:
    data = json.load(file)
  return data

def get_polygon_coordinates(json_data):
  """
  Return polygon coordinates from json data.
  """
  index = len(json_data["outputs"]["object"])-1
  polygon_length = int(len(json_data["outputs"]["object"][index]["polygon"])/2) +1
  polygon_coordinates = []
  for i in range(1,polygon_length):
    x = json_data["outputs"]["object"][index]["polygon"][f"x{i}"]
    y = json_data["outputs"]["object"][index]["polygon"][f"y{i}"]
    point = (x, y)
    polygon_coordinates.append(point)

  return polygon_coordinates

def create_mask_with_ImageDraw(image_size,polygon_coords):
  """
  Returns black image contains white mask.
  """
  mask = Image.new('L',image_size,0)
  draw = ImageDraw.Draw(mask)
  draw.polygon(polygon_coords, fill=255)  # fill mask region with white color
  return mask

def draw_overlay_mask_on_org_image(original_image_path,json_file_path):
  """
  Draw overlay mask on original image.
  """
  data = get_data_from_json(json_file_path)

  polygon_coords = get_polygon_coordinates(data)

  # open original image
  org_image = Image.open(original_image_path).convert("RGB")

  mask = create_mask_with_ImageDraw(org_image.size,polygon_coords)
  # Create red overlay on mandibula
  overlay = Image.new("RGBA",org_image.size)
  color = (255,0,0,64) # to make overlay color red
  overlay.paste(color, mask=mask)
  mask_image = Image.alpha_composite(org_image.convert("RGBA"), overlay)
  return mask_image

def draw_mask_with_imageDraw(original_image_path,json_file_path):
  """
  Draw black image contains white mask.
  """
  image = Image.open(original_image_path).convert('RGB')
  data = get_data_from_json(json_file_path)
  polygons = get_polygon_coordinates(data)
  mask = create_mask_with_ImageDraw(image.size,polygons)
  return mask
  
def draw_mask_with_torchvision(original_image_path, json_file_path,class_names=None,withBox:bool=False):
  """
  Draw mask using torchvision modules. 
  """
  data = get_data_from_json(json_file_path)

  polygon_coords = get_polygon_coordinates(data)

  org_image = Image.open(original_image_path).convert('RGB')

  mask = create_mask_with_ImageDraw(org_image.size,polygon_coords)

  transform_to_tensor = T.PILToTensor()
  mask = transform_to_tensor(mask).bool()    # Convert mask to boolean tensor

  box = torchvision.ops.masks_to_boxes(mask)    ## create bounding box

  annotated_tensor = draw_segmentation_masks(
    image=T.PILToTensor()(org_image),
    masks=mask,
    alpha=0.3,
    colors = 'red'
  )
  if withBox:
    # to draw bounding box
    annotated_tensor = draw_bounding_boxes(
      image=annotated_tensor,
      boxes=box,
      labels= class_names[1],
      colors = 'red'
    )
  return tensor_to_pil(annotated_tensor)

def extract_all_masks(images_paths,image_json_dict,dir_name="./masks",save:bool=False):
  """
  Extract all masks and store them as numpy array in list.
  You can save them by make save = True.
  """
  masks = []
  if save:
    dir = Path(dir_name)
    dir.mkdir(parents=True,exist_ok=True)
  loop = tqdm(images_paths)
  for _,file in enumerate(loop):
    image = Image.open(file).convert('RGB')
    image_arr = np.array(image,dtype=np.uint8)
    json_path = image_json_dict[file]
    mask = draw_mask_with_imageDraw(file,json_path)
    mask_arr = np.array(mask,dtype=np.uint8)
    result = cv2.bitwise_and(image_arr,image_arr,mask=mask_arr)
    if save:
      result_to_pil = Image.fromarray(result)
      image_id = Path(file).stem
      exten = file[-4:]
      image_name = f"{dir_name}{image_id}{exten}"
      result_to_pil.save(image_name)
    masks.append(result)
  return masks

def extract_mask(image_path,json_path):
  """
  Extract mask from given image
  Return PIL image mask
  """
  image = Image.open(image_path).convert('RGB')
  mask = draw_mask_with_imageDraw(image_path,json_path)
  image_arr = np.array(image,dtype=np.uint8)
  mask_arr = np.array(mask,dtype=np.uint8)
  masked_img = cv2.bitwise_and(image_arr,image_arr,mask=mask_arr)
  return Image.fromarray(masked_img)

## For controls.
def search(arr,start_index,element):
  """
  Search for elemnt in given Arrays.
  If found returns searched element, else returns -1.
  """
  if element ==-1:
    j=start_index+1
    element = arr[start_index]
  else:
    j=0
  while j<len(arr):
    if element == arr[j]:
      return element
    else:
      j+=1
  return -1

def getNotFoundElements(arr1,arr2):
  """
  Given aar1, returns elements that not found in arr2.
  """
  not_found= []
  for elm in arr1:
    xx = search(arr2,-1,elm)
    if xx ==-1:
      not_found.append(elm)
  return not_found

def findRepeatedElements(arr):
  """
  Returns repeated elements list found in given arr.
  """
  i=0
  repeated = []
  while i<len(arr)-1:
    found = search(arr,i,-1)
    if found!=-1:
      repeated.append(found)
    i+=1
  return repeated