import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self , images , masks):
        self.imgs = images
        self.masks = masks
     
    # Modelin hangi parametreye bağlı çalıştığını anlamak için 
    # Dataloader return değeri çok önemli.
    
    def __getitem__(self , idx):

        img = Image.open("C:/Users\emirb\OneDrive\Desktop\BLM3010\Project\dataset\images/" + self.imgs[idx]).convert("L")
        mask = Image.open("C:/Users\emirb\OneDrive\Desktop\BLM3010\Project\dataset\masks/" + self.masks[idx])
        mask = np.array(mask)

# FIXME maske değerleri hatalı olabilir. 0 ve 255 arasında değerler var.        
        mask[mask < 123] = 0
        mask[mask > 123] = 255

        # Maskede kaç adet obje var? Background hariç.
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        
        if num_objs != 1:
            print(f"num_objs != 1 for file {self.masks[idx]}")
            
        masks = np.zeros((num_objs , mask.shape[0] , mask.shape[1]))
        # Her obje için farklı bir maske.
        for i in range(num_objs):
            masks[i][mask == 255] = True
                
        # Her obje için farklı bir kutu.
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin , ymin , xmax , ymax])

        # Tüm veriyi tensora çeviriyoruz.
        boxes = torch.as_tensor(boxes , dtype = torch.float32)
        labels = torch.ones((num_objs,) , dtype = torch.int64)
        masks = torch.as_tensor(masks , dtype = torch.uint8)
        
        # Verimizi modelin kabul edeceği dictionary içine yerleştiriyoruz 
        # Bu veriler görseldeki tüm yayalar için label, kutu ve maskeler.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
                
        # Görselin kendisini ve verileri tuttuğumuz dictionary değişkenini döndürüyoruz.
        return T.ToTensor()(img), target
    
    def __len__(self):
        return len(self.imgs)