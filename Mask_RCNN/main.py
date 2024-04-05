import numpy as np
import os
from ourDataset import CustomDataset
from tqdm import tqdm
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
from PIL import Image

def custom_collate(data):
    return data

def create_model(model_path):
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes=2)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    return model
        
def extract_mask(pred):
    
    masks = pred[0]["masks"]
    mask = masks[0, 0] > 0.5
    mask = mask.cpu().numpy().astype("uint8") * 255
    
    return mask

def change_color(picture, width, height, ex_color, new_color):
    # Process every pixel
    for x in range(width):
        for y in range(height):
            current_color = picture.getpixel( (x,y) )
            if current_color == ex_color:
                picture.putpixel( (x,y), new_color)
    return picture


def get_overlapped_img(img, mask):
    mask = Image.fromarray(mask)
    width, height = mask.size
    # Convert the white color (for blobs) to magenta
    mask_colored = change_color(mask, width, height, (255, 255, 255), (186,85,211))
    # Convert the black (for background) to white --> important to make a good overlapping
    mask_colored = change_color(mask_colored, width, height, (0, 0, 0), (255,255,255))

    return cv2.addWeighted(np.array(img),0.4,np.array(mask_colored),0.3,0)

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    print(f"os.environ: {os.environ['KMP_DUPLICATE_LIB_OK']}")
    
    image_path = "C:/Users\emirb\OneDrive\Desktop\BLM3010\Project\dataset/images"
    mask_path = "C:/Users\emirb\OneDrive\Desktop\BLM3010\Project\dataset/masks"
    
    images = sorted(os.listdir(image_path))
    masks = sorted(os.listdir(mask_path))
    
    # Modelin oluşturulması.
    # num_classes default 2
    
    model = create_model()
    
    # Train-Test Split İşlemi
    num = int(0.9 * len(images))
    num = num if num % 2 == 0 else num + 1
    train_imgs_inds = np.random.choice(range(len(images)), num, replace = False)
    val_imgs_inds = np.setdiff1d(range(len(images)), train_imgs_inds)
    
    # Tüm görsellerin ve maskelerin numpy array formatına dönüştürülmesi.
    
    train_imgs = np.array(images)[train_imgs_inds]
    val_imgs = np.array(images)[val_imgs_inds]
    train_masks = np.array(masks)[train_imgs_inds]
    val_masks = np.array(masks)[val_imgs_inds]
    
    del images, masks
    
    # Train ve Validation Dataloaderlarının tanımlanması.
    # Number of workers parametresi (=2) hataya sebep olduğu için kaldırıldı.
    # Default batch_size was 2
    b_size = 64
#    num_workers = multiprocessing.cpu_count() - 2
    
    train_data_loader = torch.utils.data.DataLoader(CustomDataset(train_imgs, train_masks), 
                                     batch_size = b_size, 
                                     shuffle = True, 
#                                     num_workers = num_workers,
                                     collate_fn = custom_collate,
                                     pin_memory = True if torch.cuda.is_available() else False)
    val_data_loader = torch.utils.data.DataLoader(CustomDataset(val_imgs, val_masks), 
                                     batch_size = b_size, 
                                     shuffle = True, 
#                                     num_workers = num_workers,
                                     collate_fn = custom_collate,
                                     pin_memory = True if torch.cuda.is_available() else False)
    
    # Modelin çalışacağı aygıtın belirlenmesi.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Model parametrelerinin belirlenmesi ve bu parametreler de kullanılarak optimizer oluşturulması.
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Default lr was 0.005 weight_decay was 0.0005
    #optimizer = torch.optim.AdamW(params, lr=0.1, weight_decay=0.0005)
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    all_train_losses = []
    all_val_losses = []
    flag = False
    epochs = 50
    
    for epoch in tqdm(range(epochs)):
        train_epoch_loss = 0
        val_epoch_loss = 0
        # Model train konfigürasyonuna alınıyor.
        model.train()
        for i, dt in enumerate(train_data_loader):
            # i enumerate iterator, dt ise __getitem__ methodundan dönen Tensor ve Dictionary.
            imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
            targ = [dt[0][1] , dt[1][1]]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
            loss = model(imgs , targets)
            if not flag:
                print(f"Loss Data: {loss}")
                flag = True
            losses = sum([l for l in loss.values()])
            train_epoch_loss += losses.cpu().detach().numpy()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()        
        all_train_losses.append(train_epoch_loss)
        with torch.no_grad():
            for j , dt in enumerate(val_data_loader):
                imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
                targ = [dt[0][1] , dt[1][1]]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
                loss = model(imgs , targets)
                losses = sum([l for l in loss.values()])
                val_epoch_loss += losses.cpu().detach().numpy()
            all_val_losses.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss} Validation Loss: {val_epoch_loss}")
        
model.eval()
img = Image.open("C:/Users\emirb\OneDrive\Desktop\BLM3010/xray-x-ray-2764828_1280-900x588.jpg")
transform = torchvision.transforms.ToTensor()
ig = transform(img)
model.to('cuda')
ig = ig.to('cuda')
with torch.no_grad():
    pred = model([ig.to(device)])
