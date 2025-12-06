import os
import glob
import torch
import torchvision.transforms as transforms  
from PIL import Image

def load_sorted_data(data_root):
    """
    Loads data from folders and buckets them by resolution.
    """
    
    buckets = {
        (32, 32): {'data': [], 'labels': []},
        (48, 48): {'data': [], 'labels': []},
        (64, 64): {'data': [], 'labels': []}
    }
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_paths = glob.glob(os.path.join(data_root, '*', '*.png'))
    
    for path in image_paths:
        parent_folder = os.path.basename(os.path.dirname(path))
        label = int(parent_folder)
        
        # Load image
        with Image.open(path) as img:
            img = img.convert('L') # grayscale
            res = img.size # (width, height)

            if res in buckets:
                t_img = transform(img) # transform to tensor
                buckets[res]['data'].append(t_img)
                buckets[res]['labels'].append(label)

    tensor_buckets = []
    
    for res, content in buckets.items():
        x_tensor = torch.stack(content['data'])
        y_tensor = torch.tensor(content['labels'], dtype=torch.long)
        tensor_buckets.append((x_tensor, y_tensor))

    return tensor_buckets
