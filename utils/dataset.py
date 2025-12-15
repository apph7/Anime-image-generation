import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class AnimeFaceDataset(Dataset):
    """Anime Face Dataset, 直接读取 images/ 下所有图片"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [os.path.join(root_dir, f)
                      for f in os.listdir(root_dir)
                      if f.endswith('.png') or f.endswith('.jpg')]
        if len(self.files) == 0:
            raise RuntimeError(f"No image files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_train_loader(data_dir, image_size, batch_size, num_workers=2, pin_memory=True):
    """返回训练 DataLoader"""
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = AnimeFaceDataset(os.path.join(data_dir, 'images'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader
