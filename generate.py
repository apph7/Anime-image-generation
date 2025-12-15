import os
import torch
from models.dcgan import Generator
from utils.visualize import save_samples

LATENT_SIZE = 256
NUM_IMAGES = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './saved_models/best_generator.pth'
OUT_DIR = 'generated_inference'

# 创建输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------
# 1. 加载模型
# ---------------------
generator = Generator(LATENT_SIZE).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
generator.load_state_dict(state_dict)
generator.eval()

# ---------------------
# 2. 生成随机 latent
# ---------------------
latent = torch.randn(NUM_IMAGES, LATENT_SIZE, 1, 1, device=DEVICE)

# ---------------------
# 3. 保存生成结果
# ---------------------
save_samples(generator, latent, epoch=0, sample_dir=OUT_DIR)

print("Done: images saved to", OUT_DIR)
