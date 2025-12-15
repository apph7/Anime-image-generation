import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from utils.dataset import get_train_loader
from utils.visualize import save_samples, plot_loss_score
from models.dcgan import Generator, Discriminator

# --------------------- 配置参数 ---------------------
DATA_DIR = './data/animefacedataset'
IMAGE_SIZE = 64
BATCH_SIZE = 128
LATENT_SIZE = 256
EPOCHS = 50
LR = 0.0002
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------- 初始化模型 ---------------------
generator = Generator(latent_size=LATENT_SIZE).to(DEVICE)
discriminator = Discriminator().to(DEVICE)

opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# --------------------- 训练函数 ---------------------
def train_discriminator(real_images):
    opt_d.zero_grad()
    real_preds = discriminator(real_images)
    real_loss = F.binary_cross_entropy(real_preds, torch.ones(real_images.size(0), 1, device=DEVICE))

    latent = torch.randn(real_images.size(0), LATENT_SIZE, 1, 1, device=DEVICE)
    fake_images = generator(latent)
    fake_preds = discriminator(fake_images.detach())
    fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros(fake_images.size(0), 1, device=DEVICE))

    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_preds.mean().item(), fake_preds.mean().item()

def train_generator(batch_size):
    opt_g.zero_grad()
    latent = torch.randn(batch_size, LATENT_SIZE, 1, 1, device=DEVICE)
    fake_images = generator(latent)
    preds = discriminator(fake_images)
    loss = F.binary_cross_entropy(preds, torch.ones(batch_size, 1, device=DEVICE))
    loss.backward()
    opt_g.step()
    return loss.item(), fake_images

# --------------------- 主训练循环 ---------------------
def main():
    os.makedirs('./generated', exist_ok=True)
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    train_loader = get_train_loader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    best_g_loss = float('inf')

    # 用于记录历史
    history = {
        'epoch': [],
        'g_loss': [],
        'd_loss': [],
        'real_score': [],
        'fake_score': []
    }

    fixed_latent = torch.randn(64, LATENT_SIZE, 1, 1, device=DEVICE)

    for epoch in range(1, EPOCHS + 1):
        g_loss_epoch = 0
        d_loss_epoch = 0
        real_score_epoch = 0
        fake_score_epoch = 0

        for batch_idx, real_images in enumerate(train_loader):
            real_images = real_images.to(DEVICE)
            d_loss, real_score, fake_score = train_discriminator(real_images)
            g_loss, fake_images = train_generator(real_images.size(0))

            g_loss_epoch += g_loss
            d_loss_epoch += d_loss
            real_score_epoch += real_score
            fake_score_epoch += fake_score

            # 只在每个 epoch 最后一个 batch 输出
            if batch_idx == len(train_loader) - 1:
                print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}], "
                      f"D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}, "
                      f"Real Score: {real_score:.4f}, Fake Score: {fake_score:.4f}")


                # 使用固定 latent 生成固定 8x8 样本图
                with torch.no_grad():
                    fixed_fake_images = generator(fixed_latent)
                save_image((fixed_fake_images + 1) / 2, f'./generated/epoch_{epoch:03d}.png', nrow=8)
        # 计算平均值
        num_batches = len(train_loader)
        g_loss_epoch /= num_batches
        d_loss_epoch /= num_batches
        real_score_epoch /= num_batches
        fake_score_epoch /= num_batches

        # 保存最佳模型
        if g_loss_epoch < best_g_loss:
            best_g_loss = g_loss_epoch
            torch.save(generator.state_dict(), './saved_models/best_generator.pth')
            torch.save(discriminator.state_dict(), './saved_models/best_discriminator.pth')
            print(f"New best model saved at epoch {epoch} with Avg G Loss: {g_loss_epoch:.4f}")

        # 记录历史
        history['epoch'].append(epoch)
        history['g_loss'].append(g_loss_epoch)
        history['d_loss'].append(d_loss_epoch)
        history['real_score'].append(real_score_epoch)
        history['fake_score'].append(fake_score_epoch)

    # 绘制训练曲线
    plot_loss_score(history, log_dir='./logs')
    print("Training completed. Best models saved and curves plotted.")

# --------------------- Windows 多进程安全 ---------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    print("Device:", DEVICE)
    main()