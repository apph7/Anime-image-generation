import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os


# ---------------------------
# 图像反归一化
# ---------------------------
def denorm(img_tensors, stats=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    return img_tensors * stats[1][0] + stats[0][0]


# ---------------------------
# 显示图像网格
# ---------------------------
def show_images(images, nmax=64, stats=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax], stats), nrow=8).permute(1, 2, 0))
    plt.show()


# ---------------------------
# 保存生成样本
# ---------------------------
def save_samples(generator, latent_tensors, epoch, sample_dir='generated', stats=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    os.makedirs(sample_dir, exist_ok=True)
    with torch.no_grad():
        fake_images = generator(latent_tensors)
    fname = f'{sample_dir}/generated-images-{epoch:04d}.png'
    save_image(denorm(fake_images, stats), fname, nrow=8)
    print(f'Saving {fname}')
    return fname


# ---------------------------
# 绘制训练曲线
# ---------------------------
def plot_loss_score(history, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)

    # Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['g_loss'], label='Generator Loss')
    plt.plot(history['epoch'], history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(f'{log_dir}/loss_curve.png')
    plt.show()

    # 判别器分数曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['real_score'], label='Real Score')
    plt.plot(history['epoch'], history['fake_score'], label='Fake Score')
    plt.xlabel('Epoch')
    plt.ylabel('Average Score')
    plt.title('Discriminator Real/Fake Score')
    plt.legend()
    plt.grid()
    plt.savefig(f'{log_dir}/score_curve.png')
    plt.show()



