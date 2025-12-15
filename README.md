# AnimeDCGAN Project
## 作业要求
- 使用 GAN 网络生成动漫人脸图像
- 数据集：Anime-Face Dataset
- 模型：DCGAN
##项目结构
AnimeDCGAN/
│
├── data/
│   └── animefacedataset/
│       ├── download.py            # 下载 Kaggle Anime Face Dataset
│       └── kaggle.json            # Kaggle API 密钥文件
│
├── generated/                     # 训练过程中生成的样本图像
│   └── epoch_XXX.png
│
├── generated_inference/           # 使用训练好的模型生成图片
│   └── generated-images-0000.png
│
├── logs/                          # 训练日志和曲线
│   ├── loss_curve.png
│   └── score_curve.png
│
├── models/
│   └── dcgan.py                   # 定义 DCGAN Generator 和 Discriminator
│
├── saved_models/                   # 保存最佳模型权重
│   ├── best_generator.pth
│   └── best_discriminator.pth
│
├── utils/
│   ├── dataset.py                  # Dataset 与 DataLoader 构建
│   └── visualize.py                # 图像可视化与保存工具
│
├── train.py                        # GAN 训练主脚本
├── generate.py                     # 使用训练好的模型生成图片
├── main.py                         # 可选主入口脚本
├── requirements.txt                # 依赖环境
└── README.md                        # 项目说明、作业报告说明

## 使用方法
1. 下载数据集: `python data/download.py`
2. 训练模型: `python train.py`
3. 生成图像: `python generate.py`
## 实验结果
- 每个 epoch 会在 `generated/` 文件夹保存生成图像
