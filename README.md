# Grabcut

## 项目简介
本项目是一个基于Python的图像分割与分析工具集，包括高斯混合模型（GMM）和图形切割（Grabcut）算法的实现。项目旨在通过先进的图像处理技术，提供对图像中前景和背景的自动和交互式分割。

## 主要组件
### GMM.py
- 实现了高斯混合模型（Gaussian Mixture Model）的自定义类，用于概率密度函数的估计和样本的分类。

### Graph.py
- 定义了图的数据结构和算法，包括节点、边、图的创建，以及广度优先搜索（BFS）和最小割集合的查找。

### main.py
- 使用Gradio库创建了一个交互式界面，允许用户通过矩形选框和打点的方式进行图像分割。

### testGMM.py
- 演示了如何使用Grabcut算法进行前景提取，包括用户交互式选择感兴趣区域（ROI）和执行Grabcut算法。

### Grabcut.py
- 实现了Grabcut算法的类，包括初始化、迭代更新GMMs、构建图割图和估计分割等功能。

## 环境要求
- Python 3.9
- NumPy
- Matplotlib（用于图形展示）
- OpenCV
- igraph（用于图的创建和操作）

## 安装指南
1. 确保Python环境已安装。
2. 使用pip安装所需的库：
   ```bash
   pip install numpy matplotlib opencv-python igraph gradio
   ```

## 使用方法
1. 运行`main.py`启动Gradio交互式界面：
   ```bash
   python main.py
   ```
2. 通过界面选择图像，使用矩形选框和打点模式进行图像分割。
3. 运行`testGMM.py`进行Grabcut算法的测试和演示。

## 项目结构
```
image_segmentation_project/
│
├── 00.jpg            # 测试图片
├── GMM.py            # 高斯混合模型实现
├── Graph.py          # 图的数据结构和算法实现
├── main.py           # Gradio交互式界面
├── testGMM.py        # Grabcut算法测试和演示
└── Grabcut.py        # Grabcut算法实现
```

## 注意事项
- 确保测试图像路径正确，或修改代码中的图像路径为正确的文件路径。
- 根据实际图像内容调整算法参数，以获得最佳分割效果。