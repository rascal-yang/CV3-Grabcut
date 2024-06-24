import numpy as np
import igraph as ig
from GMM import MyGaussianMixture
from Graph import Graph

BG, FG, PR_BG, PR_FG = 0, 1, 2, 3

class GrabCut:
    def __init__(self, img, mask, rect):
        # 转换为NumPy数组
        self.img = np.asarray(img, dtype=np.float64)
        self.height, self.width, _ = img.shape
        self.mask = mask
        self.mask[rect[1] : rect[1]+rect[3], rect[0] : rect[0]+rect[2]] = PR_FG

        # 执行GrabCut算法的主要循环 包含初始化、迭代更新GMMs、构建图割图和估计分割 直到达到指定迭代次数 
        self.main()

    def get_beta(self):
        '''
        计算平滑项beta 它反映了图像中相邻像素间的颜色差异 用于平滑度惩罚项
        '''
        # gamma参数用于平滑度惩罚项
        self.gamma = 50  # 论文中建议的gamma参数

        R_L = self.img[:, 1:] - self.img[:, :-1]
        DR_UL = self.img[1:, 1:] - self.img[:-1, :-1]
        D_U = self.img[1:, :] - self.img[:-1, :]
        DL_UR = self.img[1:, :-1] - self.img[:-1, 1:]

        # 平滑度惩罚项的权重
        self.beta = np.sum(np.square(R_L)) + np.sum(np.square(DR_UL)) + \
            np.sum(np.square(D_U)) + np.sum(np.square(DL_UR))
        self.beta = 1 / (2 * self.beta / (
            # 除以R_L DR_UL D_U DL_UR 所有的像素个数进行归一化
            4 * self.width * self.height- 3 * self.width- 3 * self.height + 2))

        # 计算平滑度惩罚项的权重
        # 在构建图的过程中 这些权重作为边的容量 用于表示相邻像素之间的分割成本
        self.left_V = self.gamma * np.exp(-self.beta * np.sum(np.square(R_L), axis=2))
        self.upleft_V = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(
            np.square(DR_UL), axis=2))
        self.up_V = self.gamma * np.exp(-self.beta * np.sum(np.square(D_U), axis=2))
        self.upright_V = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(
            np.square(DL_UR), axis=2))

    def update_pixels(self):
        '''根据掩码更新 b_indexes f_indexes'''
        # 确定已知背景和前景像素的索引
        self.b_indexes = np.where(np.logical_or(self.mask == BG, self.mask == PR_BG))
        self.f_indexes = np.where(np.logical_or(self.mask == FG, self.mask == PR_FG))

    def init_GMMs(self):
        '''初始化GMM模型'''
        # 最佳GMM组件数K建议在论文中
        gmm_num = 5

        # 创建GaussianMixture对象
        self.back_gmms = MyGaussianMixture(gmm_num)
        self.fore_gmms = MyGaussianMixture(gmm_num)

        # 使用前景和背景的像素索引对GaussianMixture对象进行初始化
        self.back_gmms.init_param(self.img[self.b_indexes])
        self.fore_gmms.init_param(self.img[self.f_indexes])

    def Assign_GMM_components_to_pixels(self):
        """
        论文中 Figure 3 的第一步
        分配像素到GMM的组件
        """
        # 为每个像素分配一个GMM组件索引
        self.pixel2gmm = np.empty((self.height, self.width), dtype=np.uint32)
        # 使用背景GMM模型对背景像素进行预测 并将预测结果赋值给对应的组件索引
        self.pixel2gmm[self.b_indexes] = self.back_gmms.predict(self.img[self.b_indexes])
        # 使用前景GMM模型对前景像素进行预测 并将预测结果赋值给对应的组件索引
        self.pixel2gmm[self.f_indexes] = self.fore_gmms.predict(self.img[self.f_indexes])

    def Learn_GMM_param(self):
        """
        论文中 Figure 3 的第二步
        学习GMM参数
        """
        
        # 使用背景像素和对应的组件索引拟合背景GMM模型
        self.back_gmms.fit(self.img[self.b_indexes],self.pixel2gmm[self.b_indexes])
        # 使用前景像素和对应的组件索引拟合前景GMM模型
        self.fore_gmms.fit(self.img[self.f_indexes],self.pixel2gmm[self.f_indexes])

    def GMM_3D(self):
        self.fore_gmms.plot_gaussian_mixture_3d(self.img[self.f_indexes].reshape(-1, 3))

    def add_all_t_links(self, b_indexes, f_indexes, pr_indexes):
        # t-links
        # 添加 源节点索引 和 可能前景像素索引 作为边
        pr_pixels = self.img.reshape(-1, 3)[pr_indexes]
        # 计算了每个前景像素属于背景的概率


        def add_t_links(source_indexes, target_index, capacities):
            self.edges.extend(list(zip([target_index] * len(source_indexes), source_indexes)))
            self.capacity.extend(capacities)

        b_scores = -self.back_gmms.score_samples(pr_pixels)
        f_scores = -self.fore_gmms.score_samples(pr_pixels)
        add_t_links(pr_indexes, self.S, b_scores.tolist())
        add_t_links(pr_indexes, self.T, f_scores.tolist())

        # 添加 源节点索引 和 背景像素索引 作为边 权重为0
        # 添加 汇节点索引 和 背景像素索引 作为边 权重为9*gamma

        add_t_links(b_indexes, self.S, [0] * len(b_indexes))
        add_t_links(b_indexes, self.T, [9 * self.gamma] * len(b_indexes))
        add_t_links(f_indexes, self.S, [9 * self.gamma] * len(f_indexes))
        add_t_links(f_indexes, self.T, [0] * len(f_indexes))

    def add_all_n_links(self):
        img_indexes = np.arange(self.height * self.width,
                    dtype=np.uint32).reshape(self.height, self.width)
        # 添加相邻像素之间的边
        def add_n_links(mask1, mask2, capacity):
            self.edges.extend(list(zip(mask1, mask2)))
            self.capacity.extend(capacity.reshape(-1).tolist())
        # 添加相邻像素之间的边 左右的连接
        add_n_links(img_indexes[:, 1:].reshape(-1), img_indexes[:, :-1].reshape(-1), self.left_V)
        # 添加相邻像素之间的边 左上右下的连接
        add_n_links(img_indexes[1:, 1:].reshape(-1), img_indexes[:-1, :-1].reshape(-1), self.upleft_V)
        # 添加相邻像素之间的边 上下的连接
        add_n_links(img_indexes[1:, :].reshape(-1), img_indexes[:-1, :].reshape(-1), self.up_V)
        # 添加相邻像素之间的边 右上左下的连接
        add_n_links(img_indexes[1:, :-1].reshape(-1), img_indexes[:-1, 1:].reshape(-1), self.upright_V)

    def init_gragh(self):
        '''
        构建一个图割图模型
        其中包含了源节点 汇节点以及代表图像像素的节点之间的连接
        每条边的容量代表了将两个相邻像素分开的成本 
        '''

        self.S = self.width * self.height  # 源节点 S 的索引
        self.T = self.S + 1       # 源节点 T 的索引
        flat_mask = self.mask.reshape(-1)
        # np.where返回两个向量 一个是返回点的x值 一个是y值
        b_indexes = np.where(flat_mask == BG)[0]
        f_indexes = np.where(flat_mask == FG)[0]
        # 不确定的前景和背景像素
        pr_indexes = np.where(np.logical_or
                (flat_mask == PR_BG, flat_mask == PR_FG))[0]

        self.edges = []
        self.capacity = []      # 边的容量

        # t-links
        self.add_all_t_links(b_indexes, f_indexes, pr_indexes)
        # n-links
        self.add_all_n_links()

        assert len(self.edges) == 4 * self.width * self.height - 3 * (self.width + self.height) + 2 + \
            2 * self.width * self.height
        
        # print(len(self.edges), len(self.capacity))
        # 初始化一个指定节点的图对象
        self.graph = ig.Graph(self.width * self.height + 2)
        # self.graph = Graph(self.width * self.height + 2)
        # 添加边
        self.graph.add_edges(self.edges)

    def Estimate_segmentation(self):
        """
        论文中 Figure 3 的第三步
        使用图割算法估计前景和背景 更新掩码
        """
        # 构建一个图割模型来表示当前图像分割问题
        self.init_gragh()
        # 使用st_mincut方法找到从源点到汇点的最小割
        mincut = self.graph.st_mincut(
            self.S, self.T, self.capacity)
        # mincut = self.graph.find_min_cut(self.S, self.T)

        # 打印前景像素和背景像素的数量
        # print('fore pixels: %d, back pixels: %d' % (
        #     len(mincut.partition[0]), len(mincut.partition[1])))

        # 找到所有标记为可能的前景或可能的背景的像素的索引 对他们进行更新
        pr_indexes = np.where(np.logical_or(
            self.mask == PR_BG, self.mask == PR_FG))
        # 创建一个与图像大小相同的索引数组
        img_indexes = np.arange(self.height * self.width,
                                dtype=np.uint32).reshape(self.height, self.width)

        # 根据最小割的结果更新可能的前景和可能的背景像素的掩码
        '''
        np.isin(img_indexes[pr_indexes], graph.partition[0])
        检查 pr_indexes 中的像素是否在前景部分中 如果是 则返回 True 否则返回 False
        使用 np.where 更新 self.mask 中这些像素的值
        如果像素在前景部分中 则设置为 PR_FG 否则设置为 PR_BG 
        '''
        self.mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], mincut.partition[0]),
                                        PR_FG, PR_BG)
        # print(np.where(self.mask==1))
        # print(np.where(self.mask==2))
        # print(np.where(self.mask==3))

    def main(self, epochs=1):
        # 将掩码中的所有其他像素设置为可能的背景
        self.update_pixels()
        # 计算平滑项beta 它反映了图像中相邻像素间的颜色差异 用于平滑度惩罚项 
        self.get_beta()
        # 初始化GMM模型
        self.init_GMMs()
        for _ in range(epochs):
            # 为每个像素分配最可能的GMM模型组件
            self.Assign_GMM_components_to_pixels()
            # 根据当前分配更新背景和前景的Gaussian Mixture Models (GMMs)
            self.Learn_GMM_param()
            # 利用图割算法来估计并更新图像的分割掩模
            self.Estimate_segmentation()
            print("完成最小割算法")
            # 更新背景和前景
            self.update_pixels()
            # self.GMM_3D()


