import numpy as np
import matplotlib.pyplot as plt

class MyGaussianMixture:
    def __init__(self, n_components, n_iterations=10, tol=1e-3):
        self.n_components = n_components  # 高斯混合模型的分量数量
        self.n_iterations = n_iterations  # 迭代次数
        self.tol = tol  # 收敛阈值

    def init_param(self, X, component_indices=None):
        """
        初始化\更新 模型均值、协方差和权重

        参数:
            X (ndarray): 输入数据形状为 (n_samples, n_features)
            component_indices (ndarray, optional): 组件索引形状为 (n_samples,) 默认为 None

        """
        n_samples, n_features = X.shape
        if component_indices is None:
            # 如果未提供组件索引则随机选择样本作为初始均值
            self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
            # 初始化协方差矩阵为单位矩阵
            self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
            # 初始化权重为均匀分布
            self.weights_ = np.ones(self.n_components) / self.n_components
        else:
            # 如果提供了组件索引则根据索引计算均值、协方差和权重
            self.means_ = np.array([X[component_indices == i].mean(axis=0) for i in range(self.n_components)])
            self.covariances_ = np.array([np.cov(X[component_indices == i].T) for i in range(self.n_components)])
            self.weights_ = np.array([np.sum(component_indices == i) for i in range(self.n_components)]) / n_samples

    def multi_gaussian(self, X, means, covariances):
        """
        计算每个样本在多元高斯概率密度函数中的概率密度

        参数:
        X: 输入数据数组形状为 (n_samples, n_features)
        means: 形状为 (n_components, n_features) 的数组 表示高斯分量的均值向量
        covariances: 形状为 (n_components, n_features, n_features) 的数组 表示高斯分量的协方差矩阵

        返回:
        probabilities: 形状为 (n_samples, n_components) 的数组 包含计算得到的概率密度

        """
        n_samples, n_features = X.shape
        # print(n_features) 3

        diff = X[:, np.newaxis, :] - means[np.newaxis, :, :]
        inv_covs = np.linalg.inv(covariances)

        diff = diff[:, :, np.newaxis, :]  # (n_samples, n_components, 1, n_features)
        inv_covs = inv_covs[np.newaxis, :, :, :]  # (1, n_components, n_features, n_features)

        # 
        exponent = -0.5 * np.einsum('ijkl,ijlm,ijkm->ijk', diff, inv_covs, diff)
        exponent = np.squeeze(exponent, axis=-1)  # (n_samples, n_components)

        norm_const = 1 / np.sqrt((2 * np.pi) ** n_features * np.linalg.det(covariances))
        norm_const = norm_const[np.newaxis, :]  # (1, n_components)

        return norm_const * np.exp(exponent)

    def expectation_step(self, X):
        """
        这个函数执行 E 步骤
        
        参数:
        X: 输入数据 形状为 (n_samples, n_features) 的二维数组
        self.responsibilities: 形状为 (n_samples, n_components) 的二维数组
        resp: 形状为 (n_samples, n_components) 的二维数组
        """
        resp = self.weights_ * self.multi_gaussian(X, self.means_, self.covariances_)
        resp_sum = resp.sum(axis=1, keepdims=True)
        # 将每个数据点属于每个高斯分布的概率归一化
        self.responsibilities = resp / resp_sum

    def maximization_step(self, X):
        """
        M步骤 更新模型参数

        参数:
            X: 输入数据形状为 (n_samples, n_features)

        """

        n_samples, n_features = X.shape
        # 计算每个组件的责任度之和
        Nk = self.responsibilities.sum(axis=0)

        # 更新每个组件的均值
        self.means_ = (self.responsibilities.T @ X) / Nk[:, np.newaxis]

        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        # 对每个组件
        for i in range(self.n_components):
            # 计算样本与当前均值的差值
            diff = X - self.means_[i]
            # 更新当前组件的协方差矩阵
            self.covariances_[i] = (self.responsibilities[:, i][:, np.newaxis] * diff).T @ diff / Nk[i]

        # 更新每个组件的权重
        self.weights_ = Nk / n_samples

    def fit(self, X, component_indices=None):
        # 更新模型参数
        self.init_param(X, component_indices)
        # 进行指定次数的迭代
        for _ in range(self.n_iterations):
            # 保存当前的均值
            prev_means_ = np.copy(self.means_)
            # 执行 E 步骤
            self.expectation_step(X)
            # 执行 M 步骤
            self.maximization_step(X)
            # 如果均值的变化小于设定的阈值 则停止迭代
            if np.linalg.norm(self.means_ - prev_means_) < self.tol:
                break

    def predict(self, X):
        """
        预测函数 计算每个样本属于每个高斯分布的概率
        并返回每个样本最可能属于的高斯分布的索引

        参数:
            X: 输入数据形状为 (n_samples, n_features)

        返回:
            predictions: 形状为 (n_samples,) 的一维数组 
            包含每个样本最可能属于的高斯分布的索引
        """
        # 计算每个样本属于每个高斯分布的概率
        resp = self.weights_ * self.multi_gaussian(X, self.means_, self.covariances_)
        return resp.argmax(axis=1)

    def score_samples(self, X):
        """
        计算每个样本的对数似然。

        参数:
        X: 形状为(n_samples, n_features)
            输入样本的特征向量。

        返回:
        log_prob:  形状为(n_samples,)
            每个样本的对数似然。

        """
        # 计算每个样本属于每个高斯分布的概率
        resp = self.weights_ * self.multi_gaussian(X, self.means_, self.covariances_)
        # 计算每个样本的对数似然
        log_prob = np.log(resp.sum(axis=1) + 1e-10)
        # 返回每个样本的对数似然
        return log_prob

    def plot_gaussian_mixture_3d(self, X):
        n_samples, n_features = X.shape
        num = 1000
        if n_samples > num:

            # 随机选择 100 个样本的索引
            random_indices = np.random.choice(n_samples, num, replace=False)

            # 根据随机选择的索引从 X 中取样
            X = X[random_indices]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, color='gray', label='Data points')
        
        for i in range(self.n_components):
            mean = self.means_[i]
            cov = self.covariances_[i]

            # 绘制均值点
            ax.scatter(mean[0], mean[1], mean[2], s=100, marker='x', color='red')

            # 计算椭球体参数
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))

            # 变换球面坐标到椭球体
            for j in range(len(x)):
                for k in range(len(x)):
                    [x[j, k], y[j, k], z[j, k]] = np.dot(eigenvectors, np.array([x[j, k], y[j, k], z[j, k]]) * np.sqrt(eigenvalues))

            x += mean[0]
            y += mean[1]
            z += mean[2]

            ax.plot_surface(x, y, z, rstride=4, cstride=4, color='blue', alpha=0.2)

        ax.set_title('Gaussian Mixture Model in 3D')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.legend()
        plt.show()