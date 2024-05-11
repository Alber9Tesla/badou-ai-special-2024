import numpy as np


class CPCA(object):
    '''
    用pca求样本矩阵X的K阶降维矩阵Z
    note
    '''

    def __init__(self, X, K):
        '''

        :param X:训练样本矩阵X
        :param K:X的降维矩阵阶数，即X要降维成K阶
        '''

        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU

    def _centralized(self):
        """矩阵X的中心化"""
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的均值特征，np.array函数的作用为可以把列表中数据转换为矩阵或者向量，用于创建一个组
        print('样本集的均值特征mean:\n', mean)          #self.X的行代表样本个数，列代表the number of feature，平均值求的是列的平均。self.X.T取行变相的就是取self.X的列
        centrX = self.X - mean  # 样本集中心化
        print('样本矩阵中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]  # shape(matrixA) [0]  ==》 行数  ，shape(matrixA) [1]  ==》 列数
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)  # np.dot 为向量内积用用中心化后的转置矩阵乘以原来的矩阵
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        """ 求X的降维矩阵转换矩阵U,shape=(n,k),n是特征维度总数,k是降维矩阵特征维度"""
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)  # 特征值赋予给a，对应特征向量赋值给b（返回的一维的arry给a，返回维度与方阵相同的给b）
        print('样本协方差矩阵C的特征值:\n', a)
        print('样本协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)     #argsort 把a按从大到小进行排序生成对应下标，相当于对原a进行了降序排序
        print(ind)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]  #[表达式 for 循环计数器 in 可迭代对象] 这是列表生成式
        print('降维转换矩阵:\n',UT)
        U = np.transpose(UT)   #transpose()函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置：
        print('%d降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        """按照Z=XU求降维矩阵Z，shape=(m,k), n是样本总数,k是降维矩阵中的特征维度总数"""
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print("打印降维矩阵Z:\n", Z)
        return Z


if __name__ == '__main__':
    '10个样本3特征的样本集，行为样例，列为特征维度'
    X = np.array([[12, 23, 29],
                  [11, 12, 10],
                  [19, 21, 24],
                  [12, 36, 39],
                  [20, 25, 28],
                  [92, 82, 88],
                  [38, 95, 97],
                  [31, 35, 37],
                  [30, 39, 32],
                  [61, 63, 64]])

    K = np.shape(X)[1] - 1  # 原先的维度减一
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)
