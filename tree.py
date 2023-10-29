# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
def loadSimpData():
    """
    创建单层决策树的数据集
    Parameters:
        无
    Returns:
        dataMat - 数据矩阵
        classLabels - 数据标签
    """
    dataMat = np.matrix([[0., 1., 3.],
                      [0., 3., 1.],
                      [1., 2., 2.],
                      [1., 1., 3.],
                      [1., 2., 3.],
                      [0., 1., 2.],
                      [1., 1., 2.],
                      [1., 1., 1.],
                      [1., 3., 1.],
                      [0., 2., 1.]])
    classLabels = np.matrix([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    return dataMat, classLabels

def showDataSet(dataMat, labelMat):
    """
    数据可视化
    Parameters:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Returns:
        无
    """
    ax = plt.axes(projection='3d')
    data_plus = []  #正样本
    data_minus = [] #负样本
    labelMat = labelMat.T   #label矩阵转置
    #将数据集分别存放到正负样本的矩阵
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)      #转换为numpy矩阵
    data_minus_np = np.array(data_minus)    #转换为numpy矩阵
    ax.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], np.transpose(data_plus_np)[2], c='r')        #正样本散点图
    ax.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], np.transpose(data_minus_np)[2], c='b')     #负样本散点图
    plt.show()

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    单层决策树分类函数
    Parameters:
        dataMatrix - 数据矩阵
        dimen - 第dimen列，也就是第几个特征
        threshVal - 阈值
        threshIneq - 划分的符号 - lt:less than，gt:greater than
    Returns:
        retArray - 分类结果
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))  # 初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0  # 如果大于阈值,则赋值为-1
    return retArray

def buildStump(dataArr, classLabels, D):
    """
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    Tips:
    在已经写好单层决策树桩的前提下，这个函数需要用来确定哪个特征作为划分维度、划分的阈值以及划分的符号，从而输出“最佳的单层决策树”。
    具体来说，我们需要做一个嵌套三层的遍历：第一层遍历所有特征，第二层遍历这一维度特征所有可能的阈值，第三层遍历划分的符号；
    在确定以上三个关键信息之后，我们只需要调用决策树桩函数并获得其预测结果，结合真值计算误差；
    将误差最小的决策树桩的信息用一个字典储存下来，作为最终的输出结果；
    """
    num=np.shape(dataArr)[0]
    bestStump=np.ones(3)
    bestClasEst=np.ones((num,1))
    minError=1
 
    for i in range(np.shape(dataArr)[1]):
         for j in range(int(min(dataArr[:,i]-1)),int(max(dataArr[:,i]+1))):
            ClasEst=stumpClassify(dataArr, i, j, 'lt')
            error=0.
            for k in range(num):
                if(classLabels[0,k]!=ClasEst[k]):
                    error=error+D[k]
            if(error<minError):
                minError=error
                bestClasEst=ClasEst
                bestStump=[i,j,1.]

            ClasEst=stumpClassify(dataArr, i, j, 'gt')
            error=0.
            for k in range(num):
                if(classLabels[0,k]!=ClasEst[k]):
                    error=error+D[k]
            if(error<minError):
                minError=error
                bestClasEst=ClasEst
                bestStump=[i,j,-1.]
               
            
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    完整决策树训练
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        numIt - 默认迭代次数
    Returns:
        weakClassArr- 完整决策树信息
        aggClassEst- 最终训练数据权值分布
    Tips:
    基于我们已经写好的最优决策树函数，我们可以在现有数据上进行迭代，不断更新数据权重、算法权重与其对应的决策树桩，
    直到误差为零，退出循环
    """
    dataArr=np.array(dataArr)
    num=np.shape(dataArr)[0]
    bestStump=np.ones((numIt,3))
    bestClasEst=np.ones((numIt,num,1))
    minError=np.ones((numIt,1))
    weight=np.zeros((numIt,1))
    totLabels=np.zeros(num)
    D=np.ones((num,1))
    D=D/num
    for i in range (numIt):
        bestStump[i],minError[i],bestClasEst[i]=buildStump(dataArr, classLabels, D)
        errrat=minError[i]
        weight[i]=0.5 *np.log((1-errrat)/errrat)

        weakClassArr=bestStump[0:i+1]
        aggClassEst=weight[0:i+1]
 
        totLabels=totLabels+weight[i]*bestClasEst[i,:,0]
        tmp=1
        for j in range(num):
           if (np.sign(totLabels)[j]!=classLabels[0,j]):
                tmp=0
        if(tmp==1):
            break
        for j in range(num):
           D[j]=D[j]*np.exp(-bestClasEst[i,j,0]*classLabels[0,j]*weight[i])    
        sumD=sum(D)
        D=D/sumD
           
    return weakClassArr,aggClassEst

if __name__ == '__main__':
    dataArr, classLabels = loadSimpData()
    showDataSet(dataArr, classLabels)
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    print(weakClassArr)
    print(aggClassEst)
