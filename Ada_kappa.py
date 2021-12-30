#!/user/bin/python3
# Author: HuangCong
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import time

#建立简单数据集
# def loadSimpData():
#     datMat = np.mat([[1.6,2.1],
#                      [2.4, 1.1],
#                      [1.3, 1.5], 
#                      [1.8, 1.4],
#                      [2.0, 1.3],
#                      [1.0, 2.1],
#                      [1.5, 1.5],
#                      [1.45, 1.86]])
#     classLabel = [1.0, 1.0, -1.0, -1.0, 1.0,1.0,-1.0,1.0]
#     return datMat, classLabel


#通过阈值比较对数据进行分类——(特征矩阵、维度、阈值、阈值不等号)
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((dataMatrix.shape[0], 1))  #先默认全是1，建立列向量[m, 1]——与标签列相对应
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= 9-threshVal] = -1  #这里是看是小于阈值为负还是大于阈值为负
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1
    return retArray


#该函数会遍历stumpClassify()函数所有可能的输入值，并找到该数据集上的最佳单层树——根据数据权重向量D来定义
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr); 
    labelMat = np.mat(classLabels).T #classLabel向量为[1,n],需要转置
    m, n = dataMatrix.shape  #m个样本，n个特征
    numSteps = 10.0  #用于在特征的所有可能值上进行遍历
    bestStump = {} #该词典保存最佳单层决策树的相应参数
    bestClasEst = np.mat(np.zeros((m, 1)))    #保存最佳估计标签值，先初始化[m,1]零向量
    minError = np.inf  #初始化为无穷大，用于寻找可能的最小错误率
    for i in range(n):  #在所有的特征上进行遍历
        rangeMin = dataMatrix[:, i].min()  #取该列特征值中的最小值
        rangeMax = dataMatrix[:, i].max()  #同理以上
        stepSize = (rangeMax - rangeMin)/numSteps  #确定步长
        for j in range(-1, int(numSteps) + 1):  #将阈值设置为整个取值范围之外也是也可以的
            for inequal in ['lt', 'gt']: #在大于和小于之间切换不等式
                threshVal = (rangeMin + float(j) * stepSize)   #确定阈值
                predictdVals = stumpClassify(dataMatrix, i, threshVal, inequal)  #进行预测
                errArr = np.mat(np.ones((m, 1)))   #错误矩阵，如果predictedVals值不等于labelMat中真正类别值，置为一
                errArr[predictdVals == labelMat] = 0
                weightedError = D.T * errArr  #权重向量与错误向量相乘得到错误率
                #适当打印，帮助理解函数的运行
              #  print("现在是第%d列的特征值, 选择的阈值是thresh %.2f, thresh 大于还是小于ineqal: %s, the 此时错误率是weighted error is %.3f"% (i, threshVal, inequal, weightedError))
                if weightedError < minError:  #当前错误率小于已有的最小错误率
                    minError = weightedError  #进行更新
                    bestClasEst = predictdVals.copy()  #保存预测值
                    bestStump['dim'] = i  #保存维度
                    bestStump['thresh'] = threshVal  #保存阈值
                    bestStump['ineq'] = inequal  #保存不等号
        # print("当前第%d列选择的阈值是：%.2f,大于阈值取%s" % (i,bestStump['thresh'],bestStump['ineq']))
    return bestStump, minError, bestClasEst


#基于单层决策树的AdaBoost的训练过程 (数据集、类别标签、迭代次数)，尾部DS代表(decision stump单层决策树)
def adaBoostTrainDS(dataArr, classLabels, numIt=40):  #迭代次数是算法中唯一需要用户指定的参数（不给的话就是默认40 次）
    weakClassArr = []   #聚焦该分类器的所有信息，最后返回
    m = dataArr.shape[0]  #样本数为m
    D = np.mat(np.ones((m, 1)) / m)  #样本权重初始化，都相等，后续迭代中会增加错分数据的权重同时，降低正确分类数据的权重
    aggClassEst = np.mat(np.zeros((m, 1)))  #（做一个m行1列全为0的矩阵）记录每个数据点的类别估计累计值
    error_times = np.mat(np.ones((m,1)) )   #创建错误次数表
    for i in range(numIt):  #numIt次迭代
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #上一行返回利用D得到的具有最小错误率的单层决策树，同时返回最小错误率和更新之后估计的类别向量
        # print("输出样本的权重D:", D.T)   #Python中是不是只要输出一个变量时就可以省略%？？？？？？？
        #下一行alpha的计算公式可详见李航蓝本，max()函数以防发生除零错误
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha  #继续把当前选出的弱分类器的权重存入该字典——包括了分类所需要的所有信息
        #weakClassArr.append(bestStump)   #保存当前弱分类器的所有信息到列表中   为了再次优化，我把它往前调。
        print("当前弱分类器的估计结果classEst: ", classEst.T) #打印类别估计值
        #以下三行用于计算下一次迭代中的新的数据权重向量D，公式可见李航蓝本
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        # 这里进行优化：引入惩罚因子  但是这种改进反而还使正确率下降了
        fault=(np.sign(aggClassEst) != np.mat(classLabels).T).tolist()    #变成列表形式，因为布尔索引装不了矩阵
        right=(np.sign(aggClassEst) == np.mat(classLabels).T).tolist()
        l=[b for a in fault for b in a]   #但是毕竟是由矩阵变来的，还要去一次括号
        error_times[l]+=1   #错误次数加一
        l=[b for a in right for b in a]
        error_times[l]=1 #只要正确了，错误次数就会中断，变成1
        print("错误次数：" , error_times)
        #csl=1-np.exp(1/error_times)
        #csl=error_times
        #csl=np.log(error_times)
        csl=np.log(error_times)/np.log(15)   
        csl[csl<1]=1   #防止分母为0的情况出现,和为了计算方便  
        D=np.multiply(D,1/csl)  #重新调整权重
        # # D=D*(1/csl)     #不能直接相乘，不符合广播原则，而且结果也不是我想要的
        D = D/D.sum()       #归一化
        #以下四行用于错误率累加的计算，通过aggClassEst变量保持一个运行时的类别估计值来实现
        aggClassEst += alpha * classEst
        print("类别估计累计值aggClassEst: ", aggClassEst)  #由于aggClassEst是浮点数，需要调用sign()函数
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1))) #矩阵相乘，前者为0-1，后者全1
        errorRate = aggErrors.sum() / m
        print("错误率为errorRate: ", errorRate)     #正常的单变量可以直接不加%的
        weakClassArr.append(bestStump)
        if errorRate == 0.0:  #如果错误率为0，停止for循环
            break

        # if errorRate >= x:
        #     continue
        # weakClassArr.append(bestStump)
    return weakClassArr  #返回信息列表


#基于adaboost进行分类——(待分类样例，多个弱分类器组成的数组)
def adaClassify(dataToClass, classifierArr):#前一个参数是矩阵类型的数据，后面是加权后的强分类器
    dataMatrix = np.mat(dataToClass)  #首先转成numpy矩阵
    m = dataMatrix.shape[0] #待分类样例的个数为m
    aggClassEst = np.mat(np.zeros((m, 1)))  #构建0列向量，与adaBoostTrainDS中含义一样
    for i in range(len(classifierArr)):  #遍历所有的弱分类器
        #基于stumpClassify()对每个分类器得到一个类别的估计值
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)  #打印出每加一次分类器后的预测结果
        print("预测数据有%d个" % len(aggClassEst))   
    num = []
    for i in range(len(classifierArr)):
        num.append(classifierArr[i]['alpha'])
    print("分类器权重方差：", np.var(num))
    print("分类器权重标准差：", np.std(num))
    print("分类器权重均值：",np.mean(num))    
    return np.sign(aggClassEst) #返回最后一次分类器预测的结果


# #定义自适应加载函数(很有用)
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split(','))     #读取有多少个列索引
    dataMat = []
    labelMat = []
    fr = open(fileName)     
    #fr.readline()  空读一行，跳过列索引
    for line in fr.readlines()[1:]:
        lineArr = []
        curLine = line.strip().split(',')
        for i in range(numFeat-1):
            print(curLine[i])
            lineArr.append(float(curLine[i]))     #加一可以跳过行索引
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1])) #这里的-1是指最后一个的意思
    dataMat=np.mat(dataMat)     #转矩阵
    labelMat=np.mat(labelMat)
    return dataMat, labelMat

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# def normalization(data):        #归一化的值在-1到1之间
#     _range = np.max(abs(data))
#     return data / _range


if __name__ == "__main__":
    time_start = time.time() #开始计时
    #dataMat, classLabels = loadDataSet('train_train.csv')   #导数据final_data_train.csv
    dataMat, classLabels = loadDataSet('final_data_train.csv')
    dataMat=normalization(dataMat)      #归一化
    classifierArr = adaBoostTrainDS(dataMat, classLabels, 20)

    #测试公式的可靠性（已经确定还OK了，就直接注掉它了）
    #dataMat, classLabels = loadDataSet('train_test.csv')   #导数据
    dataMat, classLabels = loadDataSet('final_data_text.csv')
    dataMat=normalization(dataMat)
    predict_mat=adaClassify(dataMat, classifierArr)  #接收回预测的值
    m = dataMat.shape[0]  #样本数为m
    TP=FN=FP=TN=0
    for i in range(m):
        # print((np.mat(classLabels).T)[i])
        # print(predict_mat[i])
        if (np.mat(classLabels).T)[i]>0 and predict_mat[i]>0:
            TP+=1
        elif (np.mat(classLabels).T)[i]>0 and predict_mat[i]<0:
            FN+=1
        elif (np.mat(classLabels).T)[i]<0 and predict_mat[i]>0:
            FP+=1
        elif (np.mat(classLabels).T)[i]<0 and predict_mat[i]<0:
            TN+=1
    aggErrors = np.multiply(predict_mat != np.mat(classLabels).T, np.ones((m, 1))) #矩阵相乘，前者为0-1，后者全1
    errorRate = aggErrors.sum() / m
    time_end = time.time()    #结束计时
    print("错误率为: ", errorRate)    #正常的单变量可以直接不加%的
    print("精确率为: ", TP/(TP+FP))
    print("召回率为: ", TP/(TP+FN))
    print("F1:", 2*TP/(2*TP+FP+FN))
    print('times:', time_end - time_start, 's')   #运行所花时间
    #测试未知数据
    # res=pd.read_csv('res.csv')      #因为之前预处理的时候loan_id已经删了所以为了方便就直接做在表里面了，但是前提是res.csv文件必须提前存在！！
    # test_data,useless= loadDataSet('test_data.csv')      #导数据
    # predict_mat=adaClassify(test_data, classifierArr)  #接收回预测的值（而且为什么在这里的预测值就完全是-1？，因为训练的数据和测试的数据字段顺序根本不一样）
    # is_default=[j for i in predict_mat for j in i]      #这里还是没有将列表完全分开
    # res['isDefault']=is_default
    # res.to_csv('res.csv',index=False)

# 但是错误率却是1，为什么？
# 1>出现过拟合，因为本身0的个数就多。
# 2>adaboost不适合这种太高维的预测
#而且当你的预测值全为同一个符号时，很有可能是过拟合了，数据要预处理
#而且事实上这样改善之后确实可以减少错误率

