from os import listdir
import numpy
import operator
import cv2

def imProcess(imagePath):
    testDigits = listdir(imagePath)
    for i in range(len(testDigits)):
        imageName = testDigits[i]#图像命名格式为N_M.png，NM含义见4）生成训练样本
        #imageClass = int((imageName.split('.')[0]).split('_')[0])#这个图像的数字是多少
        image = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        ret, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        with open(r'F:\Users\yang\PycharmProjects\OCR_KNN\testDigits\\'+imageName.split('.')[0]+'.txt','w+') as file:
            for i in range(32):
                for j in range(32):
                    if image[i][j] == 255:
                        file.write('0')
                    else:
                        file.write('1')
                file.writelines('\n')

def img2vector(filename):
    returnvec = numpy.zeros((1,1024))
    file = open(filename)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            returnvec[0,32*i+j] = int(line[j])
    return returnvec

def handWritingClassifyTest():
    labels=[]
    trainingFile = listdir(r'F:\Users\yang\PycharmProjects\OCR_KNN\trainingDigits')
    m = len(trainingFile)
    trainingMat = numpy.zeros((m,1024))
    for i in range(m):
        file = trainingFile[i]
        filestr = file.strip('.')[0]
        classnum = int(filestr.strip('_')[0])
        labels.append(classnum)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % file)
    testFileList = listdir(r'F:\Users\yang\PycharmProjects\OCR_KNN\testDigits')
    error = 0.0
    testnum = len(testFileList)
    for i in range(testnum):
        file_test = testFileList[i]
        filestr_test = file_test.strip('.')[0]
        classnum_test = int(filestr_test.strip('_')[0])
        vector_test = img2vector('testDigits/%s'%file_test)
        result = classify(vector_test,trainingMat,labels,1)
        if(result!=classnum_test):error+=1.0
    print("准确率：%f"%(1.0-(error/float(testnum))))

def classify(inX,dataSet,labels,k):
    size = dataSet.shape[0]
    distance = (((numpy.tile(inX,(size,1))-dataSet)**2).sum(axis=1))**0.5
    sortedDistance = distance.argsort()
    count = {}
    for i in range(k):
        label = labels[sortedDistance[i]]
        count[label]=count.get(label,0)+1
    sortedcount = sorted(dict2list(count),key=operator.itemgetter(1),reverse=True)
    return sortedcount[0][0]

def dict2list(dic:dict):#将字典转换为list类型
    keys=dic.keys()
    values=dic.values()
    lst=[(key, value)for key,value in zip(keys,values)]
    return lst

# def imProcess(image):
#     image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
#     ret, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
#     cv2.imshow('result',image)
#     cv2.waitKey(0)
#     with open(r'F:\Users\yang\PycharmProjects\OCR_KNN\testDigits\6_0.txt','w+') as file:
#         for i in range(32):
#             for j in range(32):
#                 if image[i][j] == 255:
#                     file.write('0')
#                 else:
#                     file.write('1')
#             file.writelines('\n')



# iamge = cv2.imread(r'C:\Users\yang\Desktop\6.png',cv2.IMREAD_GRAYSCALE)
# image = imProcess(iamge)
imProcess(r'F:\Users\yang\PycharmProjects\OCR_KNN')
handWritingClassifyTest()