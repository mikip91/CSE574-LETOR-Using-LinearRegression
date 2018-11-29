
# coding: utf-8

# In[1]:


# These are all the imports required for the project.
from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt


# In[2]:


# These are the values for initializations
maxAcc = 0.0
maxIter = 0
# Trade off of error terms or it is a regularization term and is added to remove overfitting.
C_Lambda = 2 #[0.0001,0.001,0.01,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5]
# Data Partitions: Partition the data into a training set, a validation set, and a testing set.
# For testing, 80% of the data set is being used.
# For validation, 10% of the data set is used and for testing 10% is used.
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
# Number of Radiobasis function. 
# Converts the input vector x into a scalar value.
# Generalization unit that replaces each input with a function of the input.
M = 100 #[1,2,3,4,5,10,15,20,25,50,75,100]
# Stores the design matrix
# Each row of the design matrix gives the features for one training input vector.
PHI = []
IsSynthetic = False


# In[3]:


# This function returns the vector of output of the target vector.
# This is the first column in the LeTor Dataset provided.
# It is the relevance label and can contain the value 0, 1 or 2.
# Uses the file namely  - Querylevelnorm_t.csv
def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t

# This function reads the raw data provided in the Querylevelnorm_X.csv file and generates the datamatrix. 
# Total size of the matrix is 69623*46.
# 69623 are the instances and 46 represents the features.
# The final matrix will have 41 features.
# Only 80% of the data allocated to testing will get generated.
def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    # The feature values for 5,6,7,8,9 have a variance of 0 and hence they are being removed from the matrix.
    if IsSynthetic == False :
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)
    dataMatrix = np.transpose(dataMatrix)     
    #print ("Data Matrix Generated..")
    return dataMatrix
 
# This function allocates first 80% of the target vector to training target vector.
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

# This function allocates first 80% of the input datamatrix to training data set.
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

# This function allocates 10% data to validation training data set.
def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

# This function allocates 10% data to validation target data set.
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

# This is the Variance Matrix generation.
# The size of the matrix would be 41*41.
# (x-mu) would be a 41*1, Big sigma 41*41 and (x-mu)' is 1*41 where N is number of samples.
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

# This functions gives us a scalar value from a set of vectors. 
# This is the exponent part as described in equation 2 of Project1.2.pdf.
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

# This is Gaussian radial basis function.
# This converts the input vector x into a scalar value.
# This is the equation 2 mentioned in Project1.2.pdf.
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

# This is PHI design matrix for closed form solution.
# Each row of the design matrix gives the features for one training input vector.
# This is the matrix described in section 1.3 Closed-form Solution of project1.2.pdf.
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

# This is closed-form solution with least-squared regularization.
# This is the equation 8 described in Closed-form Solution of project1.2.pdf which is 
# \begin{equation*} w∗ = (λI + Φ'Φ)^−1*Φ't \end{equation*}
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

# def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
#     DataT = np.transpose(Data)
#     TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
#     PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
#     BigSigInv = np.linalg.inv(BigSigma)
#     for  C in range(0,len(MuMatrix)):
#         for R in range(0,int(TrainingLen)):
#             PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
#     #print ("PHI Generated..")
#     return PHI

# This function gives the value of linear regression function.
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

# This function returns accuracy and root mean square error.
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# ## Fetch and Prepare Dataset

# In[4]:


RawTarget = GetTargetVector('Querylevelnorm_t.csv')
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)


# ## Prepare Training Data

# In[5]:


# Prepares the training data
TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[6]:


# Prepares the validation data
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[7]:


# Prepares the testing data
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[8]:


ErmsArr = []
AccuracyArr = []
# k-means is an unsupervised algorithm and it aims to partition the n observations into k (≤ n) sets S = {S1, S2, …, Sk} so as to minimize the within-cluster sum of squares.
# n_clusters = The number of clusters to form as well as the number of centroids to generate.
# random_state = Determines the random number generation for centroid initialization.
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 

# TEST_PHI stores PHI matrix of testing data set.
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
# VAL_PHI stores PHI matrix of validation data set.
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[9]:


# This prints out dimensions of various matrices such as Validation PHI matrix, Testing PHI matrix etc.
print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 

# In[10]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)

# Dot product of the weights and basis function of the input vectors which is y(x,w) = w'φ(x)
TEST_OUT     = GetValTest(TEST_PHI,W)

# Getting the accuracy and the error function for training, testing and validation data set.
# This compares predicted to the actual target values.
TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[11]:


print ('UBITname      = mikipadh')
print ('Person Number = 50286289')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = 10 \nLambda = 0.9")
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
print ("Training Dataset Accuracy   = " + str(float(TrainingAccuracy.split(',')[0])))
print ("Validation Dataset Accuracy = " + str(float(ValidationAccuracy.split(',')[0])))
print ("Testing Dataset Accuracy = " + str(float(TestAccuracy.split(',')[0])))


# ## Gradient Descent solution for Linear Regression

# In[12]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[13]:


# This is the current weight.
W_Now        = np.dot(220, W)
# This is regularization term which has been added to reduce overfitting. 
# This is a form of regression, that constrains/ regularizes or shrinks the coefficient estimates towards zero
La           = 2
# Learning Rate is an hyper-parameter that controls how much we are adjusting the weights of our network. 
# The lower the value of learning rate the slower we travel along the downward slope.
learningRate = 0.02
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    # ∇ ED = −(tn −w(τ)'φ(xn))φ(xn)\n
    # This represents equation 11 of project1.2.pdf
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    # This represents equation 12 of project1.2.pdf
    La_Delta_E_W  = np.dot(La,W_Now)
    # This represents equation 10 of project1.2.pdf
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
     # This represents equation 9 of project1.2.pdf
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    # ∇ E = ∇ ED + λ*∇ EW\n
    # Stochastic Gradient Descent Solution for w
    
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[14]:


print ('----------Gradient Descent Solution--------------------')
print ("M = 15 \nLambda  = 0.0001\neta=0.01")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
print ("Training Dataset Accuracy  = " + str(float(Erms_TR.split(',')[0])))
print ("Validation Dataset Accuracy = " + str(float(Erms_Val.split(',')[0])))
print ("Testing Dataset Accuracy    = " + str(float(Erms_Test.split(',')[0])))


# In[ ]:




