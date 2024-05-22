# Part 1: Preproccesing
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import gc

local_file="emails-1.csv"
df = pd.read_csv(local_file,sep=',',header=0,engine='python')
# Use pd.read_csv to read the csv
print(df[:10]) # Print the first 10 rows

# Data mining by tokenizing, stemming
# This method reduces words to their base or root form
def stem(text):
    ps=PorterStemmer()
    return [ps.stem(word) for word in text]

# Tokenize the words 
token=lambda x:word_tokenize(x)

# Transform elements
df['text']=df['text'].apply(token)
df['text']=df['text'].apply(lambda x:stem(x)) # 
df['text']=df['text'].apply(lambda x:' '.join(x))
print("Data after data mining:")
print(df.head())

# Vectorize the features
maxfea=5000
cv = CountVectorizer(max_features = maxfea, stop_words = 'english')
fea = cv.fit_transform(df['text']).toarray()

# Divide the dataset into training and testing
text_train, text_test, spam_train, spam_test= train_test_split(fea, np.array(df['spam']))

# Part 2: Logistic Regression
# Define the logistic regression algorithm in this class
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.l1 = nn.Linear(5000, 1000)
        self.l2 = nn.Linear(1000, 100)
        self.l3 = nn.Linear(100, 10)
        self.l4 = nn.Linear(10, 2)
    def forward(self, x):
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        x = torch.nn.functional.relu(self.l3(x))
        x = self.l4(x)
        return x
      
# Convert NumPy arrays into PyTorch tensors
text_train = Variable(torch.from_numpy(text_train)).float()
spam_train = Variable(torch.from_numpy(spam_train)).long()
text_test = Variable(torch.from_numpy(text_test)).float()
spam_test = Variable(torch.from_numpy(spam_test)).long()

# Collect garbage
gc.collect()

# The first classifier trained
# Logistic regression
# Adam optimizer
# Cross entropy loss
def LR(text_train,spam_train):
    LR = LogisticRegression()
    criterion = nn.CrossEntropyLoss()
    # Adam is an optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent:
    # Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).
    opt = torch.optim.Adam(params=LR.parameters() , lr=0.001)
    LR.train()
    n=30
    for epoch in range(n):
        #opt.zero_grad() clears old gradients, otherwise they would be added to the new gradients during the backward pass. 
        opt.zero_grad()
        predict = LR(text_train)
        #A loss function, also known as a cost function or objective function, is a crucial component in machine learning and neural networks. 
        #It measures how well a model's predictions match the actual target values. 
        #The goal of training a model is to minimize the loss function, thereby improving the model's performance.
        loss = criterion(predict, spam_train)
        pred = torch.max(predict, 1)[1].eq(spam_train).sum()
        acc = pred * 100.0 / len(text_train)
        print('Epoch:',epoch+1)
        print('loss:',loss.item())
        print('Accuracy:', acc.numpy())
        loss.backward()
        opt.step()
    return LR,criterion
LR,criterion=LR(text_train,spam_train)

# Evaluate the algorithm
LR.eval()
with torch.no_grad():
    predict = LR(text_test)
    loss = criterion(predict, spam_test)
    pred = torch.max(predict, 1)[1].eq(spam_test).sum()
    print("LR:")
    print ("Accuracy=",format(pred/len(text_test)))

# The second classifier trained
# Regularized logistic regression
# Adam optimizer with seight decay
# Cross entropy loss
def LR_Regularized(text_train,spam_train):
    LR_R = LogisticRegression()
    criterion = nn.CrossEntropyLoss()
    #weight_decay is an additional parameter that adds a penalty to the loss function. 
    #This technique is also known as L2 regularization.
    opt = torch.optim.Adam(params=LR_R.parameters() , lr=0.001, weight_decay=0.001)
    LR_R.train()
    n=30
    for epoch in range(n):
        opt.zero_grad()
        predict = LR_R(text_train)
        loss = criterion(predict, spam_train)
        pred = torch.max(predict, 1)[1].eq(spam_train).sum()
        acc = pred * 100.0 / len(text_train)
        print('Epoch:',epoch+1)
        print('loss:',loss.item())
        print('Accuracy:', acc.numpy())
        loss.backward()
        opt.step()
    return LR_R,criterion
LR_R,criterion=LR_Regularized(text_train,spam_train)

# Evaluate the algorithm
LR_R.eval()
with torch.no_grad():
    predict = LR_R(text_test)
    loss = criterion(predict, spam_test)
    pred = torch.max(predict, 1)[1].eq(spam_test).sum()
    print("LR with regularization:")
    print ("Accuracy=",format(pred/len(text_test)))

# Part 3: Decision Tree
# Iterate depth from 10 to 25
def DT(text_train,spam_train,text_test,md):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(max_depth=md)
    clf = clf.fit(text_train, spam_train)
    print(clf.get_depth())
    return clf.predict(text_test)
min_depth = 10
max_depth = 25
for depth in range(min_depth,max_depth+1):
    predict = DT(text_train,spam_train,text_test,depth)
    n=0
    for i in range(len(predict)):
        n+=(predict[i] == spam_test[i])
    print("Depth=",depth)
    print ("Accuracy: ",float(n/len(predict)))

# Part 4: random forest
# Iterate depth from 35 to 50
def random_forest(text_train, spam_train, text_test,md):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=md, random_state=0)
    clf.fit(text_train, spam_train)
    return clf.predict(text_test)
min_depth = 35
max_depth = 50
for dep in range(min_depth,max_depth+1):
    predict = random_forest(text_train, spam_train, text_test,dep)
    n=0
    for i in range(len(predict)):
        n+=(predict[i] == spam_test[i])
    print("Depth=",dep)
    print ("Accuracy: ",float(n/len(predict)))
