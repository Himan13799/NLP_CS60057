from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier

import untangle 
import nltk 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from numpy import asarray 
from numpy import zeros

homographic_test = "subtask1-homographic-test.xml"

heterographic_test = "subtask1-heterographic-test.xml"

homographic_gold = "subtask1-homographic-test.gold"

heterographic_gold = "subtask1-heterographic-test.gold"

input_data = []
result_data = []
sentenceList= []
obj = untangle.parse(homographic_test)

with open(homographic_gold) as file:
    for line in file:
        line = line.strip()
        temp = nltk.word_tokenize(line)
        result_data.append(temp[1])


i = 0
while ( i < len(obj.corpus)):
    sublist = []
    j = 0
    while( j < len(obj.corpus.text[i]) ):
        data = obj.corpus.text[i].word[j].cdata
        sublist.append(data)
        j = j+1
    input_data.append(sublist)
    i = i+1

for item in input_data:
    string = ' '.join(item)
    sentenceList.append(string)

lengthList = []
for sublist in input_data:
    lengthList.append(len(sublist)) 

sorted_lengthList =  sorted(lengthList, reverse = True)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentenceList)
vocab_size = len(tokenizer.word_index)+1

encoded_sent = tokenizer.texts_to_sequences(sentenceList)
 
maxLength = 50 
x = pad_sequences(encoded_sent,maxlen=maxLength,padding='pre',truncating= 'post', value= 0.5)

input_data1 = []
result_data1 = []
sentenceList1= []
obj1 = untangle.parse(heterographic_test)

with open(heterographic_gold) as file:
    for line in file:
        line = line.strip()
        temp = nltk.word_tokenize(line)
        result_data1.append(temp[1])


i = 0
while ( i < len(obj1.corpus)):
    sublist = []
    j = 0
    while( j < len(obj.corpus.text[i]) ):
        data = obj.corpus.text[i].word[j].cdata
        sublist.append(data)
        j = j+1
    input_data1.append(sublist)
    i = i+1

for item in input_data1:
    string = ' '.join(item)
    sentenceList1.append(string)

lengthList = []
for sublist in input_data1:
    lengthList.append(len(sublist)) 

sorted_lengthList =  sorted(lengthList, reverse = True)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentenceList1)
vocab_size = len(tokenizer.word_index)+1

encoded_sent1 = tokenizer.texts_to_sequences(sentenceList1)
 
maxLength = 50 
x1 = pad_sequences(encoded_sent1,maxlen=maxLength,padding='pre',truncating= 'post', value= 0.5)

fraction = 0.8
#print(len(x))
limit = int(fraction*len(x)) 
limit1 = int(fraction*len(x1))
x_train = x[:limit]
x_test = x[limit:]
x_train1 = x1[:limit1]
x_test1 = x1[limit1:]
y_train = result_data[:limit]
y_test = result_data[limit:]
y_train1 = result_data1[:limit1]
y_test1 = result_data1[limit1:]

print("     \t\t\t\t\tHomo    \t\t\t        \t\t\tHetero")
print("Model\t\t\t\t\tAccuracy\t\t\tF1 Score\t\t\tAccuracy\t\t\tF1 Score")

model = GaussianNB()

model.fit(x, result_data)

predicted= model.predict(x_test)

def svmClassifier(): 
	classifier = svm.SVC( kernel= 'rbf' ,C=500, gamma='auto')
	predictions = cross_val_predict(classifier, x, result_data, cv=4)
	y_pred = predictions
	accuracy = accuracy_score(result_data, predictions)
	f1 = f1_score(result_data, y_pred, labels=None, pos_label='1', average='binary', sample_weight=None)
	classifier1 = svm.SVC( kernel= 'rbf' ,C=500, gamma='auto')
	predictions1 = cross_val_predict(classifier1, x1, result_data1, cv=4)
	y_pred1 = predictions1
	accuracy1 = accuracy_score(result_data1, predictions1)
	f11 = f1_score(result_data1, y_pred1, labels=None, pos_label='1', average='binary', sample_weight=None)
	return [accuracy, f1, accuracy1, f11]
c1 = svmClassifier()
print("SVM Classifier:\t\t\t\t" + str(c1[0]) + "\t\t" +str(c1[1])+"\t\t"+str(c1[2])+"\t\t"+str(c1[3]))

#def adaBoostClassifier():
#	abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
# 		n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
#	abc.fit(x_train,y_train)
#	y_pred = abc.predict(x)
#	accuracy = accuracy_score(result_data, y_pred)
#	f1 = f1_score(result_data, y_pred, labels=None, pos_label='1', average='binary', sample_weight=None)
#	return [accuracy, f1]
#c2 = adaBoostClassifier()
#print("AdaBoost Classifier:\t\t\t" + str(c2[0]) + "\t\t" +str(c2[1]))

def randomForestClassifier():
	rfc = RandomForestClassifier(n_estimators=15)
	rfc.fit(x_train,y_train)
	y_pred = rfc.predict(x_test)
	#print(len(y_pred))
	accuracy = accuracy_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred, labels=None, pos_label='1', average='binary', sample_weight=None)
	rfc1 = RandomForestClassifier(n_estimators=15)
	rfc1.fit(x_train1,y_train1)
	y_pred1 = rfc1.predict(x_test1)
	#print(len(y_pred))
	accuracy1 = accuracy_score(y_test1, y_pred1)
	f11 = f1_score(y_test1, y_pred1, labels=None, pos_label='1', average='binary', sample_weight=None)
	return [accuracy, f1, accuracy1, f11]
c2 = randomForestClassifier()
print("Random Forest Classifier:\t\t" + str(c2[0]) + "\t\t" +str(c2[1])+"\t\t"+str(c2[2])+"\t\t"+str(c2[3]))