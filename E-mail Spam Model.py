import pandas as pd
import seaborn as sns
import pickle

data=pd.read_csv('emails.csv',delimiter=',')

ham=data[data['spam']==0]
spam=data[data['spam']==1]
print('spam percentage: ',len(spam)/len(data['spam'])*100)
sns.countplot(x=data['spam'],label='count spam vs count')
output=data['spam'].values

from sklearn.feature_extraction.text import CountVectorizer

vectorizer= CountVectorizer()
spamham_convertor= vectorizer.fit_transform(data['text'])


from sklearn.naive_bayes import MultinomialNB

NB_classifier=MultinomialNB()
NB_classifier.fit(spamham_convertor,output)


x=spamham_convertor
y=output


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)

NB_classifier.fit(x_train,y_train)
y_pred_train=NB_classifier.predict(x_train)


inputt = []
test_vectorizer=vectorizer.transform(inputt)
predict=NB_classifier.predict(test_vectorizer)
if predict==1:
    print('It is a Spam!!!!!')
else:
    print('It is Not a Spam.')


filename='E-mail Spam Model.pkl'
pickle.dump(NB_classifier,open(filename,'wb'))
model=pickle.load(open(filename,'rb'))
