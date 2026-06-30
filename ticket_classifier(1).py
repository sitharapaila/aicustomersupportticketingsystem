from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

texts=['payment issue','login problem']
labels=['billing','technical']

vec=TfidfVectorizer()
X=vec.fit_transform(texts)
model=MultinomialNB().fit(X,labels)

print('Model Ready')
