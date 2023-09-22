#import all libraries and modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

#create data frame
df = pd.read_csv('email_info.csv')

#removes rows that have missing values
df.dropna(subset=['Mail', 'Class'], inplace=True)

#maps spam and ham
df['Class'] = df['Class'].map({'spam': 0, 'ham': 1})
X = df['Mail']
y = df['Class']

#train using 80% of the data and test using 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#created tfid vectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#trains model
email_mod = LogisticRegression()
email_mod.fit(X_train_tfidf, y_train)

#make predictions on training and test data, then output the accuracy scores
y_train_pred = email_mod.predict(X_train_tfidf)
accuracy_train = accuracy_score(y_train, y_train_pred)
print("Training accuracy:", accuracy_train)

y_test_pred = email_mod.predict(X_test_tfidf)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Test accuracy:", accuracy_test)

#input value is entered then transformed using tfid vectorizer
test_mail = ["Congratulations, you've won a free iPhone! Click the link to claim your prize now!"]
test_mail_tfidf = tfidf_vectorizer.transform(test_mail)
output = email_mod.predict(test_mail_tfidf)

#print output
if output[0] == 0:
    print("Predicted as spam")
else:
    print("Predicted as ham")

