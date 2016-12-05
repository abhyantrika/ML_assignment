import numpy as np 
from sklearn import linear_model
from sklearn import svm
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def q1(x,y):

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

	print "The classifier used is a linear kernel SVM"
	clf = svm.SVC(kernel = 'linear',C=1).fit(X_train,y_train)

	print "Normal Testing Scores "
	print clf.score(X_test,y_test)

	print "The predictions after 10-fold Cross validation "
	predicted = cross_val_predict(clf, x,y, cv=10)
	print predicted

	print "10-fold CV Scores"
	scores = cross_val_score(clf,x,y,cv =10)
	print scores
	print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

	print "Confusion Matrix"
	conf = confusion_matrix(y,predicted)
	print conf

iris = datasets.load_iris()
y = iris.target
x = iris.data

pca = PCA(n_components=2)
x_reduced = pca.fit_transform(x)
print pca.explained_variance_ratio_
#print x_reduced


q1(x,y)
q1(x_reduced,y)
