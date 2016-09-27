import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np 
import seaborn as sns
import scipy.spatial.distance as ssd
import pandas as pd 

iris = datasets.load_iris()
x = iris['data'] 
y = iris.target


def question1():
	global x,y 
	
	sns.set(color_codes=True)
	fig,ax = plt.subplots()
	ax.scatter(x[:,0][y == 0],x[:,1][y ==0],marker = '>',c = 'r',s = 50,label = 'Setosa') #Diff markers and colors for diff classes!
	ax.scatter(x[:,0][y == 1],x[:,1][y ==1],marker = '*',c = 'b',s = 50,label = 'Versicolor')
	ax.scatter(x[:,0][y == 2],x[:,1][y ==2],marker = 'o',c = 'g',s = 50,label = 'Virginica')
	
	legend = ax.legend(loc='upper right', shadow=True) # Legend customizations!
	for label in legend.get_texts():
		label.set_fontsize('large')

	plt.xlabel('sepal length')
	plt.ylabel('sepal width')
	plt.show()

def question2():
	global x,y 
	sns.set(color_codes=True)
	c = x[:,2]

	sns.distplot(c[:50],rug = True,bins = 6)
	sns.distplot(c[50:100],rug = True,bins = 6)
	sns.distplot(c[100:150],rug = True,bins = 6)
	plt.xlabel('petal length ')
	plt.show()

def maha(c,mean_col,inv_cov):
	m = []
	for i in range(c.shape[0]):
		m.append(ssd.mahalanobis(c.ix[i,:],mean_col,inv_cov) **2 )
	return (m)

def question3():
	global x,y
	a = x[:,2]
	b = x[:,3]

	class1 = zip(a[y == 0],b[y == 0])
	class2 = zip(a[y == 1],b[y == 1])
	class3 = zip(a[y == 2],b[y == 2])

	print np.cov(class1)
	print np.cov(class2)
	print np.cov(class3)

	c = x[:,2:]

	for i in range(3):
		l = []

		c = c[ 50*i : (i + 1)*50]	
		c = pd.DataFrame(c)
	
		sx = c.cov().values
		sx = np.linalg.inv(sx)

		mean = c.mean().values

		mahal = maha(c,mean,sx)
		print mahal
		mahal = np.asarray(mahal)

		for i,j,k in zip(mahal,c[0],c[1]):
			if i >= 4.0:
				l = l + [(j,k)]

		l = np.asarray(l)
		print l,l.shape
		try:
			sns.kdeplot(l[:,0],l[:,1],shade = True)
			
		except:
			print "No distance above 4 for class {i}".format(i = i+1)
		plt.show()



def all_in_one():
	iris = sns.load_dataset("iris")
	sns.pairplot(iris)
	g = sns.PairGrid(iris)
	g.map_diag(sns.kdeplot)
	g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);
	plt.show()

question3()
#all_in_one()
