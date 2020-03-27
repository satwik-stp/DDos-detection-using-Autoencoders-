import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def main():
#	packet=['packet'+str(i) for i in range(1,101)]
#	tme=['dimension'+str(i) for i in range(1,10)]
#	size=['dimension'+str(i) for i in range(10,21)]
#	print(tme)
	df=pd.read_csv('final_dataset.csv',header=None,low_memory=False,nrows=100000)
#	pd.set_option('display.max_rows', 1000)
#	pd.set_option('display.max_columns', 1000)
	print(df)
	data=df.loc[1:,5:83]
	del(data[7])
	tmp='a'
	print('loading and striping done')
	print(data[21][1])
	for j in range(5,83):
		for i in range(1,len(data[21])):#12794627
			if(j==7):
				pass
			elif(type(data[j][i])==type(tmp)):
				data[j][i]=float(data[j][i])
#		print(f"{i/12794627}% done")
	print('conversions done')
	for j in range(5,83):
		for i in range(1,len(data[5])):
#			print(i)
			if(j==7):
				pass
			elif(type(data[j][i])==type(tmp)):
				print(j,type(data[j][i]))
				break
	print(tmp)

	data=data.fillna(0)
#	for pac in data.index:
#		data.loc[pac,'dimension1':'dimension10']=np.random.poisson(lam=rd.randrange(10,10000),size=10)
#		data.loc[pac,'dimension11':'dimension20']=np.random.poisson(lam=rd.randrange(10,10000),size=10)
#	print(data)
	data=data.astype(float)
#	scaled_data=preprocessing.scale(data)
#	pca=PCA()
#	pca.fit(scaled_data)
#	pca_data=pca.transform(scaled_data)
#	per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
#	labels=['PC'+str(x) for x in range(1,len(per_var)+1)]
#	plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
#	plt.ylabel('Percentage of Explained var')
#	plt.xlabel('Principal Component')
#	plt.title('screen plt')
#	plt.show()
#	pca_df=pd.DataFrame(pca_data,columns=labels)
	#pca_df=pca_df.T
#	pca_df=pca_df.loc[0:,"PC1":"PC10"]
	#pca_df=pca_df.T
#	print(pca_df)
#	plt.scatter(pca_df.PC1,pca_df.PC2)
#	plt.title('pca graph')
#	plt.xlabel(f'pc1-{per_var[0]}')
#	plt.ylabel(f'pc2-{per_var[1]}')
#	plt.show()
	km=KMeans(n_clusters=2)
	y=km.fit_predict(data)
	print(y)
	tots=len(y)
	count=0
	count2=0
	for i in y:
		if(i==1):
			count+=1
		count2+=1
		#print(i,df.loc[count,84])
	print(count/tots)
if(__name__=='__main__'):
	main()
