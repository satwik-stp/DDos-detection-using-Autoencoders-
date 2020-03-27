import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.layers import Input,Dense,LeakyReLU
from keras.models import Model
from keras import optimizers
def main():
#	packet=['packet'+str(i) for i in range(1,101)]
#	tme=['dimension'+str(i) for i in range(1,10)]
#	size=['dimension'+str(i) for i in range(10,21)]
#	print(tme)
	df=pd.read_csv('final_dataset.csv',header=None,low_memory=False, nrows=10000)
#	pd.set_option('display.max_rows', 100)
#	pd.set_option('display.max_columns', 100)
	
	data=df
#	print(data)
#	pd.set_option('display.max_columns', 1000)
#	print(df)
	data=df.loc[1:,5:83]
	del(data[7])
#	for pac in data.index:
#		data.loc[pac,'dimension1':'dimension10']=np.random.poisson(lam=rd.randrange(10,10000),size=10)
#		data.loc[pac,'dimension11':'dimension20']=np.random.poisson(lam=rd.randrange(10,10000),size=10)
#	print(data)
#	data.astype(float)
	data.columns=range(78)
	scaled_data=preprocessing.scale(data)
	print(data)
	print(scaled_data)
#	print(scaled_data.T)
	encoded_dim_0=10
	
	encoding_dim_2=30
	input_features=Input(shape=(78,))
	lrelu=LeakyReLU(alpha=0.2)
	encoded_hidden_1=Dense(50,activation=lrelu)(input_features)
	encoded_hidden_2=Dense(30,activation=lrelu)(encoded_hidden_1)
	encoded_hidden_3=Dense(encoded_dim_0,activation=lrelu)(encoded_hidden_2)
	encoded=Dense(5,activation=lrelu)(encoded_hidden_3)
	decoded_hidden_1=Dense(10,activation=lrelu)(encoded)
	decoded_hidden_2=Dense(30,activation=lrelu)(decoded_hidden_1)
	decoded_hidden_3=Dense(50,activation=lrelu)(decoded_hidden_2)
	decoded=Dense(78,activation=lrelu)(decoded_hidden_3)
	autoencoder=Model(input_features,decoded)
	encoder=Model(input_features,encoded)
	encoded_input=Input(shape=(5,))
	decoder_layer=autoencoder.layers[-1]
	decode_layer2=autoencoder.layers[-2]
	decode_layer3=autoencoder.layers[-3]
	decoder_layer4=autoencoder.layers[-4]
	decoder=Model(encoded_input,decoder_layer(decode_layer2(decode_layer3(decoder_layer4(encoded_input)))))
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	autoencoder.compile(optimizer='sgd', loss='mse')
	autoencoder.fit(scaled_data, scaled_data,epochs=1000,verbose=1)
	
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
#	pca_df=pca_df.loc[0:,"PC1":"PC20"]
	#pca_df=pca_df.T
#	print(pca_df)
#	plt.scatter(pca_df.PC1,pca_df.PC2)
#	plt.title('pca graph')
#	plt.xlabel(f'pc1-{per_var[0]}')
#	plt.ylabel(f'pc2-{per_var[1]}')
#	plt.show()
#	km=KMeans(n_clusters=2)
#	y=km.fit_predict(data)
#	print(y)
#	tots=len(y)
#	count=0
#	count2=0
#	input()
#	for i in y:
#		if(i==1):
#			count+=1
#		count2+=1
#		print(i,df.loc[count,84])
#	print(count/tots)
	
if(__name__=='__main__'):
	main()
