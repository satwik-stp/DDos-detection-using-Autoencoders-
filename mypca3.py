import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.layers import Input,Dense,LeakyReLU,Dropout
from keras.models import Model
from keras import optimizers
from sklearn.metrics import mean_squared_error
#problems with dataset na and inf 40420 both at 15 16 after renaming the column
def main():
#	packet=['packet'+str(i) for i in range(1,101)]
#	tme=['dimension'+str(i) for i in range(1,10)]
#	size=['dimension'+str(i) for i in range(10,21)]
#	print(tme)
	df=pd.read_csv('final_dataset.csv',header=None,low_memory=False,skiprows=6472648)
	print(df)
	print("--------------------------------------------------------------------------------------------------------------")
#	pd.set_option('display.max_rows', 100)
#	pd.set_option('display.max_columns', 100)
	
#	data=df
#	print(data)
#	pd.set_option('display.max_columns', 1000)
#	print(df)
	data=df.loc[1:,5:83]
	print(data[7])
	del(data[7])
#	for pac in data.index:
#		data.loc[pac,'dimension1':'dimension10']=np.random.poisson(lam=rd.randrange(10,10000),size=10)
#		data.loc[pac,'dimension11':'dimension20']=np.random.poisson(lam=rd.randrange(10,10000),size=10)
#	print(data)
	data=data.astype(float)
	data.columns=range(78)
	data=data.replace(np.inf,np.nan)
	data=data.fillna(0)
	#scaled_data=preprocessing.scale(data)
	scaler = preprocessing.StandardScaler()
	scaled_data= scaler.fit_transform(data)
#	print(data)
	print("----------------------------------------------------here------------------------------------------------")
	print(scaled_data)
#	print(scaled_data.T)
	encoded_dim_0=10
	
	encoding_dim_2=30
	input_features=Input(shape=(78,))
	lrelu=LeakyReLU(alpha=0.2)
	encoded_hidden_1=Dense(50)(input_features)
	act_1=lrelu(encoded_hidden_1)
	encoded_hidden_2=Dense(30)(act_1)
	act_2=lrelu(encoded_hidden_2)
	encoded_hidden_3=Dense(encoded_dim_0)(act_2)
	act_3=lrelu(encoded_hidden_3)
	final_encoded_layer=Dense(5)(act_3)
	Dropout(0.5)
	encoded=lrelu(final_encoded_layer)
	decoded_hidden_1=Dense(10)(encoded)
	dec_act_1=lrelu(decoded_hidden_1)
	decoded_hidden_2=Dense(30)(dec_act_1)
	dec_act_2=lrelu(decoded_hidden_2)
	decoded_hidden_3=Dense(50)(dec_act_2)
	dec_act_3=lrelu(decoded_hidden_3)
	decoded=Dense(78)(dec_act_3)
	autoencoder=Model(input_features,decoded)
	encoder=Model(input_features,encoded)
	encoded_input=Input(shape=(5,))
	decoder_layer=autoencoder.layers[-1]
	decode_layer2=autoencoder.layers[-2]
	decode_layer3=autoencoder.layers[-3]
	decoder_layer4=autoencoder.layers[-4]
	decoder=Model(encoded_input,decoder_layer(decode_layer2(decode_layer3(decoder_layer4(encoded_input)))))
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
	autoencoder.compile(optimizer='adam', loss='mse')
	autoencoder.fit(scaled_data, scaled_data,epochs=1000,verbose=1,batch_size=200,validation_split=0.1)#change this to 20 ,,, 1000 is for presentation purposes
	autoencoder.summary()
#	autoencoder.save("final_ddos.h5")	
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
