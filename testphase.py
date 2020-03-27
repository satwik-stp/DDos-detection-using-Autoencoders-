from sklearn import preprocessing
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import random as rd
def main():
	df=pd.read_csv('final_dataset.csv',header=None,low_memory=False,skiprows=0+rd.randint(0,10000000),nrows=10)
	new_df=pd.read_csv('final_dataset.csv',header=None,low_memory=False,skiprows=6472648+rd.randint(0,1000000),nrows=10)
	data=df.loc[1:,5:83]
	malicious_data=new_df.loc[1:,5:83]
	del(data[7])
	del(malicious_data[7])
	data=data.astype(float)
	malicious_data=malicious_data.astype(float)
	data.columns=range(78)
	malicious_data.columns=range(78)
	data=data.replace(np.inf,np.nan)
	malicious_data=malicious_data.replace(np.inf,np.nan)
	data=data.fillna(0)
	malicious_data=malicious_data.fillna(0)
	#print(data)
	#scaled_data=preprocessing.scale(data)
	scaler = preprocessing.StandardScaler()
	scaled_data= scaler.fit_transform(data)
	mscaled_data=scaler.fit_transform(malicious_data)
	print (scaled_data[1])
	a=load_model("final_ddos.h5")
	b=a.predict(scaled_data)
	c=a.predict(mscaled_data)
	print(mean_squared_error(scaled_data, b))
	print(mean_squared_error(scaled_data,c))
if(__name__=="__main__"):
	main()
