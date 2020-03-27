from sklearn import preprocessing
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import random as rd
import time
def main():
	a=load_model("final_ddos.h5")
	tmp2=0
#	count=0
	daf=pd.read_csv('unbalaced_20_80_dataset.csv',header=None,low_memory=False,nrows=6482648)
	while(True):
		offset=rd.randint(0,6482618)
		end=offset+30
		df=daf.iloc[offset:end]
#		print(df)
		df=df.reset_index(drop=True)
#		print(df)
#		start=time.time()	
#		df=pd.read_csv('unbalaced_20_80_dataset.csv',header=None,low_memory=False,skiprows=count+rd.randint(100,6472648),nrows=30)
#		intermediate=time.time()
#		count+=10
#		new_df=pd.read_csv('unbalaced_20_80_dataset.csv',header=None,low_memory=False,skiprows=6472648+rd.randint(0,1000),nrows=10)
		data=df.loc[1:,5:83]
#		malicious_data=new_df.loc[1:,5:83]
		del(data[7])
#		del(malicious_data[7])
		data=data.astype(float)
#		malicious_data=malicious_data.astype(float)
		data.columns=range(78)
#		malicious_data.columns=range(78)
		data=data.replace(np.inf,np.nan)
#		malicious_data=malicious_data.replace(np.inf,np.nan)
		data=data.fillna(0)
#		malicious_data=malicious_data.fillna(0)
		#print(data)
		#scaled_data=preprocessing.scale(data)
		scaler = preprocessing.StandardScaler()
		scaled_data= scaler.fit_transform(data)
#		mscaled_data=scaler.fit_transform(malicious_data)
#		print (scaled_data[1])
		b=a.predict(scaled_data)
#		c=a.predict(mscaled_data)
		predicted=""
		tmp2=mean_squared_error(scaled_data,b)
		if(tmp2>0.54):
			print(tmp2,"actual",df[84][0],"predicted:ddos","ddos" in df[84][0])
		else:
			print(tmp2,"actual",df[84][0],"predicted:normal","Benign" in df[84][0])
#		tmp=mean_squared_error(scaled_data,c)
#		if(tmp<1):
#			print(tmp,"actual",new_df[84][0],"predicted:ddos")
#		else:
#			print(tmp,"actual",new_df[84][0],"predicted:normal")
#		print((intermediate-start)/(time.time()-start))
		time.sleep(0.5)

if(__name__=="__main__"):
	main()
