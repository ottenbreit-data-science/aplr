import numpy as np
import aplr

import pandas as pd

d=pd.read_fwf("data/auto-mpg.data",header=None) #loading the data

d.columns=["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin","car_name"] #Column names
d["origin_text"]=d["origin"].replace([1,2,3],['USA','Europe','Japan']) #Converting origin to text for readable dummy variables
d=pd.concat([d,pd.get_dummies(d["origin_text"])],axis=1) #Creating dummies for origin because it is a nominal variable

terms=["cylinders","displacement","horsepower","weight","acceleration","model_year","USA","Europe","Japan"]
response="mpg"

d["horsepower"]=pd.to_numeric(d["horsepower"],errors="coerce")
d.dropna(inplace=True)
random_state=2020
d_train=d.sample(frac=0.7,random_state=random_state).sort_index()
d_test=d[~d.index.isin(d_train.index)].dropna()

r=aplr.APLRRegressor(verbosity=2)
r.fit(d_train[terms].values,d_train[response].values)
out=r.predict(d_test[terms].values)
print(np.corrcoef(out,d_test[response]))