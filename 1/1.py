import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df=pd.read_csv('/content/RS-A1_yield.csv')

print(df.columns)

df.dropna()

x=df[['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
y=df['hg/ha_yield']

xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=42)
rf=RandomForestRegressor(n_estimators=100,random_state=42).fit(xtr,ytr)

ximg,yimg=np.random.rand(100,64,64,3),np.random.randint(2,size=100)

cnn=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(1,activation='sigmoid')

])

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn.fit(ximg,yimg,epochs=3,batch_size=10)

test_img=np.random.rand(1,64,64,3)
dpred=cnn.predict(test_img).squeeze()
print(dpred)

ypred=rf.predict([[100,30,50]]).squeeze()
print(ypred)

def recommend(d_pred,y_pred):
    if d_pred>=0.5: return "Disease detected! Apply pesticide."
    if y_pred<50000: return "Low yield! Improve irrigation."
    return "Crop healthy, yield optimal."

print("Recommendation:", recommend(dpred,ypred))