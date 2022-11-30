
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Ignore Warnings.
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv('pacific.csv')
data.head()
import re
def hemisphere(seperation):
        hem = re.findall(r'[NSWE]' ,seperation)[0]
        if hem == 'N' or hem == 'E':
            return 0
        else:
            return 1
        
data['Latitude_Hemisphere'] = data['Latitude'].apply(hemisphere)
data['Longitude_Hemisphere'] = data['Longitude'].apply(hemisphere)

data['Latitude'] =  data['Latitude'].apply(lambda x: re.match('[0-9]{1,3}.[0-9]{0,1}' , x)[0])
data['Longitude'] =   data['Longitude'].apply(lambda x: re.match('[0-9]{1,3}.[0-9]{0,1}' , x)[0])

for column in data.columns:
    missing_cnt = data[column][data[column] == -999].count()
    print('Missing Values in column {col} = '.format(col = column) , missing_cnt )
    if missing_cnt!= 0:
#         print('in ' , column)
        mean = round(data[column][data[column] != -999 ].mean())
#         print("mean",mean)
        index = data.loc[data[column] == -999 , column].index
#         print("index" , index )
        data.loc[data[column] == -999 , column] = mean
#         print(df.loc[index , column])

df = data.drop(columns=["ID","Name","Date","Time","Event"])
df['Latitude'] = data['Latitude'].astype('float')
df['Longitude'] = data['Longitude'].astype('float')

replace_map = {'Status': {'TS': 1, 'TD': 2, 'HU': 3, 'LO': 4,
                                  'DB': 5, 'ET': 6, 'EX': 7 , 'SS': 8 , 'ST': 9,'PT': 10,'SD': 11}}

labels = df['Status'].astype('category').cat.categories.tolist()
replace_map_comp = {'Status' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

print(replace_map_comp)

df.replace(replace_map_comp, inplace=True)

print(df.head())

df.drop(df[df['Status']==1].index, inplace = True)
df.drop(df[df['Status']==2].index, inplace = True)
df.drop(df[df['Status']==3].index, inplace = True)
df.drop(df[df['Status']==5].index, inplace = True)
df.drop(df[df['Status']==7].index, inplace = True)
df.drop(df[df['Status']==8].index, inplace = True)
df.drop(df[df['Status']==9].index, inplace = True)
df.drop(df[df['Status']==10].index, inplace = True)

df.Status.replace((12,11,4,6),('Tropical storm-34 to 63 knots','Tropical depression-v< 34 knots','hurricane- > 64 knots','no cyclone'),inplace=True)



X = df.drop(['Status'],axis='columns')
X.head(3)

y = df.Status
y.head(3)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

# Import Random Forest
from sklearn.ensemble import RandomForestClassifier

# First I want to determine the important features.
rf = RandomForestClassifier(oob_score=True , n_estimators=1000)
rf.fit(X_train , y_train)
features = pd.Series(rf.feature_importances_ , index= X_train.columns).sort_values(ascending=False)
features

x_trainf = df[features.index[0:6]]
y_train = df['Status']


from sklearn.model_selection import train_test_split
X_trains, X_tests, y_trains, y_tests = train_test_split(x_trainf,y_train,test_size=0.2,random_state=20)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=30)
model.fit(X_trains, y_trains)

y_predicted = model.predict(X_tests)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_tests, y_predicted)
cm


import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


from sklearn.metrics import classification_report
print(classification_report(y_tests, y_predicted))

row = [67,995,16.7,117.0,40,35]
# predict the class label
new_data = model.predict([row])

print(new_data)


import pickle
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[67,995,16.7,117.0,40,35]]))
    
