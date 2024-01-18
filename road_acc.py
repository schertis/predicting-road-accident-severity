#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 23:41:50 2024

@author: salmacherti
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


#Load dataset
carac=pd.read_csv(r'/Users/salmacherti/Desktop/ProjetML/carcteristiques-2022.csv',sep=';')
lieu=pd.read_csv(r'/Users/salmacherti/Desktop/ProjetML/lieux-2022.csv',sep=';')
pers=pd.read_csv(r'/Users/salmacherti/Desktop/ProjetML/usagers-2022.csv',sep=';')
voit=pd.read_csv(r'/Users/salmacherti/Desktop/ProjetML/vehicules-2022.csv',sep=';')


#------Data preprocessing-----------

#CARACTERISTIQUES ==>  '''circonstances générales de l'accident'''

'''mettre en format la date'''
def get_date(df):
    columns=['jour', 'mois', 'an']
    df['date']=df[columns].apply(lambda row:'/'.join(row.values.astype(str)),axis=1)
    df['date']=df['date']+' '+df['hrmn']
    df=df.drop(['jour', 'mois', 'an','hrmn'],axis=1)
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M'))
    return df
carac=get_date(carac)



'''clustering de la longitude et latitude'''
carac['lat']=[x.replace(',','.').replace(' ','') for x in carac['lat'] ]
carac['long']=[x.replace(',','.').replace(' ','') for x in carac['long'] ]
carac['lat']=carac['lat'].astype(float)
carac['long']=carac['long'].astype(float)
data=carac[['lat','long']]
data=data.values.astype(np.float64)
# Specify the number of clusters (k)
k = 5
# Fit the k-means model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)
# Get cluster labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_
# Visualize the results
plt.scatter(data[:, 1], data[:, 0], c=labels, cmap='viridis', marker='o')
#plt.scatter(centers[:, 1], centers[:, 0], c='red', marker='x', s=200, label='Cluster Centers')
plt.title('K-Means Clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
result=pd.concat([pd.DataFrame(data),pd.DataFrame(labels)],axis=1,ignore_index=True)
result.rename(columns={0:'lat',1:'long',2:'cluster'},inplace=True)
carac=pd.merge(carac,result,how='inner',on=['lat','long'])



'''remplacer les valeurs -1 par nan'''
columns=['atm', 'col','lum','dep', 'com', 'agg', 'int']
carac[columns]=carac[columns].mask(carac[columns].isin([-1,'-1']), np.nan)
carac['dep']=carac['dep'].replace('2A',1000).replace('2B',1001)
carac['com']=[x.replace('2B','1001').replace('2A','1000').replace('N/C','-1') for x in carac['com']]



'''rename accident columns'''
carac=carac.rename(columns={'Accident_Id':'Num_Acc'})

carac['dep']=carac['dep'].astype(int)
carac['com']=carac['com'].astype(int)

'''preprocess time data'''
carac['month'] = carac['date'].dt.month
carac['day'] = carac['date'].dt.day
carac['day_of_week'] = carac['date'].dt.dayofweek

'''
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

carac=encode(carac,'month',12)
carac=encode(carac,'day',31)'''

''' drop not wanted columns'''
#columns=['lat','long','adr','date','day','month']
columns=['lat','long','adr','date']
carac=carac.drop(columns,axis=1)


del centers,columns,data,k,kmeans,labels,result


#LIEUX

'''remplacer les valeurs -1 par nan'''
lieu['nbv']=lieu['nbv'].replace('#ERREUR',-1)
lieu['nbv']=[int(str(x).replace(' ','')) for x in lieu['nbv'] ]
columns=['circ', 'vosp','prof','pr','pr1','plan','surf','infra','situ','nbv','vma']
lieu[columns]=lieu[columns].mask(lieu[columns].isin([-1,'-1']), np.nan)

''' drop not wanted columns'''
columns=['lartpc','v2','v1','voie','larrout']
lieu=lieu.drop(columns,axis=1)

del columns


#VEHICULES

'''replacer les valeurs -1 par nan'''
columns=['senc','obs','obsm','choc','manv','motor','catv']
voit[columns]=voit[columns].mask(voit[columns].isin([-1,'-1']), np.nan)


'''drop not wanted columns'''
columns=['occutc','num_veh']
voit=voit.drop(columns,axis=1)
'''calculer le nombre de vehicule par accident'''
nbr=voit.groupby('Num_Acc')['id_vehicule'].nunique().reset_index().rename(columns={'id_vehicule':'nbr_voit_accident'})
voit=pd.merge(voit,nbr,how='inner',on='Num_Acc')
del columns,nbr

#USAGERS
'''replacer les valeurs -1 par nan'''
columns=['trajet','secu1','secu2','secu3','locp','actp','etatp','sexe']
pers['actp']=pers['actp'].replace('B',-1).replace('A',10)
pers['actp']=[int(str(x).replace(' ','')) for x in pers['actp']]
pers[columns]=pers[columns].mask(pers[columns].isin([-1,'-1']), np.nan)

'''compute the age of the user'''
pers['age']=[2024-int(x)  if not pd.isna(x) else np.nan for x in pers['an_nais']]


'''drop not wanted columns and rows where gravitiy not given'''
columns=['secu3','etatp','an_nais','num_veh']
pers=pers.drop(columns,axis=1)
pers=pers.loc[pers['grav']!=-1]
'''convert columns'''
pers['id_usager']=pers['id_usager'].astype(str)
pers['id_vehicule']=pers['id_vehicule'].astype(str)


del columns





#CONCATENATION DES DATASET
data1=pd.merge(carac,lieu,how='inner',on='Num_Acc')
data2=pd.merge(data1,voit,how='inner',on='Num_Acc').drop_duplicates().reset_index(drop=True)


data=pd.merge(data2,pers,how='inner',on=['Num_Acc','id_vehicule']).drop_duplicates().reset_index(drop=True)


del data1,data2,carac,lieu,pers,voit

data.to_csv(r'/Users/salmacherti/Desktop/ProjetML/data.csv',sep=';')



#DATA PREPROCESSING
''' SUR LES COLONNES PR ET PR1 PRÉSENCE D'UNE VAL INCONNUE NON CORRI, drop id vehicule et usager et accident'''
data=data.drop(['pr','pr1','id_vehicule','Num_Acc','id_usager'],axis=1)

'''remplace empty by mean or most used value'''
columns_to_impute_mean=[ 'nbv', 'vma', 'age']

columns_to_impute_most_frequent=['lum', 'dep', 'com', 'agg', 'int', 'atm', 'col', 'catr',  'circ', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ', 
       'senc', 'catv', 'obs', 'obsm', 'choc', 'manv', 'motor', 'place', 'catu', 'sexe', 'trajet', 'secu1','secu2', 'locp', 'actp']

remaining_columns = [col for col in data.columns if col not in columns_to_impute_mean + columns_to_impute_most_frequent]


column_transformer = ColumnTransformer(
    transformers=[
        ('mean_imputer', SimpleImputer(strategy='mean'), columns_to_impute_mean),
        ('most_frequent_imputer', SimpleImputer(strategy='most_frequent'), columns_to_impute_most_frequent),
    ],
    remainder='passthrough'
)

# Create a pipeline with the ColumnTransformer
pipeline = Pipeline(steps=[('preprocessor', column_transformer)])

# Fit and transform the DataFrame using the pipeline
data_imputed = pd.DataFrame(pipeline.fit_transform(data), columns=columns_to_impute_mean + columns_to_impute_most_frequent + remaining_columns)



#MODEL

X=data_imputed.drop(columns=['grav'],axis=1)
y=data_imputed['grav']


'''One hot encoding'''
encoded_df = pd.get_dummies(X, drop_first=True)
encoded_df.shape

y_en = pd.Series(y)


'''SMONTEC'''
# importing the SMOTENC object from imblearn library 
from imblearn.over_sampling import SMOTENC

# categorical features for SMOTENC technique for categorical features
n_cat_index = np.array(range(3,38))

# creating smote object with SMOTENC class
smote = SMOTENC(categorical_features=n_cat_index, random_state=42, n_jobs=True)
X_n, y_n = smote.fit_resample(encoded_df,y_en)

# print the shape of new upsampled dataset
X_n.shape, y_n.shape




'''MACHINE LEARNING MODELING'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score,accuracy_score

# train and test split and building baseline model to predict target features
X_trn, X_tst, y_trn, y_tst = train_test_split(X_n, y_n, test_size=0.2, random_state=42)

# modelling using random forest baseline RANDOM FOREST
rf = RandomForestClassifier(n_estimators=1200, max_depth=45, random_state=42)
rf.fit(X_trn, y_trn)

# predicting on test data
predics = rf.predict(X_tst)

# train score 
rf.score(X_trn, y_trn)


#Test data
# classification report on test dataset
classif_re = classification_report(y_tst,predics)
print(classif_re)

# f1_score of the model
f1score = f1_score(y_tst,predics, average='weighted')
print(f1score)




# Get feature importances
feature_importances = rf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(feature_importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} ({feature_importances[indices[f]]})")

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_trn.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X_trn.shape[1]), indices)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.show()




#Adaboost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

base_classifier = DecisionTreeClassifier(max_depth=3)

# Create AdaBoost classifier
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)

# Train the model
adaboost_classifier.fit(X_trn, y_trn)

# predicting on test data
predics = rf.predict(X_tst)

# train score 
rf.score(X_trn, y_trn)
classif_re = classification_report(y_tst,predics)
print(classif_re)

f1score = f1_score(y_tst,predics, average='weighted')
print(f1score)







#Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier



# Define the base classifier (e.g., DecisionTreeClassifier)
base_classifier = DecisionTreeClassifier(max_depth=30)

# Define the BaggingClassifier
rf = BaggingClassifier(base_classifier, n_estimators=40, random_state=42)
rf.fit(X_trn, y_trn)

# predicting on test data
predics = rf.predict(X_tst)

# train score 
rf.score(X_trn, y_trn)


#Test data
# classification report on test dataset
classif_re = classification_report(y_tst,predics)
print(classif_re)

# f1_score of the model
f1score = f1_score(y_tst,predics, average='weighted')
print(f1score)





















