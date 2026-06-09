import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.model_selection import cross_val_score, cross_val_predict , train_test_split
from sklearn.pipeline import Pipeline
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
await download(filepath, "insurance.csv")
file_name="insurance.csv"
# Import the dataset
df = pd.read_csv(file_name)
headers = ['age',
'gender',
'bmi',	
'no_of_children',
'smoker',	
'region',	
'charges']
df.columns = headers

# Data Wrangling
df.replace('?',np.NaN,inplace=True)
m = df['age'].astype('float').mean(axis=0)
df['age'].replace(np.nan,m,inplace=True)
i = df['smoker'].value_counts().idxmax()
df['smoker'].replace(np.nan,i,inplace=True)
df[['age','smoker']].astype('int')
df[['charges']]= np.round(df[['charges']],2)

# Exploratory Data Analysis (EDA)
sns.regplot(y=df['charges'],x=df['bmi'],data=df,line_kws={'color':'red'})
sns.boxplot(x=df['smoker'],y=df['charges'])
df.corr()

# Model Development
X = df[['smoker']]
Y = df['charges']
lm = LinearRegression()
lm.fit(X,Y)
print(lm.score(X, Y))
Z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]
lm.fit(Z,Y)
print(lm.score(Z, Y))
Input = [('sccale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(Input)
Z = Z.astype('float')
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
print(r2_score(Y,ypipe))

# Model Refinement
x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.2, random_state=1)
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test,yhat))
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))
