import pandas as apd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FuncFormatter
df=pd.read_csv(r"D:\coding journey\aiml\python\task\data set of ML project\LInear regression\House Price India.csv")
print(df.sample(5))
print(df.columns)
print(df.shape)
df=df.drop(columns=['id', 'Date'])
print(df.columns)
print(df.isna().sum())
print(df.info())
print(df["number of bedrooms"].value_counts())
df=df[df['number of bedrooms']<8]
sns.scatterplot(x=df['number of bedrooms'],y=df["Price"])
plt.show()
df["number of bathrooms"]=df["number of bathrooms"].astype(int)
print(df["number of bathrooms"].value_counts())
df=df[df['number of bathrooms']<8]
sns.scatterplot(x=df["number of bathrooms"],y=df["Price"])
plt.show()

print(df["living area"].value_counts())
sns.scatterplot(x=df["living area"],y=df["Price"])
plt.show()

print(df["number of floors"].value_counts())
df=df[df['number of floors']<3.5]

sns.barplot(x=df["number of floors"],y=df["Price"])
plt.show()

print(df["lot area"].value_counts())
sns.scatterplot(x=df["lot area"],y=df["Price"])
plt.xlim(3000,30000)
plt.show()

print(df["waterfront present"].value_counts())

sns.barplot(x=df["waterfront present"],y=df["Price"])
plt.show()

print(df["number of views"].value_counts())

sns.barplot(x=df["number of views"],y=df["Price"])
plt.show()

print(df["condition of the house"].value_counts())
sns.barplot(x=df["condition of the house"],y=df["Price"])
plt.show()

print(df["grade of the house"].value_counts())
sns.barplot(x=df["grade of the house"],y=df["Price"])
plt.show()

sns.scatterplot(x=df["Area of the house(excluding basement)"],y=df["living area"],hue=df["Price"])
plt.show()

sns.scatterplot(x=df["Area of the house(excluding basement)"],y=df["Price"])
plt.show()

sns.scatterplot(x=df["Area of the basement"],y=df["Price"])
plt.show()
#df=df.drop(columns=["Area of the basement"]) #if result is less i will use it

print(df["Built Year"].value_counts())
sns.lineplot(x=df["Built Year"],y=df["Price"])
plt.show()
sns.scatterplot(x=df["Built Year"],y=df["Price"],hue=df['Renovation Year'])
plt.show()


df=df.drop(columns=['Postal Code',"Longitude"])

sns.scatterplot(x=df["living_area_renov"],y=df["Price"])
plt.show()

sns.scatterplot(x=df["lot_area_renov"],y=df["Price"])
plt.show()
df=df.drop(columns=['lot_area_renov'])

print(df["Number of schools nearby"].value_counts())
sns.barplot(x=df["Number of schools nearby"],y=df["Price"])
plt.show()
df=df.drop(columns=["Number of schools nearby"])

print(df["Distance from the airport"].value_counts())
sns.scatterplot(x=df["Distance from the airport"],y=df["Price"])
plt.show()
df=df.drop(columns=['Distance from the airport'])
df['total_area'] = df['living area'] + df['lot area']

x=df.drop(columns=["Price"])
y=df["Price"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
x_train=st.fit_transform(x_train)
x_test=st.transform(x_test)
print(df.corr()["Price"])
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
list={"Linear regression":LinearRegression(),"Decision tree":DecisionTreeRegressor(random_state=50),"SVM":SVR(kernel="linear"),"RAndomforest":RandomForestRegressor(n_estimators=200)}
accuracy={}
for name,model in list.items():
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    accuracy[name]=r2_score(y_test,y_predict)

print(accuracy)

