import pandas as pd
import numpy as np
student=pd.read_csv(r"D:\coding journey\aiml\python\task\data set of ML project\student\student-mat.csv")
print(student)
#student['school;sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;guardian;traveltime;studytime;failures;schoolsup;famsup;paid;activities;nursery;higher;internet;romantic;famrel;freetime;goout;Dalc;Walc;health;absences;G1;G2;G3']=student['school;sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;guardian;traveltime;studytime;failures;schoolsup;famsup;paid;activities;nursery;higher;internet;romantic;famrel;freetime;goout;Dalc;Walc;health;absences;G1;G2;G3'].str.split(";",expand=True)[1]
#print(student['school;sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;guardian;traveltime;studytime;failures;schoolsup;famsup;paid;activities;nursery;higher;internet;romantic;famrel;freetime;goout;Dalc;Walc;health;absences;G1;G2;G3'])
column = []
for i in student.columns:
    list = []
    for j in i:

        if j == ";":
            column.append("".join(list))
            list.clear()
            continue
        list.append(j)
df=pd.DataFrame()
column.append("G3")
print(column)


for i in range(0,33):
    df[column[i]]=student['school;sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;guardian;traveltime;studytime;failures;schoolsup;famsup;paid;activities;nursery;higher;internet;romantic;famrel;freetime;goout;Dalc;Walc;health;absences;G1;G2;G3'].str.split(";",expand=True)[i]
print(df.columns)
for i in df.columns:
    print(i," : ",df[i].unique())
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
print(df['sex'].value_counts())
df=df.drop(columns=["school"])
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
df["G3"]=df["G3"].astype(int)

sns.barplot(x=df["sex"],y=df["G3"])
plt.show()
print(df.info())
#Onehote encoding:-["sex","address",'reason']
#ordinaryencodinf:-["famsize","schoolsup","famsup","paid",'activities','higher',"internet",'romantic']
print(df["age"].value_counts())
df["age"]=df["age"].astype(int)
df["G3"]=df["G3"].astype(int)

df=df[df["age"]<20]

sns.lineplot(x=df["age"],y=df["G3"])
plt.show()

print(df['address'].value_counts())
sns.barplot(x=df["address"],y=df["G3"])
plt.show()

print(df['famsize'].value_counts())
sns.barplot(x=df['famsize'],y=df["G3"])
plt.show()
print(pd.crosstab(df['famsize'],df["Pstatus"]))


print(df["Pstatus"].value_counts())
sns.barplot(x=df['Pstatus'],y=df["G3"],hue=df['famsize'])
plt.show()
#i have observed from  the data that rather than parents being apart the child perforance depend more on its size
df=df.drop(columns=["Pstatus"])

df["Medu"]=df["Medu"].astype(int)
df["Fedu"]=df["Fedu"].astype(int)
print(pd.crosstab(df["Medu"],df["Fedu"]))
sns.lineplot(x=df["Fedu"],y=df["G3"],color="r")
sns.lineplot(x=df["Medu"],y=df["G3"],color="g")
plt.legend()
plt.show()
sns.scatterplot(x=df["Fedu"],y=df["Medu"],hue=df["G3"])
plt.show()
def parents(f,m):
    if(f==0 and m==0 or f==0 and  m==1 or f==1 and m==0 or f==0 and m==2 or f==2 and m==0 or f==1 and m==2):
        return 0
    elif(  f==0 and m==3  or f==1 and m==3 or   f==2 and m==1 or f==2 and m==2 or f==3 and m==0 or f==3 and m==1):
        return 1
    else:
        return 2
df["parets-education"]=df.apply(lambda x: parents(x["Fedu"],x["Medu"]),axis=1)
print(df["parets-education"].value_counts())
sns.barplot(x=df["parets-education"],y=df["G3"])
plt.show()

df=df.drop(columns=["Medu","Fedu"])
print(df.columns)

print(pd.crosstab(df['Mjob'],df['Fjob']))
sns.barplot(x=df['Mjob'],y=df['G3'])
plt.show()
sns.barplot(x=df['Fjob'],y=df['G3'])
plt.show()

def job(f, m):


    if f == "at_home" and m == "at_home":
        return "NO"
    elif f == "at_home":
        return "mother"
    elif m == "at_home":
           return "father_earner"
    else:
        return "both"

# Apply the function to create the new column
df["Fjob"] = df.apply(lambda x: job(x["Fjob"], x["Mjob"]), axis=1)
print(df["Fjob"].value_counts())
df=df.drop(columns=["Fjob","Mjob"])
print(df['reason'].value_counts())
sns.barplot(x=df['guardian'],y=df['G3'])
plt.show()
print(df['guardian'].value_counts())
df=df.drop(columns=['guardian'])
df['traveltime']=df['traveltime'].astype(int)
df['studytime']=df['studytime'].astype(int)
sns.lineplot(x=df['traveltime'],y=df['G3'])
plt.show()
sns.scatterplot(x=df['traveltime'],y=df['G3'])
plt.show()
print(pd.crosstab(df['traveltime'],df['studytime']))
sns.lineplot(x=df['traveltime'],y=df['studytime'])
plt.show()
sns.lineplot(x=df['studytime'],y=df['G3'])
plt.show()


print(df['failures'].value_counts())
sns.barplot(x=df['failures'],y=df['G3'])
plt.show()
df['failures']=df['failures'].astype(int)

print(df['schoolsup'].value_counts())
sns.barplot(x=df['schoolsup'],y=df['G3'])
plt.show()

print(df['famsup'].value_counts())
sns.barplot(x=df['famsup'],y=df['G3'])
plt.show()

print(df['paid'].value_counts())
sns.barplot(x=df['paid'],y=df['G3'])
plt.show()

print(df['activities'].value_counts())
print(df['activities'].value_counts())
sns.barplot(x=df['activities'],y=df['G3'])
plt.show()

print(df['nursery'].value_counts())
sns.barplot(x=df['nursery'],y=df['G3'])
plt.show()
df=df.drop(columns=['nursery'])

print(df['higher'].value_counts())
sns.barplot(x=df['higher'],y=df['G3'])
plt.show()

print(df['internet'].value_counts())
sns.barplot(x=df['internet'],y=df['G3'])
plt.show()

sns.violinplot(x='internet',y='G3',data=df)
plt.show()

sns.boxplot(x='internet',y='G3',data=df)
plt.show()

print(df['romantic'].value_counts())
sns.barplot(x=df['romantic'],y=df['G3'])
plt.show()

sns.violinplot(x='romantic',y='G3',data=df)
plt.show()

sns.boxplot(x='romantic',y='G3',data=df)
plt.show()

print(df['famrel'].value_counts())
df['famrel']=df['famrel'].astype(int)
sns.barplot(x=df['famrel'],y=df['G3'])
plt.show()
df=df.drop(columns=['famrel'])

print(df['freetime'].value_counts())
df['freetime']=df['freetime'].astype(int)
sns.barplot(x=df['freetime'],y=df['G3'])
plt.show()

print(df['goout'].value_counts())
df['goout']=df['goout'].astype(int)
sns.barplot(x=df['goout'],y=df['G3'])
plt.show()

print(df['Dalc'].value_counts())
df['Dalc']=df['Dalc'].astype(int)
sns.barplot(x=df['Dalc'],y=df['G3'])
plt.show()

print(df['Walc'].value_counts())
df['Walc']=df['Walc'].astype(int)
sns.barplot(x=df['Walc'],y=df['G3'])
plt.show()
df=df.drop(columns=['Walc','Dalc'])

print(df['health'].value_counts())
df['health']=df['health'].astype(int)
sns.barplot(x='health',y='G3',data=df)
plt.show()

print(df['absences'].value_counts())
df['absences']=df['absences'].astype(int)
sns.lineplot(x='absences',y='G3',data=df)
plt.show()

df['G1'] = df['G1'].str.replace('"', '').astype(int)
print(df['G1'])
sns.scatterplot(x=df['G1'],y=df['G3'])
plt.show()


df['G2'] = df['G2'].str.replace('"', '').astype(int)
print(df['G1'])
sns.scatterplot(x=df['G2'],y=df['G3'])
plt.show()

df['a1']=(df["G1"]+df["G2"])/40
sns.scatterplot(x=df['a1'],y=df['G3'])
plt.show()
df=df.drop(columns=["G1","G2"])
print(df.columns)
from sklearn.model_selection import train_test_split
x=df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21]]
y=df.iloc[:, 19]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
ct = ColumnTransformer(
    transformers=[

        ("tnf2", OneHotEncoder(sparse=False, drop="first"), ["sex", "address", "reason","famsize","schoolsup","famsup","paid",'activities','higher',"internet",'romantic'])
    ],
    remainder="passthrough"
)
x_train=ct.fit_transform(x_train)
x_test=ct.transform(x_test)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
list={"Linear regression":LinearRegression(),"Decision tree":DecisionTreeRegressor(random_state=50),"SVM":SVR(kernel="linear"),"RAndomforest":RandomForestRegressor(n_estimators=100)}
accuracy={}
for name,model in list.items():
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    accuracy[name]=r2_score(y_test,y_predict)

print(accuracy)