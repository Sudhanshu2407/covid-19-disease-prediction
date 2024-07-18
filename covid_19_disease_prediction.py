#------------------------------------------------------------
#Here we import the Important Libraries.

#Data Processing Library.
import pandas as pd

#Linear Algebra Library.
import numpy as np

#Data Visualization Library.
import matplotlib.pyplot as plt

#Statistic Visualization Library.
import seaborn as sns

#Model Selection Library.
from sklearn.model_selection import train_test_split,cross_val_score

#Data preprocessing Library.
from sklearn.preprocessing import StandardScaler,LabelEncoder

#Linear Model Library.
from sklearn.linear_model import LogisticRegression

#Support Vector Machine Library.
from sklearn.svm import SVC

#Knearest Neighbour Library.
from sklearn.neighbors import KNeighborsClassifier

#Naive Bayes Library.
from sklearn.naive_bayes import GaussianNB,BernoulliNB

#Decision tree Library.
from sklearn.tree import DecisionTreeClassifier

#Random Forest Library.
from sklearn.ensemble import RandomForestClassifier

#Gradient Boosting Library.
from sklearn.ensemble import GradientBoostingClassifier

#Xtream Gradient Boosting Library.
from xgboost import XGBClassifier

#Light Gradient Boosting Library.
from lightgbm import LGBMClassifier

#Model statistic library.
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#Pickle Library.
import pickle

#Import warning Handling Library.
import warnings

#Here we avoid/ignore the warnings.
warnings.filterwarnings("ignore")

#------------------------------------------------------
#Now here we read the dataset.
df=pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\covid-19_prediction\covid_symptoms.csv")

#-------------------------------------------------------
#Here we print the dataframe.
print(df)

#-------------------------------------------------------
#Here we get the top record from the dataset.
print(df.head())

#-------------------------------------------------------
#Now here we check the data types of each column of dataset.
df.dtypes

#There are 6 numerical columns and 3 categorical column.

#-------------------------------------------------------
#Now we check the number of null values contained by each column of dataset.
df.isnull().sum()

#-------------------------------------------------------
#Now here we check the percentage of null value contained by each column.
df.isnull().sum()/len(df)

#-------------------------------------------------------
#Here we check the shape of dataset.
df.shape

#Tha shape of dataset is (211429,9).i.e there are 211429 instances and 9 features.

#--------------------------------------------------------
#As null values are very less in number,so we can afford to remove it from dataset.
df.dropna(inplace=True)

#--------------------------------------------------------
#Now we again check the shape of dataset.
df.shape

#Now the shape is (211429,9).

#---------------------------------------------------------
#Now we check is there any null values present in the dataset.
df.isnull().sum()

#Now no column contain any null value.

#---------------------------------------------------------
#Now here we check the names of columns of dataset.
df.columns

#----------------------------------------------------------
#Now we convert the categorical column to numerical column.
cat_cols=['age_60_and_above', 'gender', 'test_indication']

#-----------------------------------------------------------
#Here we check the value counts and unique values of each categorical column.
for col in cat_cols:
    print(f"The number of unique classes of {col} column is: {len(df[col].unique())}")
    print(f"The value count of {col} column is: ")
    print(df[col].value_counts())
    print()


#---------------------------------------------------------------
#There are two ways to do it,one is we replace manually and also we can do it by labelencoder.


#-----------------------------------------------------------
#Firstly we encode age_60_and_above column.
df=df.replace(to_replace="No", value=0.0)
df=df.replace(to_replace="Yes", value=1.0)

#------------------------------------------------------------
#Now we encode gender column.
df=df.replace(to_replace="Female", value=0.0)
df=df.replace(to_replace="Male", value=1.0)

#------------------------------------------------------------
#Now we encode test_indication column.
df=df.replace(to_replace="Other", value=0.0)
df=df.replace(to_replace="Abroad", value=1.0)
df=df.replace(to_replace="Contact with confirmed", value=2.0)


#------------------------------------------------------------------------
#Now we again check the data type of each column.
df.dtypes

#Now all the column become numeric.

#---------------------------------------------------------------
#Now here we find the correlation between the features of the dataset.
corr=df.corr()

corr

#---------------------------------------------------------------
#Here we find the correlation of all different features with target variable.

corr["corona_result"].sort_values(ascending=False)

#As this is mostly depend on test_indication,fever,cough,head_ache.

#---------------------------------------------------------------
#Now here we create the heatmap for the correlation matrix.
sns.heatmap(corr,cmap="YlGnBu")

#----------------------------------------------------------------------
#Now here we do some visualization.

#(1) Here we find the countplot of target variable.

sns.countplot(data=df,x="corona_result")
plt.title("countplot of corona_result")
plt.xlabel("corona_result")
plt.ylabel("count")
plt.show()

#-------------------------------------------------------------------
#Here we find the value count of this target variable.
df["corona_result"].value_counts()

#Conclusion: corona_result 1.0-->107472 and 0.0-->98586.

#-------------------------------------------------------------------
#Here we find the null accuracy of target variable.

null_accuracy=107472/(107472+98586)

null_accuracy

#Conclusion: As the null accuracy is 52.15%.

#-------------------------------------------------------------------
#(2) Now here we find the countplot of all the feature except target variable by take target variable as hue.

cols=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
      'age_60_and_above', 'gender', 'test_indication']


for col in cols:
    sns.countplot(data=df,x=col,hue="corona_result")
    plt.title(f"countplot of {col} column.")
    plt.xlabel(f"{col}")
    plt.ylabel("count")
    plt.show()
    
    
#--------------------------------------------------------------------
#(3) Now here we create the barplot for all above columns.

for col in cols:
    sns.barplot(data=df,x=col,y="corona_result")
    plt.title(f"countplot of {col} column.")
    plt.xlabel(f"{col}")
    plt.ylabel("corona_result")
    plt.show()

#As all above plots give some insights,and we make the conclusions and get insights.

#----------------------------------------------------------------------
#Now here we check the value counts of each column and their level.

for col in df.columns:
    print(f"The value count of {col} column is: ")
    print(df[col].value_counts())
    print()
    print(f"The level of {col} column is: {len(df[col].unique())}")
    print()
    
#As no column level is greater than 3,so it is good for prediction.

#------------------------------------------------------------------------
#Now here we check the dependent and independent feature.

x=df.drop("corona_result",axis=1) #Independent features.

y=df["corona_result"] #Dependent feature.

#-------------------------------------------------------------------------
#Now here we check the shape of dependent and independent feature.

print(f"The shape of Independent feature is: {x.shape}")

#The shape of independent feature is: (206058,8).

print(f"The shape of dependent feature is: {y.shape}")

#The shape of dependent feature is: (206058,).

#-------------------------------------------------------------------------
#Here we split the dataset into train and test data in 75-25 ratio with random state 0.

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#--------------------------------------------------------------------------
#Here we check the shape of x-train/test and y-train/test.

print(f"The shape of x_train is: {x_train.shape}")

#The shape of x_train is: (164846,8).

print(f"The shape of x_test is: {x_test.shape}")

#The shape of x_test is: (41212,8).

print(f"The shape of y_train is: {y_train.shape}")

#The shape of y_train is: (164846,).

print(f"The shape of y_test is: {y_test.shape}")

#The shape of y_test is: (41212,).

#---------------------------------------------------------------------------
#Now here we do the standard scaling of the train data.

#Here we create the standard scaler object.
sc=StandardScaler()

#Here we scaled the x_train.
x_train_sc=sc.fit_transform(x_train)

#Here we scaled the x_test.
x_test_sc=sc.transform(x_test)

#---------------------------------------------------------------------------
#Now here we apply different classification model.

'''
models={
  "Logistic Regression":LogisticRegression(),
  "Support Vector Machine":SVC(),
  "Knearest Neighbour":KNeighborsClassifier(),    
  "Gaussian Naive Bayes":GaussianNB(),      
  "Bernoulli Naive Bayes":BernoulliNB(),      
  "Decision Tree":DecisionTreeClassifier(),      
  "Random Forest":RandomForestClassifier(),
  "Gradient Boosting":GradientBoostingClassifier(),
  "Xtream Gradient Boosting":XGBClassifier(),
  "Light Gradient Boosting":LGBMClassifier()      
        }


for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test, y_pred)
    bias=model.score(x_train,y_train)
    variance=model.score(x_test,y_test)
    cm=confusion_matrix(y_test, y_pred)
    cr=classification_report(y_test, y_pred)
    
    print(f"The accuracy of {name} model is: {accuracy}.")
    print()
    print(f"The bias of {name} model is: {bias}.")
    print()
    print(f"The variance of {name} model is: {variance}.")
    print()
    print(f"The confusion matrix of {name} model is: ")
    print(cm)
    print()
    print(f"The classification report of {name} model is: ")
    print(cr)
    print()
    print()
    
'''

#---------------------------------------------------------------------
#Now we apply each model separately.

#-------------------------------------------------------------------
#(1) Logistic Regression on original data.

lor=LogisticRegression()

lor.fit(x_train,y_train)
y_pred_lor=lor.predict(x_test)
accuracy_lor=accuracy_score(y_test, y_pred_lor)
bias_lor=lor.score(x_train,y_train)
variance_lor=lor.score(x_test,y_test)
cm_lor=confusion_matrix(y_test, y_pred_lor)
cr_lor=classification_report(y_test, y_pred_lor)

print(f"The accuracy of lor model is: {accuracy_lor}.")
print()
print(f"The bias of lor model is: {bias_lor}.")
print()
print(f"The variance of lor model is: {variance_lor}.")
print()
print("The confusion matrix of lor model is: ")
print(cm_lor)
print()
print("The classification report of lor model is: ")
print(cr_lor)
print()



#-------------------------------------------------------------------
#(2) Logistic Regression on Scaled data.

lor1=LogisticRegression()

lor1.fit(x_train_sc,y_train)
y_pred_lor1=lor1.predict(x_test_sc)
accuracy_lor1=accuracy_score(y_test, y_pred_lor1)
bias_lor1=lor1.score(x_train_sc,y_train)
variance_lor1=lor1.score(x_test_sc,y_test)
cm_lor1=confusion_matrix(y_test, y_pred_lor1)
cr_lor1=classification_report(y_test, y_pred_lor1)

print(f"The accuracy of lor1 model is: {accuracy_lor1}.")
print()
print(f"The bias of lor1 model is: {bias_lor1}.")
print()
print(f"The variance of lor1 model is: {variance_lor1}.")
print()
print("The confusion matrix of lor1 model is: ")
print(cm_lor1)
print()
print("The classification report of lor1 model is: ")
print(cr_lor1)
print()


#-------------------------------------------------------------------
#(3) knearest neighbour on original data.

'''
knc=KNeighborsClassifier()

knc.fit(x_train,y_train)
y_pred_knc=knc.predict(x_test)
accuracy_knc=accuracy_score(y_test, y_pred_knc)
bias_knc=knc.score(x_train,y_train)
variance_knc=knc.score(x_test,y_test)
cm_knc=confusion_matrix(y_test, y_pred_knc)
cr_knc=classification_report(y_test, y_pred_knc)

print(f"The accuracy of knc model is: {accuracy_knc}.")
print()
print(f"The bias of knc model is: {bias_knc}.")
print()
print(f"The variance of knc model is: {variance_knc}.")
print()
print("The confusion matrix of knc model is: ")
print(cm_knc)
print()
print("The classification report of knc model is: ")
print(cr_knc)
print()



#-------------------------------------------------------------------
#(4) knearest neighbour on scaled data.

knc1=KNeighborsClassifier()

knc1.fit(x_train_sc,y_train)
y_pred_knc1=knc1.predict(x_test_sc)
accuracy_knc1=accuracy_score(y_test, y_pred_knc1)
bias_knc1=knc1.score(x_train_sc,y_train)
variance_knc1=knc1.score(x_test_sc,y_test)
cm_knc1=confusion_matrix(y_test, y_pred_knc1)
cr_knc1=classification_report(y_test, y_pred_knc1)

print(f"The accuracy of knc1 model is: {accuracy_knc1}.")
print()
print(f"The bias of knc1 model is: {bias_knc1}.")
print()
print(f"The variance of knc1 model is: {variance_knc1}.")
print()
print("The confusion matrix of knc1 model is: ")
print(cm_knc1)
print()
print("The classification report of knc1 model is: ")
print(cr_knc1)
print()



#-------------------------------------------------------------------
#(5) Support Vector on original data.

svc=SVC()

svc.fit(x_train,y_train)
y_pred_svc=svc.predict(x_test)
accuracy_svc=accuracy_score(y_test, y_pred_svc)
bias_svc=svc.score(x_train,y_train)
variance_svc=svc.score(x_test,y_test)
cm_svc=confusion_matrix(y_test, y_pred_svc)
cr_svc=classification_report(y_test, y_pred_svc)

print(f"The accuracy of svc model is: {accuracy_svc}.")
print()
print(f"The bias of svc model is: {bias_svc}.")
print()
print(f"The variance of svc model is: {variance_svc}.")
print()
print("The confusion matrix of svc model is: ")
print(cm_svc)
print()
print("The classification report of svc model is: ")
print(cr_svc)
print()

'''

#----------------------------------------------------------------------
#(6) Decision tree classifier on original data.

dtc=DecisionTreeClassifier()

dtc.fit(x_train,y_train)

y_pred_dtc=dtc.predict(x_test)

ac_dtc=accuracy_score(y_test,y_pred_dtc)
print(f"The accuracy of dtc is: {ac_dtc}")

bias_dtc=dtc.score(x_train,y_train)
print(f"The bias of dtc is: {bias_dtc}")

variance_dtc=dtc.score(x_test,y_test)
print(f"The variance of dtc is: {variance_dtc}")


#------------------------------------------------------------------------
#(7) Random Forest classifier on original data.

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

y_pred_rfc=rfc.predict(x_test)

ac_rfc=accuracy_score(y_test,y_pred_rfc)
print(f"The accuracy of rtc is: {ac_rfc}")

bias_rfc=rfc.score(x_train,y_train)
print(f"The bias of rfc is: {bias_rfc}")

variance_rfc=rfc.score(x_test,y_test)
print(f"The variance of rfc is: {variance_rfc}")

#-------------------------------------------------------------------------
#(8) Gradient Boosting classifier on original data.

gbc=GradientBoostingClassifier()

gbc.fit(x_train,y_train)

y_pred_gbc=gbc.predict(x_test)

ac_gbc=accuracy_score(y_test, y_pred_gbc)
print(f"The accuracy of gbc is: {ac_gbc}.")

bias_gbc=gbc.score(x_train,y_train)
print(f"The bias of gbc is: {bias_gbc}")

variance_gbc=gbc.score(x_test,y_test)
print(f"The variance of gbc is: {variance_gbc}")

#---------------------------------------------------------------------------
#(9) Xtream gradient boosting on original data.

xgb=XGBClassifier()

xgb.fit(x_train,y_train)

y_pred_xgb=xgb.predict(x_test)

ac_xgb=accuracy_score(y_test,y_pred_xgb)
print(f"The accuracy of xgb is: {ac_xgb}")

bias_xgb=xgb.score(x_train,y_train)
print(f"The bias of xgb is: {bias_xgb}")

variance_xgb=xgb.score(x_test,y_test)
print(f"The variance of xgb is: {variance_xgb}") 

#---------------------------------------------------------------------------
#(10) light gradient boosting on original data.

lgb=LGBMClassifier()

lgb.fit(x_train,y_train)

y_pred_lgb=lgb.predict(x_test)

ac_lgb=accuracy_score(y_test,y_pred_lgb)
print(f"The accuracy of lgb is: {ac_lgb}")

bias_lgb=lgb.score(x_train,y_train)
print(f"The bias of lgb is: {bias_lgb}")

variance_lgb=lgb.score(x_test,y_test)
print(f"The variance of lgb is: {variance_lgb}") 

#----------------------------------------------------------------------------
#Now we have to find roc_curve.

from sklearn.metrics import roc_auc_score,roc_curve

pred=[y_pred_lor,y_pred_dtc,y_pred_rfc,y_pred_gbc,y_pred_xgb,y_pred_lgb]


for col in pred:
    fpr,tpr,threshold=roc_curve(y_test,col)
    
    plt.plot(fpr,tpr,linewidth=3)
    
    plt.plot([0,1],[0,1],"k--")
    
    plt.title("Roc-auc-curve")
    
    plt.xlabel("fpr/1-specificity")
    
    plt.ylabel("tpr/sensitivity")
    
    plt.legend()
    
    plt.show()

#----------------------------------------------------------------------------
#Now here we have to find the roc_auc_score.

for col in pred:
    score=roc_auc_score(y_test,col)
    print(f"The roc_auc_score of {col} column is: {score}")
    
    
    
#-----------------------------------------------------------------------------
#Now here we have to find the cross-val-score of each model.

cols=[lor,dtc,rfc,gbc,xgb,lgb]


for col in cols:
  cross_validation_score=cross_val_score(estimator=col,X=x_train,y=y_train,cv=5)
  print(f"The cross_val_score of {col} model is: {cross_validation_score.mean()}")
  

#-----------------------------------------------------------------------------
#Now we check the statistics of our best mpodel.(rfc model)

#Here we find the confusion matrix of rfc model.
cm_rfc=confusion_matrix(y_test, y_pred_rfc)
print("The confusion matrix of rfc model is: ")
print(cm_rfc)
print()

#Now here we find the dataframe of confusion matrix.
cm_rfc_df=pd.DataFrame(cm_rfc,columns=["Actual Yes:1","Actual No:0"],index=["Predicted Yes:1","Predicted No:0"])
print("The dataframe of confusion matrix of rfc model is: ")
print(cm_rfc_df)

#Now here we represent the confusion matrix in the form of heatmap.
sns.heatmap(cm_rfc_df,annot=True,cmap="YlGnBu")
plt.title("Heatmap of confusion matrix is: ")
plt.show()

#Here we find the terms related to confusion matrix.
tp_rfc=cm_rfc[0,0] #True Positive.
tn_rfc=cm_rfc[1,1] #True Negative.
fp_rfc=cm_rfc[0,1] #False Positive.
fn_rfc=cm_rfc[1,0] #False Negative.
  

#Now here we find the terminologies of confusion matrix.
accuracy_rfc=(tp_rfc+tn_rfc)/(tp_rfc+tn_rfc+fp_rfc+fn_rfc)
print(f"The accuracy of rfc model using confusion matrix is: {accuracy_rfc}")

error_rfc=1-accuracy_rfc
print(f"The error of rfc model using confusion matrix is: {error_rfc}")

precision_rfc=(tp_rfc)/(tp_rfc+fp_rfc)
print(f"The precision of rfc model using confusion matrix is: {precision_rfc}")

recall_rfc=(tp_rfc)/(tp_rfc+fn_rfc)
print(f"The recall of rfc model using confusion matrix is: {recall_rfc}")

f1_score_rfc=(2*precision_rfc*recall_rfc)/(precision_rfc+recall_rfc)
print(f"The f1-score of rfc model using confusion matrix is: {f1_score_rfc}")

sensitivity_rfc=tpr_rfc=(tp_rfc)/(tp_rfc+fn_rfc)
print(f"The true positive rate/sensitivity of rfc model using confusion matrix is: {tpr_rfc}")

fpr_rfc=(fp_rfc)/(tn_rfc+fp_rfc)
print(f"The false positive rate of rfc model using confusion matrix is: {fpr_rfc}")

specificity_rfc=tn_rfc/(tn_rfc+fp_rfc)
print(f"The specificity of rfc model using confusion matrix is: {specificity_rfc}")


#Now here we find the classification report of rfc model.
cr_rfc=classification_report(y_test, y_pred_rfc)
print("The classification report of rfc model is: ")
print(cr_rfc)


#----------------------------------------------------------------------------
#Now we have to save the best model.(i.e rfc model)

pickle.dump(rfc,open(r"C:\sudhanshu_projects\project-task-training-course\covid-19_prediction\covid_19_prediction.pkl","wb"))

#----------------------------------------------------------------------------
#Here we load the save model.

model=pickle.load(open(r"C:\sudhanshu_projects\project-task-training-course\covid-19_prediction\covid_19_prediction.pkl","rb"))

#Now here we test the model.

model.score(x_test,y_test)
