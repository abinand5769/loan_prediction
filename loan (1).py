#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("loan_prediction.csv")
df


# In[2]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


#drop the loan id column 


# In[7]:


df=df.drop("Loan_ID",axis=1)


# In[8]:


df.head()


# In[9]:


#data has missing values 


# In[10]:


df.isna().sum()


# In[11]:


df.describe()


# In[12]:


# Fill missing values in categorical columns with mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)


# In[13]:


df.head()


# In[14]:


df.isna().sum()


# In[15]:


#Fill missing values in LoanAmount with the median


# In[16]:


df['LoanAmount'].fillna(df['LoanAmount'].median(),inplace=True)


# In[17]:


#Fill missing values in Loan_Amount_Term with the mode


# In[18]:


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)


# In[19]:


df.isna().sum()


# In[20]:


# Fill missing values in Credit_History with the mode


# In[21]:


df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)


# In[22]:


df.isna().sum()


# # Exploratory Data Analysis

# In[23]:


import plotly.express as px


# In[25]:


gender_count=df['Gender'].value_counts()
gender_count


# In[26]:


fig_gender=px.bar(gender_count,x=gender_count.index,y=gender_count.values,title='gender distribution')
fig_gender.show()


# In[27]:


married_count=df['Married'].value_counts()


# In[28]:


married_count


# In[29]:


education_count=df['Education'].value_counts()
education_count


# In[30]:


fig_education=px.bar(education_count,
                    x=education_count.index,
                    y=education_count.values,
                    title='education distribution')


# In[31]:


fig_education.show()


# In[32]:


#relation ship between of the income of the loan application and loan ststus


# In[33]:


fig_income=px.box(df,x='Loan_Status',
                 y='ApplicantIncome',
                 color='Loan_Status',
                 title='loan_sttus vs applicantincome')
fig_income.show()


# In[34]:


#applicantincome have outliers so we can remove it 


# In[35]:


#calculate the IQR
q1=df['ApplicantIncome'].quantile(0.25)
q3=df['ApplicantIncome'].quantile(0.75)
IQR=q3-q1


# In[36]:


#define the lower and upper bound for outliers


# In[37]:


lower_bond=q1-1.5 * IQR
upper_bond=q1+1.5 * IQR


# In[38]:


lower_bond


# In[39]:


upper_bond


# In[40]:


#remove outliers


# In[41]:


df=df[(df['ApplicantIncome']>=lower_bond) &(df['ApplicantIncome']<=upper_bond)]


# In[42]:


fig_income=px.box(df,x='Loan_Status',
                 y='ApplicantIncome',
                 color='Loan_Status',
                 title='loan_sttus vs applicantincome')
fig_income.show()


# In[43]:


df.columns


# In[44]:


df['CoapplicantIncome']


# In[45]:


fig_coapplicant_income=px.box(df,x='Loan_Status',
                             y='CoapplicantIncome',
                             color='Loan_Status',
                             title='loan status vs coapplicant income')
fig_coapplicant_income.show()


# In[46]:


#calculate the IQR and remove outliers coapplicant


# In[47]:


q1=df['CoapplicantIncome'].quantile(0.25)
q3=df['CoapplicantIncome'].quantile(0.75)
IQR=q3-q1


# In[48]:


IQR


# In[49]:


#define lower and upper bond


# In[50]:


lower_bond=q1-1.5*IQR
upper_bond=q3+1.5*IQR


# In[51]:


lower_bond


# In[52]:


upper_bond


# In[53]:


#remove outliear


# In[54]:


df=df[(df['CoapplicantIncome']>=lower_bond)&(df['CoapplicantIncome']<=upper_bond)]


# In[55]:


fig_coapplicant_income=px.box(df,x='Loan_Status',
                             y='CoapplicantIncome',
                             color='Loan_Status',
                             title='loan status vs coapplicant income')
fig_coapplicant_income.show()


# In[56]:


#loan amount and loan status relation


# In[57]:


fig_loan_amount=px.box(df,x='Loan_Status',
                      y='LoanAmount',
                      color='Loan_Status',
                      title='loan status  vs  loan amount')
fig_loan_amount.show()


# In[58]:


#credict history and loan_status


# In[59]:


fig_credict_history=px.histogram(df,x='Credit_History',color='Loan_Status',
                                barmode='group',
                                title='Loan_Status vs credict_his')
fig_credict_history.show()


# In[60]:


#'group' mode (barmode='group'): Bars for different sets of values are grouped together for each category, side by side.

#'stack' mode (barmode='stack'): Bars for different sets of values are stacked on top of each other for each category, showing the cumulative total.


# In[61]:


fig_property_area=px.histogram(df,x='Property_Area',color='Loan_Status',barmode='group',title='Loan status vs property area')
fig_property_area.show()


# # Data preparation and training loan approval prediction model

# In[62]:


#convert categorical columns to numerical colum using one-hot-encoding


# In[63]:


cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']


# In[64]:


cat_cols


# In[65]:


df.info()


# In[66]:


# Applying one-hot encoding using pd.get_dummies


# In[67]:


df = pd.get_dummies(df, columns=cat_cols)


# In[68]:


df


# In[69]:


# split the dataset into features (x) and target(y)


# In[70]:


x=df.drop("Loan_Status",axis=1)


# In[71]:


x


# In[72]:


y=df['Loan_Status']
y


# In[73]:


#split the data set into traing and testing data


# In[74]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[75]:


# Scale the numerical columns using StandardScaler
scaler = StandardScaler()
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_test[numerical_cols] = scaler.transform(x_test[numerical_cols])


# In[76]:


from sklearn.svm import SVC
model=SVC(random_state=42)
model.fit(x_train,y_train)


# In[77]:


y_pred=model.predict(x_test)
print(y_pred)


# In[78]:


from sklearn.metrics import accuracy_score, classification_report


# In[79]:


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)


# In[80]:


# Convert X_test to a DataFrame
x_test_df = pd.DataFrame(x_test, columns=x_test.columns)

# Add the predicted values to X_test_df
x_test_df['Loan_Status_Predicted'] = y_pred
print(x_test_df.head())


# # summary

# Loan approval prediction involves the analysis of various factors, such as the applicant’s financial history, income, credit rating, employment status, and other relevant attributes. By leveraging historical loan data and applying machine learning algorithms, businesses can build models to determine loan approvals for new applicants. 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




