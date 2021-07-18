#!/usr/bin/env python
# coding: utf-8

# Create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to 
# predict the right class accordingly.

# In[14]:


import sklearn.datasets as datasets
import pandas as pd


# # loading dataset

# In[15]:


# Loading the iris dataset
iris=datasets.load_iris()


# # Data analysis

# In[22]:


# Forming the iris dataframe
dataset = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[20]:


dataset.shape


# In[23]:


dataset.head()


# In[24]:


dataset.info()


# # preparing the dataset

# In[21]:


y=iris.target
y


# In[3]:


X.describe()


# # spliting and training dataset

# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # importing decision tree classifier and fiting model

# In[5]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)


print('Decision Tree Classifer Created Successfully')


# In[6]:


y_predict = dtree.predict(X_test)
y_predict


# # confusion matrix and classification report

# In[13]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))


# # visualizing decision tree

# In[7]:


import matplotlib.pyplot as plt
from sklearn import tree


a=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
b=['setosa','versicolor','virginica']

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)

tree.plot_tree(dtree, feature_names = a, class_names = b, filled = True);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




