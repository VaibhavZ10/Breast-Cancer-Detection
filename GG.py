#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Detection

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing Dataset

# In[2]:


df = pd.read_csv("data.csv")
df.head()


# In[3]:


df1 = df.copy()


# In[4]:


df.columns[1:]


# In[5]:


df.drop(columns="Unnamed: 32",inplace = True)


# In[6]:


df.columns[1:]


# ### Visual Analysis

# In[7]:


from IPython.display import Image
Image("Visual.png")


# 
# ### As obvious from the image above, malignant cells are larger in size, thus have a bigger radius, perimeter and area.  Due to an arbitrary shape, the malignant cells are also expected to be more concave, with a large number of concave, finger-like projections.

# ## Encoding Categorical Values

# In[8]:


#Using Label Encoders to replace values in a single column
from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()


# In[9]:


df["diagnosis"] = labelEncoder.fit_transform(df["diagnosis"])
df["diagnosis"].head()


# ## Plotting the initial data

# In[10]:


diagnosis_count = df["diagnosis"].value_counts()
diagnosis_count


# In[11]:


# Pie chart representation
sns.set_theme(style="whitegrid")
pie_labels = ['Benign', 'Malignant']
pie_explode = [0, 0.1]
plt.figure(figsize=(10, 8))
#Number of benign and malignant cases
plt.pie(diagnosis_count, labels=pie_labels, shadow=True,  autopct='%1.1f%%', explode=pie_explode, textprops={'fontsize': 14})
plt.legend()
plt.title("Percent of Cases in the Data")
plt.show()


# In[12]:


sns.set_theme(style="whitegrid")

x = df['diagnosis']

y = ['radius_mean', 'texture_mean', 'perimeter_mean',
     'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
     'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

fig, axes = plt.subplots(len(y)//2, 2, figsize=(20, 35))
axes_flat = axes.flatten()

index = 0
for para in y:
    axis = axes_flat[index]
    axis.xaxis.label.set_size(20)
    axis.yaxis.label.set_size(20)
    axis.tick_params(labelsize=18)
    
    sns.boxplot(x=x, y=df[para], data=df, ax=axis)
    index += 1
    
plt.show()


# The classification of cancer depending on variation in each individual parameter has been shown with the help of boxplots. The following inferrences can be drawn by observing these graphs:-
# 
# - For each parameter (eg. radius_mean), more separated the two boxes are, the more significant role the parameter would play in deciding whether the cancer is benign or malignant. This is because, more the separation, more clear would be the signs of an abnormal behaviour by the cells.
# - This implies that the "Fractal dimension" of the cell will have little to no impact in determining the outcome.
# - Similarly, "Symmetry" of the cell is also not that influential for the result.
# - The gap between the boxes (and in turn distribution of data) in "Smoothness" of the cell is not that significant. The upper limit of smoothness in benign cases almost overlaps with the median of the malignant ones. Thus this property of the cell should not be given a lot of weight (but cannot be neglected) in the detection, since there is a probability that it can classify average and below average smoothness cases as benign instead of malignant.
# - Almost all the other properties of the cells show a clearer distinction in benign and malignant cases, indicating that they will probably have a stronger say in determination of the result.
# 
# Important: It must be noted that the above observations have be made by considering only the middle 50 percentile (i.e the box part) as it is just an human observation rather than a calculated judgement (Which would be too complicated for an initial observation and cannot be done with graphical observation).

# In[13]:


y = df["diagnosis"]
df.drop(["diagnosis"],axis = 1 , inplace = True)


# In[14]:


y.head(3)


# In[30]:


df.head()


# ## More Preprocessing and Scaling

# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)

print(X_train.shape, X_test.shape)


# In[16]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[17]:


cols = df.columns

scaled_data = pd.DataFrame(X_train,columns = cols)

scaled_data.head()


# ### Finding the best model

# In[18]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[19]:


#create param
model_param = {
    'DecisionTreeClassifier':{
        'model':DecisionTreeClassifier(),
        'param':{
            'criterion': ['gini','entropy']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'param' : {
            'n_estimators': [1,5,10]
        }
    },
        'KNeighborsClassifier':{
        'model':KNeighborsClassifier(),
        'param':{
            'n_neighbors': [5,10,15,20,25]
        }
    },
        'SVC':{
        'model':SVC(),
        'param':{
            'kernel':['rbf','linear','sigmoid'],
            'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
         
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'param': {}
    },
      "Logistic Regression":{
          "model" : LogisticRegression(random_state = 0, penalty='l1', solver='liblinear'),
          "param" : {
              "C" : [0.001,0.01,0.1,1,10,100,1000],
         'penalty': ['l1','l2']
          }
      }
}


# In[20]:


scores =[]
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(estimator=mp['model'],param_grid=mp['param'],cv=5,return_train_score=False)
    model_selection.fit(X_train,y_train)
    scores.append({
        'model': model_name,
        'best_score': model_selection.best_score_,
        'best_params': model_selection.best_params_
    })


# In[21]:


df_model_score = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df_model_score


# The above results tells us that SVC model is the most accurate for prediction of Breast Cancer w.r.t the dataset collected by Dr. William H. Wolberg, physician at the University Of Wisconsin Hospital at Madison, Wisconsin, USA.

# In[22]:


model_svc = SVC( C = 0.1 , kernel = "linear" , gamma = 1)
model_svc.fit(X_train,y_train)


# In[23]:


Y_pred = model_svc.predict(X_test)


# In[24]:


y_test.value_counts()


# ### Confusion Matrix

# In[25]:


Image("param.jpg")


# ### Thus we can define the two parameters in detection as:
# ### Sensitivity: The ability of a test to correctly identify people with a disease.
# ### Specificity: The ability of a test to correctly identify people without the disease.

# In[26]:


class_names = ["B","M"]
class_names


# In[27]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[28]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, Y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='Non-Normalized confusion matrix')
plt.grid(False)
plt.show()
#Print Important Medical terms
Specificity = (cnf_matrix[0][0]) / (cnf_matrix[0][0] + cnf_matrix[0][1])
print("\nSpecificity is: {0:.2f}%".format(Specificity*100))

Sensitivity = (cnf_matrix[1][1]) / (cnf_matrix[1][1] + cnf_matrix[1][0])
print("\nSensitivity is: {0:.2f}%".format(Sensitivity*100))


# # Conclusion

# The final model has a commendable Specificity of 98.15%, implying that the model can correctly identify 98% of the people without Breast Cancer.
# The model also has a Sensitivity of 95.24%, implying that the model can correctly identify approximately 95% of the people with Breast Cancer.
