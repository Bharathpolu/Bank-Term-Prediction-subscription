# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
#import XGboost
import graphviz
warnings.filterwarnings('ignore')


# In[2]:


import sklearn


# In[3]:


from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier ,GradientBoostingClassifier
#from XGboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import roc_auc_score ,mean_squared_error,accuracy_score,classification_report,roc_curve,confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from scipy.stats.mstats import winsorize
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns',None)
import six
import sys
sys.modules['sklearn.externals.six'] = six


# In[4]:


banking = pd.read_csv("C:/Users/pbhar/OneDrive - University of Cincinnati/2024 resume/2nd sem/special topics in BA/project/Bank data.csv")
banking.shape


# In[5]:


banking.head()


# In[6]:


banking['job'].unique()


# In[7]:


banking['marital'].unique()


# In[ ]:





# In[8]:


banking.isnull().sum()


# In[11]:


import matplotlib.pyplot as plt 

import seaborn as sns 

 

# Plotting the distribution of age 

plt.figure(figsize=(5, 5)) 

sns.histplot(banking['age'], bins=50, kde=True, color='skyblue') 

plt.title('Distribution of Age') 

plt.xlabel('Age') 
#plt.ylabel('Frequency') 

plt.show() 


# In[10]:


plt.figure(figsize=(3, 3)) 

sns.boxplot(x='y', y='age', data=banking) 

plt.title('Age vs Subscription') 

plt.xlabel('Subscription Status') 

plt.ylabel('Age') 

plt.show() 


# In[11]:


plt.figure(figsize=(8, 8)) 

sns.heatmap(banking.corr(), annot=True, cmap='coolwarm', fmt=".2f") 

plt.title('Correlation Heatmap of Banking Features') 

plt.show() 


# In[17]:


from sklearn.ensemble import RandomForestClassifier 

 

model = RandomForestClassifier(random_state=42) 

model.fit(X_train, Y_train) 

 

# Plotting feature importance 

importances = model.feature_importances_ 

indices = np.argsort(importances)[::-1] 

plt.figure() 

plt.title('Feature Importance') 

plt.bar(range(X_train.shape[1]), importances[indices], color="b", align="center") 

plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90) 

plt.xlim([-1, X_train.shape[1]]) 

plt.show() 


# In[16]:


from sklearn.model_selection import train_test_split

# Predictors
X = banking.iloc[:,:-1]

# Target
Y = banking.iloc[:,-1]

# Dividing the data into train and test subsets
X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.2,random_state=41)



# In[19]:


# run Logistic Regression model

model_1 = LogisticRegression(max_iter=1000)
# fitting the model
model_1.fit(X_train, Y_train)
# predicting the values
Y_scores = model_1.predict(X_val)


# In[20]:


# Evaluate the accuracy of the model
accuracy = accuracy_score(Y_val, Y_scores)
print("Accuracy:", accuracy)


# In[21]:


print("Classification Report:")
print(classification_report(Y_val, Y_scores))


# In[24]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_val, Y_scores)
print('ROC_AUC_SCORE is',roc_auc_score(Y_val, Y_scores))
    
#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])
    
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# In[25]:


import pickle


# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Assume X, y are your data and labels
X, y = make_classification(n_samples=1000, n_features=12, n_informative=5, n_redundant=5, random_state=42)
clf = RandomForestClassifier()
clf.fit(X, y)
importance = clf.feature_importances_

# Print the feature importance
for i,v in enumerate(importance):
    with open("filename.prediction", "wb") as f:
        pickle.dump(clf, f)     
        print('Feature: %0d, Score: %.5f' % (i,v))


# In[27]:


with open("filename.prediction", "wb") as f:
    pickle.dump(clf, f) 
with open("filename.prediction", "rb") as f:
    clf  = pickle.load(f)


# In[13]:


import gradio as gr


# In[14]:


def make_prediction(age, job,marital,education,default,housing,loan,contact,month,duration,campaign,poutcom):
    with open("filename.prediction", "rb") as f:
        clf  = pickle.load(f)
        preds = clf.predict([[age, job,marital,education,default,housing,loan,contact,month,duration,campaign,poutcom]])
    if preds == 1:
            return "Prefers the term deposit"
    return "Doesn't prefers the term deposit"

#Create the input component for Gradio since we are expecting 4 inputs

age = gr.Number(label = "Enter the Age of the Individual")
job = gr.Number(label=("Enter Employment Status {0:Admin, 1:Blue collar, 3:entrepreneur, "
                       "4:housemaid, 5:management, 6:self-employed, 7:services, "
                       "8:Student, 9:Technician, 10:unemployed, 11:unknown}"))
marital = gr.Number(label = "Enter marital status{1:married,2:divorce or widowed,3:unkown}")
education = gr.Number(label = "Enter your education{0:basic.4y,1:basic.6y,2:basic.9y,"
                      "3:Highschool,4:illiterate,5:professional course,6:university degree,7:unknown}")
default=gr.Number(label="Enter default {0:No,1:Yes}")
housing=gr.Number(label="Enter Housing {0:No,1:Yes}")
loan=gr.Number(label="Enter loan {0:No,1:Yes}")
contact=gr.Number(label="Enter contact {1:Cellular,2:Telephone}")
month=gr.Number(label="Enter month number")
#day_of_week=gr.Text(label="Enter the week")
duration=gr.Number(label="Enter time ")
campaign=gr.Number(label="Enter campaign number")
poutcome=gr.Number(label="Enter outcome{0:failure,1:nonexistence,2:success}")
# We create the output
output = gr.Textbox()


app = gr.Interface(fn = make_prediction, inputs=[age, job,marital,education,default,housing,
                                                 loan,contact,month,duration,campaign,poutcome], outputs=output)
app.launch(share=True)


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate a random binary classification problem.
X, y = make_classification(n_samples=1000, n_features=12, n_informative=5, n_redundant=5, random_state=42)

# Create a random forest classifier.
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the data.
clf.fit(X, y)

# Get the feature importances.
importances = clf.feature_importances_

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [f'Feature {i}' for i in indices]

# Barplot: Add bars
plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), names, rotation=20, ha='right')

# Show plot
plt.show()


# In[52]:


# Run Decision Tree Classifier
model_2 = DecisionTreeClassifier()

model_2.fit(x_train, y_train)
y_scores = model_2.predict(x_val)
auc = roc_auc_score(y_val, y_scores)
print('Classification Report:')
print(classification_report(y_val,y_scores))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_scores)
print('ROC_AUC_SCORE is',roc_auc_score(y_val, y_scores))
    
#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])
    
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# In[53]:


from sklearn import tree
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image
from subprocess import call
import matplotlib.image as mpimg


# In[54]:


from sklearn import tree
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image
from subprocess import call
import matplotlib.image as mpimg


# In[55]:


@interact
def plot_tree(crit=["gini", "entropy"],
              split=["best", "random"],
              depth=IntSlider(min=1,max=5,value=2, continuous_update=False),
              min_split=IntSlider(min=2,max=5,value=2, continuous_update=False),
              min_leaf=IntSlider(min=1,max=5,value=1, continuous_update=False)):
    
    estimator = DecisionTreeClassifier(random_state=0,
                                       criterion=crit,
                                       splitter = split,
                                       max_depth = depth,
                                       min_samples_split=min_split,
                                       min_samples_leaf=min_leaf)
    estimator.fit(x_train, y_train)
    print('Decision Tree Training Accuracy: {:.3f}'.format(accuracy_score(y_train, estimator.predict(x_train))))
    print('Decision Tree Test Accuracy: {:.3f}'.format(accuracy_score(y_val, estimator.predict(x_val))))

   # graph = Source(tree.export_graphviz(estimator,
                                        #out_file=None,
                                        #feature_names=x_train.columns,
                                        #class_names=['0', '1'],
                                       # filled = True))
    
   # display(Image(data=graph.pipe(format='png')))
#     graph = graphviz.Source(dot_data, format="png") 

    
    return estimator


# In[12]:


model = RandomForestClassifier()

model.fit(x_train, y_train)
y_scores = model.predict(x_val)
auc = roc_auc_score(y_val, y_scores)
print('Classification Report:')
print(classification_report(y_val,y_scores))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_scores)
print('ROC_AUC_SCORE is',roc_auc_score(y_val, y_scores))
    
#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])
    
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# In[ ]:




