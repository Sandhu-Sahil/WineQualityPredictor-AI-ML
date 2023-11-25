# %%
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as ps
import joblib
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('wine_dataset.csv')

# %%
df.head()

# %%
df.rename(columns={'type':'wine_type'},inplace=True)
df.head()

# %%
print(f'Data Frame Shape (rows, columns): {df.shape}')

# %%
df.describe()

# %%
len(df.quality.unique())

# %%
Sum = df.isnull().sum()
Percentage = ( df.isnull().sum()/df.isnull().count())

pd.concat([Sum,Percentage], axis =1, keys= ['Sum', 'Percentage'])

# %%
df.head()

# %%
def get_missing_values(df):
    values = {}
    for feat in df.columns:
        if df[feat].isna().any():
            values[feat] = round(df[feat].mean(),2)
    return values

# %%
values = get_missing_values(df)

# %%
values

# %%
df.fillna(value=values,inplace=True)

# %%
sns.pairplot(df, hue="quality")

# %%
plt.figure(figsize=(10,5))
ax = df.corr()['quality'].plot(kind='bar',title='correlation of target variable to features')
ax.set_ylabel('correlation')

# %%
plt.figure(figsize=(30,10));plt.subplot(121,title = 'Correletion plot')
sns.heatmap(df.corr(),annot=True)

# %%
df.head()

# %%
df.wine_type = df.wine_type.map({'white':0, 'red':1})

# %%
df.head()

# %%
sns.countplot(data=df, x="quality").set_title("Wine quality")

# %%
import tpot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

# %%
X = df.drop(columns=['quality'],axis=1) 
y = df.quality

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=40)

# %%
lr = LogisticRegression(random_state=40)
lr.fit(X_train, y_train)

# %%
train_accuracy = lr.score(X_train, y_train)
test_accuracy = lr.score(X_test, y_test)
print('Accuracy in Train set   : {:.2f}'.format(train_accuracy), 
      'Accuracy in Test  set   : {:.2f}'.format(test_accuracy), sep='\n')

# %%
df_bins3= df.copy()

# %%
bins = [0,4,7,10]

labels = [0,1,2] # 'low'=0,'average'=1, 'high'=2

df_bins3['quality_range']= pd.cut(x=df_bins3['quality'], bins=bins, labels=labels)

#df_bins3.type = df_bins3.type.map({'white':0, 'red':1})

print(df_bins3[['quality_range','quality']].head(5))

# %%
sns.countplot(data=df_bins3, x="quality_range").set_title("Wine quality")

# %%
df_bins3 = df_bins3.drop(columns=['quality'],axis=1).copy()

# %%
X = df_bins3.drop(columns=['quality_range'],axis=1) 
y = df_bins3.quality_range

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=40)

# %%
X.columns

# %%
lr    = LogisticRegression(random_state=40)
lr.fit(X_train, y_train)

# %%
train_accuracy = lr.score(X_train, y_train)
test_accuracy = lr.score(X_test, y_test)
print('One-vs-rest', '-'*35, 
      'Accuracy Score of Train Model : {:.2f}'.format(train_accuracy), 
      'Accuracy Score of Test  Model : {:.2f}'.format(test_accuracy), sep='\n')

# %%
AutoML = tpot.TPOTClassifier(
                            generations=5,
                            population_size=100,
                            offspring_size=None,
                            mutation_rate=0.9,
                            crossover_rate=0.1,
                            scoring=None,
                            cv=5,
                            subsample=1.0,
                            n_jobs=1,
                            max_time_mins=None,
                            max_eval_time_mins=5,
                            random_state=None,
                            config_dict=None,
                            template=None,
                            warm_start=False,
                            memory=None,
                            use_dask=False,
                            periodic_checkpoint_folder=None,
                            early_stop=None,
                            verbosity=2,
                            disable_update_check=False,
                            )

# %%
AutoML.fit(X_train,y_train)

# %%
AutoML = tpot.TPOTClassifier(
                            generations=5,
                            population_size=100,
                            offspring_size=None,
                            mutation_rate=0.9,
                            crossover_rate=0.1,
                            scoring=None,
                            cv=5,
                            subsample=1.0,
                            n_jobs=1,
                            max_time_mins=None,
                            max_eval_time_mins=5,
                            random_state=None,
                            config_dict=None,
                            template=None,
                            warm_start=False,
                            memory=None,
                            use_dask=False,
                            periodic_checkpoint_folder=None,
                            early_stop=None,
                            verbosity=2,
                            disable_update_check=False,
                            )

# %%
AutoML.fit(X,y)
#save model 
with open('model-v1.joblib', 'wb') as f:
    joblib.dump(lr,f)

# %%
from sklearn.metrics import confusion_matrix as cm

predictions = AutoML.predict(X_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)
plt.show()

# %%
AutoML.export('pipeline.py')


