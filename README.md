# **Stat 426 Project**  
## ML Analysis of Pew Research Data  
### Eirik Scoville

## **Introduction**  
  
For this project, I am using data from the Pew Research Center 2014 U.S. Religious Landscape Study to try and build a classification model that can predict a person's religious affiliation based on their views of several social, political, and theological issues.  
  
The data to be used consists of two separate datasets--one for data from the continental US, and the other containing survey answers from Alaska and Hawaii. The first dataset contains 35,556 observations of 135 variables (including survey responses), and the second contains 401 observations of 135, making the final combined dataset size 35,957x135.


```python
# Initialize packages
import numpy as np
import pandas as pd
```


```python
# Read in data
con = pd.read_spss('Religious Landscape Survey Data - Continental US.sav')
aah = pd.read_spss('Religious Landscape Survey Data - Alaska and Hawaii.sav')

# Combine into one dataset
df = pd.concat([con, aah], ignore_index=True)
```

## **EDA and Feature Analysis**

There are currently too many features to see in either an info call or by printing the head of the data, so as a first step, we must manually look at the features and determine which of them if any may be dropped as superfluous to our analysis. First, we will look at features that were not part of the survey questions themselves. These features are  
- weight,  
- psraid (a unique id number),  
- int_date (date of the interview),  
- lang (language of the interview),  
- type (type of sample used),  
- cregion (census region),  
- state,  
- usr (community type),  
- usr1 (redundant),  
- form,  
- density3 (population density quintiles),  
- marital (marital status),  
- hisp (is the respondent hispanic),  
- race,  
- chr (is the respondent Christian),  
- denom (specific religious affiliation),  
- family (religious affiliation group),  
- reltrad (religious affiliation tradition),  
- protfam (subsets of Protestantism),  
- children (does the respondant have any children under 18),  
- sex,  
- age,  
- educ (education level),  
- income,  
- regist (is the respondent registered to vote),  
- regicert (is the respondent certain that they are registered),  
- party (Republican, Democrat, or Independent),  
- partyln (does the respondent lean more Rep. or Dem.),  
- ideo (conservative or liberal ideology),  
- pvote04a (did the respondent vote in 2004),  
- pvote04b (did the respondent vote for Bush, Kerry, or someone else).

Right off the bat, there are a few of these variables that I will drop. Weight is only used for Pew's own analysis, and won't be helpful to me. The unique id assigned will not translate into any useful information ML analysis, so it will be dropped as well. The interview date, form, children, registration, '04 voting history, and type features are unlikely to be related to a person's religion. The language feature has only two values (English and a handful of Spanish), so I've chosen to drop it because the Spanish responses only account for about 3% of the total. Finally, there are several variables which represent different levels of granularity about the respondent's religious affiliation. One of these will be the label that we are trying to predict, but we must drop the others, as they will be too highly correlated with the outcome. After deliberation, 'reltrad' will be the value we will use as the prediction label, which has 16 possible values. This will likely need to be trimmed down for a good prediction.


```python
# List of variables to drop
drop = ['denom','protfam','family','weight','psraid','int_date','lang','type','usr','form','marital','children','sex','age','regist','regicert','pvote04a','pvote04b','chr']
df = df.drop(drop, axis=1)
# New size of dataset
df.shape
```




    (35957, 116)



Next, there are a few of the actual survey questions that will need to be dropped: the ones that actually ask about the respondents' religious affiliation. Additionally, there are a few questions which gather information that we've determined to be unecessary for this analysis. There are also some features in which the majority of the observations are missing. These are removed below:


```python
# List of questions to drop
drop2 = ['q17','q17a','q17b','q17c','q17d','q17e','q17f','q17g','q17h','q17i','q17j','q17k','q17l','q17m','q17n','q17o','q17p','q17q','q17r','q17s','q17t','q17u','q17v','q50','q50a','q50b','q51','q52','q53']
df = df.drop(drop2, axis=1)
# New size of dataset
df.shape
```




    (35957, 87)



Some questions are directed more at one group of people than another, and those tend to have lots of missing values. Rather than remove those, it will suffice to impute a placeholder string in any question where a response was not recorded.


```python
# Fill all NaNs with placeholder string
df = df.astype(str)
df = df.replace(np.nan, 'No entry')
```

The majority of this dataset consists of survey answers, which are actual text fields, and therefore cannot really be visualized in a graphical way. It's also hard to display a header with so many variables. Instead, here is a glimpse of the text of just some of the variables.


```python
df[['q18','q27','q34','q36','q41']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>q18</th>
      <th>q27</th>
      <th>q34</th>
      <th>q36</th>
      <th>q41</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Yes</td>
      <td>Between 100 and 500</td>
      <td>Absolutely certain</td>
      <td>Yes</td>
      <td>A few times a week</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>nan</td>
      <td>Fairly certain</td>
      <td>Yes</td>
      <td>Seldom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>nan</td>
      <td>Fairly certain</td>
      <td>Yes</td>
      <td>Seldom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nan</td>
      <td>Less than 100</td>
      <td>Absolutely certain</td>
      <td>No</td>
      <td>A few times a week</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>No</td>
      <td>Seldom</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>No</td>
      <td>Never</td>
    </tr>
    <tr>
      <th>6</th>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>No</td>
      <td>Never</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Don't Know/Refused (VOL.)</td>
      <td>nan</td>
      <td>Not at all certain</td>
      <td>No</td>
      <td>Never</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Yes</td>
      <td>Less than 100</td>
      <td>Absolutely certain</td>
      <td>Yes</td>
      <td>Several times a day</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nan</td>
      <td>Between 500 and 2,000</td>
      <td>nan</td>
      <td>No</td>
      <td>A few times a month</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Engineering and Preparation

In order to feed the data to a ML algorithm, the values in each observation must be numeric rather than text, as they are now. The following code attempts to systematically encode each of the 87 remaining features in groups based on the type of data.


```python
from sklearn.preprocessing import LabelEncoder

# To save memory, many ordinal questions will be encoded using label encoding instead of one hot encoding.
# First, we must specify which questions belong in this group
labencode = df.drop('reltrad', axis=1).columns

# Instantiate the encoder
le = LabelEncoder()
# Perform the transformation
df[labencode] = df[labencode].apply(le.fit_transform)
```

Next, we will split the data into a training set on which to train the models, and a test set on which to validate the performance of each model.


```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=.3, stratify=df.reltrad, random_state=777)
```


```python
X_train = train.drop('reltrad', axis=1)
X_test = test.drop('reltrad', axis=1)

y_train = train['reltrad']
y_test = test['reltrad']
```

Since every question has a different number of possible answers, and to facilitate the following computations, we will take the numeric values in the dataframe and scale them all to be between 0 and 1.


```python
from sklearn.preprocessing import MinMaxScaler

# Instantiate scaler
minmax = MinMaxScaler()
# Fit scaler to training data
mmtrans = minmax.fit(X_train)
# Transform both training and test sets
X_train = minmax.transform(X_train)
X_test = minmax.transform(X_test)
```

## Machine Learning Models

In order to determine whether an effective model can be built to classify the religion of survey respondents, three different ML algorithms will be tried: A K-Nearest Neighbors classifier, a Random Forest classifier, and a Gradient Boost classifier. First, we have the KNN classifier:


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Build a KNN model
knn = KNeighborsClassifier()
parameters = {
    'n_neighbors': [11,21,31,41,51,101]
}
knncv = GridSearchCV(knn, parameters, cv=5)
knncv.fit(X_train, y_train)
```




    GridSearchCV(cv=5, estimator=KNeighborsClassifier(),
                 param_grid={'n_neighbors': [11, 21, 31, 41, 51, 101]})




```python
# See which k had the best results
knncv.best_params_
```




    {'n_neighbors': 41}




```python
knn = KNeighborsClassifier(n_neighbors = 41)
knn.fit(X_train, y_train)
yhatknn = (knn.predict(X_test))
print("Accuracy: " + str(round(accuracy_score(y_test, yhatknn),4)))
```

    Accuracy: 0.5943
    

Clearly, the KNN classifier did not perform spectacularly well. Next, we will try a Random Forest algorithm.


```python
from sklearn.ensemble import RandomForestClassifier

# Build a Random Forest model
rf = RandomForestClassifier(n_jobs=-1)
parameters = {
    'n_estimators': [100, 250, 500],
    'max_depth': [5, 10, 50, 100, None]
}

rfcv = GridSearchCV(rf, parameters, cv=5)
rfcv.fit(X_train, y_train)
```




    GridSearchCV(cv=5, estimator=RandomForestClassifier(n_jobs=-1),
                 param_grid={'max_depth': [5, 10, 50, 100, None],
                             'n_estimators': [100, 250, 500]})




```python
# Print results of the tuning
def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

print_results(rfcv)
```

    BEST PARAMS: {'max_depth': None, 'n_estimators': 500}
    
    0.78 (+/-0.002) for {'max_depth': 5, 'n_estimators': 100}
    0.78 (+/-0.002) for {'max_depth': 5, 'n_estimators': 250}
    0.779 (+/-0.003) for {'max_depth': 5, 'n_estimators': 500}
    0.837 (+/-0.008) for {'max_depth': 10, 'n_estimators': 100}
    0.838 (+/-0.007) for {'max_depth': 10, 'n_estimators': 250}
    0.838 (+/-0.009) for {'max_depth': 10, 'n_estimators': 500}
    0.851 (+/-0.003) for {'max_depth': 50, 'n_estimators': 100}
    0.852 (+/-0.007) for {'max_depth': 50, 'n_estimators': 250}
    0.851 (+/-0.007) for {'max_depth': 50, 'n_estimators': 500}
    0.849 (+/-0.006) for {'max_depth': 100, 'n_estimators': 100}
    0.851 (+/-0.005) for {'max_depth': 100, 'n_estimators': 250}
    0.851 (+/-0.007) for {'max_depth': 100, 'n_estimators': 500}
    0.848 (+/-0.008) for {'max_depth': None, 'n_estimators': 100}
    0.851 (+/-0.007) for {'max_depth': None, 'n_estimators': 250}
    0.852 (+/-0.006) for {'max_depth': None, 'n_estimators': 500}
    


```python
rf = RandomForestClassifier(max_depth=None, n_estimators=500, n_jobs=-1)
rf.fit(X_train, y_train)
yhatrf = (rf.predict(X_test))
```


```python
print("Accuracy: " + str(round(accuracy_score(y_test, yhatrf),4)))
```

    Accuracy: 0.8554
    

The Random Forest model performed much better than the KNN model did. As a final check, we'll perform a Gradient Boost model as well using the Adaboost package, and see if it can improve on the score of the Random Forest.


```python
from sklearn.ensemble import AdaBoostClassifier

# Create an Adaptive Boosting model
ada = AdaBoostClassifier()
parameters = {
    'n_estimators': [100, 250, 500],
    'learning_rate': [.1, .01, .001]
}
gbcv = GridSearchCV(ada, parameters, cv=5)
gbcv.fit(X_train, y_train)
```




    GridSearchCV(cv=5, estimator=AdaBoostClassifier(),
                 param_grid={'learning_rate': [0.1, 0.01, 0.001],
                             'n_estimators': [100, 250, 500]})




```python
gbcv.best_params_
```




    {'learning_rate': 0.001, 'n_estimators': 250}




```python
ada = AdaBoostClassifier(learning_rate=0.001, n_estimators=250)
ada.fit(X_train, y_train)
yhatgb = ada.predict(X_test)
```


```python
print("Accuracy: " + str(round(accuracy_score(y_test, yhatgb),4)))
```

    Accuracy: 0.661
    

It appears that the Adaptive Boosting algorithm did better than the KNN model, but not as well as the Random Forest model.

## Conclusions

The model that performed the poorest for this analysis was the K Nearest Neighbor algorithm, consistently scoring somewhere around 59% accuracy. The next best performing model was the Adaptive Boosting algorithm, scoring around 66% accuracy. And finally, the Random Forest algorithm knocked it out of the park with an impressive 85% accuracy. This outperformed my own expectations, because there were quite a few categories in the response label, and usually the more categories you have, the harder it is to get really good accuracy. I was expecting to have to combine some of the response categories into macro categories, but with an accuracy of 85%, it almost feels unnecessary to do so. A full list of the categories being predicted follows:


```python
df['reltrad'].value_counts()
```




     Evangelical Protestant Churches                                 9569
     Catholic                                                        8126
     Mainline Protestant Churches                                    7550
     Unaffiliated                                                    5141
     Historically Black Protestant Churches                          2000
     Jewish                                                           683
     Mormon                                                           599
     Other Faiths                                                     460
     Buddhist                                                         423
     Orthodox                                                         367
     Don’t know/refused (no information on religious affiliation)     274
     Hindu                                                            258
     Jehovah's Witness                                                218
     Other Christian                                                  130
     Muslim                                                           117
     Other World Religions                                             42
    Name: reltrad, dtype: int64



The fact that there were so many of certain groups and so few of some other groups may have skewed the prediction algorithm somewhat, introducing bias and lowering the overall accuracy scores. This also could be addressed by combining some of the smaller religious groups into a larger macro-category, and would be a good jumping-off point for further analysis in the future.

In conclusion, with the overwhelming amount of data available, good results were able to be acheived with minimal feature engineering. Considering the number of features excluded from the analysis, and the use of label encoding rather than one hot encoding, there is good reason to believe that including more of that data would result in even better results. Unfortunately, I could not explore those options because of hardware and memory constraints. But even so, I think it's safe to conclude that the Pew Research survey answers do give a very good indication of which religion a person belongs to, based on demographic, social, and political issues. It serves as a testament to how diverse smaller branches of the same religion. And yet, the members of those groups tend to share enough in common to identify themselves as part of a certain group.
