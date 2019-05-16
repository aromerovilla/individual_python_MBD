#Libraries
from dask_ml.preprocessing import DummyEncoder, Categorizer
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask import dataframe as dd
from dask_glm.datasets import make_classification
from dask_ml.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import ElasticNet
from dask_ml.wrappers import ParallelPostFit

#Data Loading
def read_data(input_path):
    raw_data = dd.read_csv(input_path)
    return raw_data
    
raw = read_data('https://gist.githubusercontent.com/aromerovilla/7170e4ff45dd943af6a920d2f510cd0f/raw/b5732e3122ca9d0dec95717b62434e2b68e642b6/hour.csv')
raw.info()

#Data Exploration

daily = raw.compute()

# Configuring plotting visual and sizes
sns.set_style('white')
sns.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)

#Daily bike rentals

trace = go.Scatter(x=list(daily.dteday),
                   y=list(daily.cnt))

chart_1 = [trace]
layout = dict(
    title='Daily bike rentals over time',
    xaxis=dict(
        rangeselector=dict(
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

fig = dict(data=chart_1, layout=layout)
py.iplot(fig)

daily_cum = daily.copy()
daily_cum['cumu'] = daily_cum.cnt.cumsum()
daily_cum['cumu_c'] = daily_cum.casual.cumsum()
daily_cum['cumu_r'] = daily_cum.registered.cumsum()

off.init_notebook_mode(connected=False)

chart5 = [dict(
  type = 'scatter',
  x = daily_cum.dteday,
  y = daily_cum.cumu,
  transforms = [dict(
    type = 'aggregate',
    groups = daily_cum.dteday,
    aggregations = [dict(
        target = 'y', func = 'sum', enabled = True),
    ]
  )]
)]



off.iplot({'data': chart5}, validate=False)


#Data preparation

selected = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', ]
categorizer = Categorizer(columns=selected)
categorizer = categorizer.fit(raw)
result = categorizer.transform(raw)
print(result)

#Baseline

data = raw.copy()
data.head()

#Splitting data

X = data.loc[:, data.columns != 'cnt']
y = data.loc[:, ['instant','cnt']]

X_train = X.loc[(X.instant <= 15211)].drop(['dteday', 'instant', 'registered','casual'], axis=1)
y_train = y.loc[(y.instant <= 15211)].drop('instant', axis=1)
X_test = X.loc[(X.instant > 15211)].drop(['dteday', 'instant', 'registered','casual'], axis=1)
y_test = y.loc[(y.instant > 15211)].drop('instant', axis=1)

X = data.loc[:, data.columns != 'cnt']
y = data.loc[:, ['instant','cnt']]

X, y = make_classification(n_samples=10000, n_features=2)

X = dd.from_dask_array(X, columns=['instance','cnt'])
y = dd.from_array(y)

#Logistic Regression

lr = LogisticRegression()
lr.fit(X.values, y.values)

el = ParallelPostFit(estimator=ElasticNet())
el.fit(X_train, y_train)
preds = el.predict(X_test)

#Linear Regression

lr = LinearRegression()
lr.fit(X.values, y.values)

el = ParallelPostFit(estimator=ElasticNet())
el.fit(X_train, y_train)
preds = el.predict(X_test)
