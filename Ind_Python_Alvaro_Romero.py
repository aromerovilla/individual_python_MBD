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

season_col = ["#749ee8", '#ffc9b9','#e69479','#b2d7ff']
month_col = ["#749ee8","#749ee8","#749ee8", '#ffc9b9','#ffc9b9','#ffc9b9','#e69479','#e69479','#e69479','#b2d7ff','#b2d7ff','#b2d7ff']

y0 = daily['casual']
y1 = daily['registered']
y2 = daily['cnt']

trace0 = go.Box(
    y=y0,
    name = 'Casual',
    marker = dict(
        color = 'rgb(255, 107, 161)',
    ),
    boxmean=True
)
trace1 = go.Box(
    y=y1,
    name = 'Registered',
    marker = dict(
        color = 'rgb(107, 208, 255)',
    ),
    boxmean=True
)
trace2 = go.Box(
    y=y2,
    name = 'Total',
    marker = dict(
        color = 'rgb(107, 255, 216)',
    ),
    boxmean=True
)
chart = [trace0, trace1, trace2]
py.iplot(chart)

#Correlation Matrix

#Variables for correlation matrix
df=daily[['season',
                  'mnth',
                  'hr',
                  'holiday',
                  'weekday',
                  'workingday',
                  'weathersit',
                  'temp',
                  'atemp',
                  'hum',
                  'windspeed',
                  'cnt']]

corrs = df.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)

figure.iplot()



#Relationship of weather variables
#scatterplot for weather data
sns.set()
cols = ['weathersit','temp','atemp','hum','windspeed']
sns.pairplot(daily[cols], size = 2.5)
plt.show();


#Bike rentals per month

fig, ax = plt.subplots()
monthly_plot = sns.barplot(data = daily[['mnth',
                                        'cnt']],
                           x = 'mnth',
                           y = 'cnt',
                           ci = None,
                           palette=month_col,
                           estimator = sum)
ax.set(xlabel = 'Month of the year', ylabel = 'Total count of bikes', title = 'Total count of bikes per month')
ax.set_xticklabels(['Jan','Feb','Mar','Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

#Bike rentals among seasons
# Number of bikes rented per season
fig, ax = plt.subplots()
plot = sns.barplot('yr',
                   'cnt',
                   hue = 'season',
                   data = daily,
                   ci = None,
                   palette=season_col,
                   estimator = sum)
ax.set(xlabel = 'Year', ylabel = 'Bikes rented',title = 'Bikes rented per year and per season')
ax.set_xticklabels(['2011', '2012'])
leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, ["Winter", "Spring", "Summer", "Autumn"])
plt.show()

# Total bike rentals per hour - Season breakdown
g = sns.pointplot(data = daily[['hr', 'cnt','season']],
                  x = 'hr',
                  y = 'cnt',
                  hue = 'season',
                  legend_out = True,
                  estimator = sum)
leg_handles = g.get_legend_handles_labels()[0]
g.legend(leg_handles, ["Winter", "Spring", "Summer", "Autumn"])
g.set(xlabel = 'Hour of the day', ylabel = 'Total count of bikes', title = 'Total count of bikes per hour of the day for each season')
plt.show()

# Casual users during working and non-working days
fig, ax = plt.subplots()
wor_week_plot = sns.pointplot(data = daily[['hr',
                                           'casual',
                                           'workingday']],
                              x = 'hr',
                              y = 'casual',
                              hue = 'workingday',
                              estimator = np.average)
leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, ["Non-working day", "Working day"])
ax.set(xlabel = 'Hour of the day', ylabel = 'Average count of bikes',title = 'Average count of bikes per hour for working days vs weekend for casual users')
plt.show()

#  Effect of temperature on bike rentals
fig, ax = plt.subplots()
sns.regplot(x = daily["temp"],
            y = daily["cnt"],
            fit_reg = True,
            ci = 100,
            truncate = True,
            color = 'DarkRed')
ax.set(xlabel = 'Temperature', ylabel = 'Total count of bikes',title = 'Number of bikes year rented in function of the temperature')
plt.show()


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
