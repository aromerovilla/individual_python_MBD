# individual_python_MBD

Link to my GitHub https://github.com/aromerovilla/individual_python_MBD

To run the code you need to pip install dask.ml

In this assigment I am rewritting into Dask the Bicycle machine learning assigment done with pandas.

For the ploting I used .compute() so I am able to plot graphs.

The first part of the code is a Data Exploration to understand the Data

3.1 Daily bike rentals

The first chart we chose, was a time series chart that displayes all the daily bike rentals over the course of the two years. We first plotted a chart based on the hourly data, however this shows even greater fluctuations and thus did not really help to detect patterns. With this line chart here we are not only able to filter based on a certain time frame but can hoover over and see the amount of bikes for each day and the corresponding day. In order to do this we created the chart as a plotly figure. In order to do so we first needed to create the chart type with the data/values for the x and y axis. This is stored in the variable trace that is then transformed to a list object and stored in the variable chart_1. The next step is to specify the layout of the chart. For this the variable layout is created that stores the title but also the specifications for the slides/hoover option. At the end the chart and the layout are stored in the dictionary variable figure. At the end this variable is then plotted as a plotly chart, which is interactive.

What we see in this charts is not only the seasonality effect during the year (more bikes in summer than winter) but also that the general level of daily bike rentals increased from one year to the other.

3.3 Boxplot for registered, casual and total users

In order to see where we have more outliers, we plotted the boxplots again for not only the total but also for registered and casual users. Eventhough we see that the general bike rental numbers for casual users are lower than for registered, both group show a similar level of outliers. Thus we do not think that the outliers in the dataset are caused by one group in particular.

3.4 Correlation matrix

In order to detect highly correlated features we created a boxplot that takes all our model variables into account. We did not include instant or date since those variables are not features within the model but rather indexes.

We see an overall low level of correlation. However, temperature and felt temparature are highly correlated. This might mean that one of the features is actually redundant for our model. Furthermore also month and season show a high correlation. This makes sense due to the fact that the seasons group month (partially) together. Thus months give a more granular view while seasons aggregate them. It might not make sense to include both variables in the model since they are quiet similar.

3.5 Relationship of weather variables

In order to detect specific relationships between the weahter variables we used the seaborn pairplots to not only plot the histograms of the variables but also the relationship between all pairs of weather variables. This might help to cluster groups of weather variables together.

Overall, it does not seem like there are very apparent clusters within the data set for weather. Only for temperature and felt temperature there seems to be a group of values that is very different from the linear relationship between the two values. It might make sense to cluster them, however we already know that these variables are highly correlated so it might even make more sense to just drop one of the two variables.

Furthermore, we see that for windspeed we have values that are 0, which are clearly a separate group for all these pairs. Thus it is likely that windspeed values that are 0 are acutally wrongly recorded values.

3.6 Bike rentals among the months

When we plot the histograms for the different months, we see again clearly the cyclical pattern. While during (hot) summer months many bikes are rented, the overall number decreases during the (cold) winter months. We created a specific color pallete to colour the chart, pointing out the temperature diffence throughout the year but also reflecting to which season which month belongs. (Please note that some months overlap since the seasons end on the 21st. Thus each month is coloured based on the majority of days that belong to that seasons. E.g. March is Winter since 20 days belong to winter).

3.7 Bike rentals among seasons

Looking at the seasonal histograms splitted within the two years, we can see again the cyclical pattern. The overall level of bike rentals is much higher in 2012 compared to 2011. We can conclude that capital bike share increased in popularity within Washington over the time since more bikes are rented out in 2012 than in 2011 for all four seasons.

3.8 Total bike rentals per hour - Season breakdown

In order to compare if the overall pattern changes during the year, we created a line chart using the seaborn library. It might be possible that the peak hours within a day change from season to season due to the changing sunrise and sundown time. However, for all four seasons we see stable peak hours. However the total amount of bikes rented differs between the seasons. While many bikes are rented throughout the summer, much less bikes are rented during winter. For the time during the night (1am to 5am) we see not really large differences among the seasons. Thus it might make sense here to create an addiotional variable that groups the hours of the day.

3.9 Casual users during working and non-working days

For casual users we see a very different pattern compared to registered users. There is a small peak during the week at 5pm but overall the usage is comparatively low. During the weekend though the usage is much higher and increases during lunch time.

3.10 Effect of temperature on bike rentals

In order to detect a general trend that might be related to temperature we created a regression plot with searborn. We see that overall there is positive relationship between temperature and the number of bike rentals. However, when it is extremely hot there are no hours with no bike rentals. Usually those hours are during the day and not during the night, thus it is reasonable that there are bike rentals eventhough it is very hot. So even though it might seem incorrect at first sight, it actually makes sense.

5.1 Splitting Data

I split the data to train it. I have divided it into X_train, X_test, y_train, y_test and fit the algorithm to do a linear regressiojn

5.2 Linear Regression

Bike rentals among the months















