import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df_data = pd.read_csv('nobel_prize_data.csv')

#Challenge 1
#Preliminary df_data exploration.

print("What is the shape of df_df_data? How many rows and columns?")
print(df_data.shape)
print("What are the column names and what kind of df_data is inside of them?")
print(df_data.columns)
print("In which year was the Nobel prize first awarded?")
print(f"{df_data.head()}")
print("Which year is the latest year included in the df_dataset?")
print(f"{df_data.tail()}")
print(f"{df_data.sample()}")
"""
print("")
print("Info about the df_dataframe:")
print(df_data.info())
"""

#Challenge 2
print("Are there any duplicate values in the df_dataset?")
print(df_data.duplicated().values.any())
print("Are there NaN values in the df_dataset?")
print(df_data.isna().values.any())
print("")
print(df_data.isna().sum())
print("Which columns tend to have NaN values?")
print("How many NaN values are there per column?")
print("Why do these columns have NaN values?")

print("")
col_subset = ['year','category', 'laureate_type',
              'birth_date','full_name', 'organization_name']
print(df_data.loc[df_data.birth_date.isna()][col_subset])
print("")
col_subset = ['year','category', 'laureate_type','full_name', 'organization_name']
print(df_data.loc[df_data.organization_name.isna()][col_subset])
print("")


#Challenge 3

print("Convert the birth_date column to Pandas Datetime objects")
df_data.birth_date = pd.to_datetime(df_data.birth_date)
print("Add a Column called share_pct which has the laureates' share as a percentage in the form of a floating-point number.")
values = df_data.prize_share.str.split("/", expand=True)
num = pd.to_numeric(values[0])
den = pd.to_numeric(values[1])
df_data["share_pct"] = num / den
print(df_data.info())

##plotly Bar & Donut Charts: Analyse Prize Categories & Women Winning Prizes
#Challenge 1
print("Create a donut chart using plotly which shows how many prizes went to men compared to how many prizes went to women")
print("What percentage of all the prizes went to women?")
biology = df_data.sex.value_counts()
fig = px.pie(labels=biology.index,
             values=biology.values,
             title="Percentage of Male vs. Female Winners",
             names=biology.index,
             hole=0.4,)
fig.update_traces(textposition='inside', textfont_size=15, textinfo='percent')
fig.show()

#Challenge 2
print("What are the names of the first 3 female Nobel laureates?")
print("What did the win the prize for?")
print("What do you see in their birth_country? Were they part of an organisation?")
col_subset = ['year','category', 'laureate_type','full_name',
              'organization_name', 'birth_country']
print(df_data[df_data.sex == "Female"].sort_values("year",
                                                   ascending=True)[col_subset][:3])

#Challenge 3
print("Did some people get a Nobel Prize more than once? If so, who were they?")
is_winner = df_data.duplicated(subset=['full_name'], keep=False)
multiple_winners = df_data[is_winner]
print(f'There are {multiple_winners.full_name.nunique()}' \
      ' winners who were awarded the prize more than once.')
col_subset = ['year', 'category', 'laureate_type', 'full_name']
print(multiple_winners[col_subset])

#Challenge 4
print("In how many categories are prizes awarded?")
print(df_data.category.nunique())
print("Create a plotly bar chart with the number of prizes awarded by category.")
print("Use the color scale called Aggrnyl to colour the chart, but don't show a color axis.")
print("Which category has the most number of prizes awarded?")
print("Which category has the fewest number of prizes awarded?")
prizes_per_category = df_data.category.value_counts()
v_bar = px.bar(
        x = prizes_per_category.index,
        y = prizes_per_category.values,
        color = prizes_per_category.values,
        color_continuous_scale='Aggrnyl',
        title='Number of Prizes Awarded per Category')

v_bar.update_layout(xaxis_title='Nobel Prize Category',
                    coloraxis_showscale=False,
                    yaxis_title='Number of Prizes')
v_bar.show()

#Challenge 5
print("When was the first prize in the field of Economics awarded?")
print("Who did the prize go to?")
print(df_data[df_data.category == 'Economics'].sort_values('year')[:3])

#Challenge 6
print("Create a plotly bar chart that shows the split between men and women by category.")
print("Hover over the bar chart. How many prizes went to women in Literature compared to Physics?")
cat_men_women = df_data.groupby(['category', 'sex'],
                                as_index=False).agg({'prize': pd.Series.count})
cat_men_women.sort_values('prize', ascending=False, inplace=True)
v_bar_split = px.bar(x = cat_men_women.category,
                     y = cat_men_women.prize,
                     color = cat_men_women.sex,
                     title='Number of Prizes Awarded per Category split by Men and Women')

v_bar_split.update_layout(xaxis_title='Nobel Prize Category',
                          yaxis_title='Number of Prizes')
v_bar_split.show()

##Visualize trends over time:
#Challenge 1
print("Are more prizes awarded recently than when the prize was first created? Show the trend in awards visually.")
print("Count the number of prizes awarded every year.")
prize_per_year = df_data.groupby(by='year').count().prize
print("Create a 5 year rolling average of the number of prizes (Hint: see previous lessons analysing Google Trends).")
moving_average = prize_per_year.rolling(window=5).mean()
print("Using Matplotlib superimpose the rolling average on a scatter plot.")
plt.scatter(x=prize_per_year.index,
           y=prize_per_year.values,
           c='dodgerblue',
           alpha=0.7,
           s=100,)

plt.plot(prize_per_year.index,
        moving_average.values,
        c='crimson',
        linewidth=3,)
plt.show()
print("Show a tick mark on the x-axis for every 5 years from 1900 to 2020. (Hint: you'll need to use NumPy).")
np.arange(1900, 2021, step=5)
print("Use the named colours to draw the data points in dogerblue while the rolling average is coloured in crimson.")
plt.figure(figsize=(16,8), dpi=200)
plt.title('Number of Nobel Prizes Awarded per Year', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(ticks=np.arange(1900, 2021, step=5),
           fontsize=14,
           rotation=45)

ax = plt.gca() # get current axis
ax.set_xlim(1900, 2020)

ax.scatter(x=prize_per_year.index,
           y=prize_per_year.values,
           c='dodgerblue',
           alpha=0.7,
           s=100,)

ax.plot(prize_per_year.index,
        moving_average.values,
        c='crimson',
        linewidth=3,)

plt.show()
print("Looking at the chart, did the first and second world wars have an impact on the number of prizes being given out?")
print("What could be the reason for the trend in the chart?")

#Challenge 2

print("Investigate if more prizes are shared than before.")
print("Calculate the average prize share of the winners on a year by year basis.")
yearly_avg_share = df_data.groupby(by='year').agg({'share_pct': pd.Series.mean})
print("Calculate the 5 year rolling average of the percentage share.")
share_moving_average = yearly_avg_share.rolling(window=5).mean()
print("Copy-paste the cell from the chart you created above.")
print("Modify the code to add a secondary axis to your Matplotlib chart.")
print("Plot the rolling average of the prize share on this chart.")
plt.figure(figsize=(16,8), dpi=200)
plt.title('Number of Nobel Prizes Awarded per Year', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(ticks=np.arange(1900, 2021, step=5),
           fontsize=14,
           rotation=45)

ax1 = plt.gca()
ax2 = ax1.twinx() # create second y-axis
ax1.set_xlim(1900, 2020)

ax1.scatter(x=prize_per_year.index,
           y=prize_per_year.values,
           c='dodgerblue',
           alpha=0.7,
           s=100,)

ax1.plot(prize_per_year.index,
        moving_average.values,
        c='crimson',
        linewidth=3,)

# Adding prize share plot on second axis
ax2.plot(prize_per_year.index,
        share_moving_average.values,
        c='grey',
        linewidth=3,)

plt.show()
print("See if you can invert the secondary y-axis to make the relationship even more clear.")
plt.figure(figsize=(16,8), dpi=200)
plt.title('Number of Nobel Prizes Awarded per Year', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(ticks=np.arange(1900, 2021, step=5),
           fontsize=14,
           rotation=45)

ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.set_xlim(1900, 2020)

# Can invert axis
ax2.invert_yaxis()

ax1.scatter(x=prize_per_year.index,
           y=prize_per_year.values,
           c='dodgerblue',
           alpha=0.7,
           s=100,)

ax1.plot(prize_per_year.index,
        moving_average.values,
        c='crimson',
        linewidth=3,)

ax2.plot(prize_per_year.index,
        share_moving_average.values,
        c='grey',
        linewidth=3,)

plt.show()

##A Choropleth Map and the Countries with the Most Prizes
#Challenge 1: Top 20 Country Ranking
top_countries = df_data.groupby(['birth_country_current'],
                                  as_index=False).agg({'prize': pd.Series.count})
top_countries.sort_values(by='prize', inplace=True)
top20_countries = top_countries[-20:]
h_bar = px.bar(x=top20_countries.prize,
               y=top20_countries.birth_country_current,
               orientation='h',
               color=top20_countries.prize,
               color_continuous_scale='Viridis',
               title='Top 20 Countries by Number of Prizes')

h_bar.update_layout(xaxis_title='Number of Prizes',
                    yaxis_title='Country',
                    coloraxis_showscale=False)
h_bar.show()

#Challenge 2: Choropleth Map
df_countries = df_data.groupby(['birth_country_current', 'ISO'],
                               as_index=False).agg({'prize': pd.Series.count})
df_countries.sort_values('prize', ascending=False)
world_map = px.choropleth(df_countries,
                          locations='ISO',
                          color='prize',
                          hover_name='birth_country_current',
                          color_continuous_scale=px.colors.sequential.matter)

world_map.update_layout(coloraxis_showscale=True,)

world_map.show()

#Challenge 3: Country Bar Chart with Prize Category
cat_country = df_data.groupby(['birth_country_current', 'category'],
                               as_index=False).agg({'prize': pd.Series.count})
cat_country.sort_values(by='prize', ascending=False, inplace=True)
merged_df = pd.merge(cat_country, top20_countries, on='birth_country_current')
# change column names
merged_df.columns = ['birth_country_current', 'category', 'cat_prize', 'total_prize']
merged_df.sort_values(by='total_prize', inplace=True)
cat_cntry_bar = px.bar(x=merged_df.cat_prize,
                       y=merged_df.birth_country_current,
                       color=merged_df.category,
                       orientation='h',
                       title='Top 20 Countries by Number of Prizes and Category')

cat_cntry_bar.update_layout(xaxis_title='Number of Prizes',
                            yaxis_title='Country')
cat_cntry_bar.show()

#Challenge 4: Prizes by Country over Time
prize_by_year = df_data.groupby(by=['birth_country_current', 'year'], as_index=False).count()
prize_by_year = prize_by_year.sort_values('year')[['year', 'birth_country_current', 'prize']]
cumulative_prizes = prize_by_year.groupby(by=['birth_country_current',
                                              'year']).sum().groupby(level=[0]).cumsum()
cumulative_prizes.reset_index(inplace=True)
l_chart = px.line(cumulative_prizes,
                  x='year',
                  y='prize',
                  color='birth_country_current',
                  hover_name='birth_country_current')

l_chart.update_layout(xaxis_title='Year',
                      yaxis_title='Number of Prizes')

l_chart.show()

##Create Sunburst Charts for a Detailed Regional Breakdown of Research Locations
#Challenge 1: The Top Research Organisations
top20_orgs = df_data.organization_name.value_counts()[:20]
top20_orgs.sort_values(ascending=True, inplace=True)
org_bar = px.bar(x = top20_orgs.values,
                 y = top20_orgs.index,
                 orientation='h',
                 color=top20_orgs.values,
                 color_continuous_scale=px.colors.sequential.haline,
                 title='Top 20 Research Institutions by Number of Prizes')

org_bar.update_layout(xaxis_title='Number of Prizes',
                      yaxis_title='Institution',
                      coloraxis_showscale=False)
org_bar.show()
#Challenge 2: Research Cities
top20_org_cities = df_data.organization_city.value_counts()[:20]
top20_org_cities.sort_values(ascending=True, inplace=True)
city_bar2 = px.bar(x = top20_org_cities.values,
                   y = top20_org_cities.index,
                   orientation='h',
                   color=top20_org_cities.values,
                   color_continuous_scale=px.colors.sequential.Plasma,
                   title='Which Cities Do the Most Research?')

city_bar2.update_layout(xaxis_title='Number of Prizes',
                        yaxis_title='City',
                        coloraxis_showscale=False)
city_bar2.show()
#Challenge 3: Laureate Birth Cities
top20_cities = df_data.birth_city.value_counts()[:20]
top20_cities.sort_values(ascending=True, inplace=True)
city_bar = px.bar(x=top20_cities.values,
                  y=top20_cities.index,
                  orientation='h',
                  color=top20_cities.values,
                  color_continuous_scale=px.colors.sequential.Plasma,
                  title='Where were the Nobel Laureates Born?')

city_bar.update_layout(xaxis_title='Number of Prizes',
                       yaxis_title='City of Birth',
                       coloraxis_showscale=False)
city_bar.show()
#Challenge 4: The Sunburst Chart
country_city_org = df_data.groupby(by=['organization_country',
                                       'organization_city',
                                       'organization_name'],
                                   as_index=False).agg({'prize': pd.Series.count})

country_city_org = country_city_org.sort_values('prize', ascending=False)
burst = px.sunburst(country_city_org,
                    path=['organization_country', 'organization_city', 'organization_name'],
                    values='prize',
                    title='Where do Discoveries Take Place?',)

burst.update_layout(xaxis_title='Number of Prizes',
                    yaxis_title='City',
                    coloraxis_showscale=False)

burst.show()

##Unearthing Patterns in the Laureate Age at the Time of the Award

#Calculate the Age at the Time of Award
print("Calculate the Age at the Time of Award")
birth_years = df_data.birth_date.dt.year
df_data['winning_age'] = df_data.year - birth_years

#Oldest and Youngest Winners
print("Oldest and Youngest Winners")
print(df_data.nlargest(n=1, columns='winning_age'))
print(df_data.nsmallest(n=1, columns='winning_age'))

#Descriptive Statistics and Histogram
plt.figure(figsize=(8, 4), dpi=200)
sns.histplot(data=df_data,
             x=df_data.winning_age,
             bins=30)
plt.xlabel('Age')
plt.title('Distribution of Age on Receipt of Prize')
plt.show()

#Winning Age Over Time (All Categories)
plt.figure(figsize=(8,4), dpi=200)
with sns.axes_style("whitegrid"):
    sns.regplot(data=df_data,
                x='year',
                y='winning_age',
                lowess=True,
                scatter_kws = {'alpha': 0.4},
                line_kws={'color': 'black'})

plt.show()

#Age Differences between Categories
plt.figure(figsize=(8,4), dpi=200)
with sns.axes_style("whitegrid"):
    sns.boxplot(data=df_data,
                x='category',
                y='winning_age')

plt.show()

#Laureate Age over Time by Category
with sns.axes_style('whitegrid'):
    sns.lmplot(data=df_data,
               x='year',
               y='winning_age',
               row = 'category',
               lowess=True,
               aspect=2,
               scatter_kws = {'alpha': 0.6},
               line_kws = {'color': 'black'},)

plt.show()

with sns.axes_style("whitegrid"):
    sns.lmplot(data=df_data,
               x='year',
               y='winning_age',
               hue='category',
               lowess=True,
               aspect=2,
               scatter_kws={'alpha': 0.5},
               line_kws={'linewidth': 5})

plt.show()
