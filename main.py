import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.layouts import row

# Read CSV and import shopping data - creating 'shoppers_df' DataFrame
shoppers_df = pd.read_csv(r'C:\Users\v-dakerv\PycharmProjects\Dale_project\Main_project_Feb_Project\Shopping_CustomerData.csv')
print(shoppers_df.head())

# Identify missing data - no missing data.
print(shoppers_df.isna().sum())

# Consider field customer research - imported from pen and paper. Create second DataFrame 'customer_data_df'
customer_research = {
    'Profession': ['Engineer', 'Healthcare', 'Unemployed', 'Executive', 'Marketing', 'Doctor', 'Doctor', 'Artist',
                   'Artist', 'Healthcare', 'Healthcare', 'Healthcare', 'Artist', 'Healthcare', 'Doctor', 'Artist',
                   'Lawyer', 'Artist', 'Artist', 'Lawyer', 'Executive', 'Lawyer', 'Artist', 'Executive', 'Healthcare',
                   'Executive', 'Engineer', 'Healthcare', 'Lawyer', 'Lawyer', 'Lawyer', 'Artist', 'Artist', 'Artist',
                   'Doctor', 'Artist', 'Artist', 'Artist', 'Artist', 'Artist', 'Healthcare', 'Executive', 'Healthcare',
                   'Artist', 'Artist', 'Lawyer', 'Entertainment', 'Artist', 'Healthcare', 'Artist', 'Lawyer', 'Lawyer',
                   'Doctor', 'Doctor', 'Executive', 'Lawyer', 'Entertainment', 'Artist', 'Artist', 'Artist',
                   'Healthcare', 'Artist', 'Artist', 'Artist', 'Entertainment', 'Entertainment', 'Artist', 'Lawyer',
                   'Healthcare', 'Artist', 'Entertainment', 'Artist', 'Artist', 'Healthcare', 'Artist', 'Lawyer',
                   'Entertainment', 'Artist', 'Doctor', 'Artist', 'Engineer', 'Lawyer', 'Lawyer', 'Artist', 'Artist',
                   'Executive', 'Lawyer', 'Healthcare', 'Artist', 'Healthcare', 'Homemaker', 'Lawyer', 'Lawyer',
                   'Artist', 'Doctor', 'Doctor', 'Artist', 'Healthcare', 'Artist', 'Artist', 'Entertainment',
                   'Entertainment', 'Healthcare', 'Engineer', 'Executive', 'Marketing', 'Doctor', 'Executive',
                   'Healthcare', 'Artist', 'Lawyer', 'Artist', 'Doctor', 'Artist', 'Artist', 'Homemaker', 'Healthcare',
                   'Engineer', 'Executive', 'Artist', 'Marketing', 'Engineer', 'Executive', 'Artist', 'Engineer',
                   'Artist', 'Artist', 'Lawyer', 'Engineer', 'Entertainment', 'Artist', 'Lawyer', 'Healthcare',
                   'Artist', 'Doctor', 'Doctor', 'Lawyer', 'Lawyer', 'Actress', 'Artist', 'Doctor', 'Executive',
                   'Doctor', 'Lawyer', 'Healthcare', 'Entertainment', 'Healthcare', 'Doctor', 'Marketing', 'Lawyer',
                   'Artist', 'Artist', 'Artist', 'Artist', 'Artist', 'Artist', 'Lawyer', 'Marketing', 'Artist',
                   'Artist', 'Artist', 'Artist', 'Artist', 'Healthcare', 'Artist', 'Entertainment', 'Lawyer',
                   'Marketing', 'Lawyer', 'Artist', 'Lawyer', 'Lawyer', 'Healthcare', 'Engineer', 'Lawyer',
                   'Healthcare', 'Lawyer', 'Engineer', 'Artist', 'CEO', 'Artist', 'Healthcare', 'Homemaker', 'Artist',
                   'Healthcare', 'Artist', 'Entertainment', 'Lawyer', 'Artist', 'Doctor', 'Executive', 'Lawyer',
                   'Artist', 'Pilot', 'Entertainment', 'Artist', 'Artist', 'Artist', 'Artist', 'Artist', 'Healthcare',
                   'Artist', 'Artist', 'Lawyer'],
    'Work_Experience': [0, 8, 0, 11, 1, 0, 5, 1, 2, 0, 0, 0, 1, 8, 0, 0, 1, 1, 8, 1, 4, 4, 0, 5, 9, 5, 3, 0, 1, 4, 1, 9,
                        1, 1, 9, 11, 0, 1, 2, 0, 2, 8, 5, 3, 1, 1, 6, 0, 7, 0, 2, 5, 0, 1, 3, 0, 0, 0, 1, 0, 3, 1, 1, 1,
                        0, 7, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 1, 0, 9, 1, 0, 1, 1, 1, 1, 9, 0, 6, 1, 1, 0, 0, 0, 1,
                        0, 2, 0, 1, 1, 0, 1, 1, 0, 0, 1, 5, 8, 1, 1, 1, 4, 6, 2, 4, 5, 2, 0, 2, 0, 1, 0, 0, 0, 1, 1, 1,
                        0, 8, 4, 4, 1, 5, 1, 1, 0, 1, 1, 2, 0, 0, 1, 1, 0, 2, 8, 5, 0, 0, 0, 1, 0, 9, 5, 1, 2, 2, 4, 1,
                        1, 0, 0, 4, 2, 7, 0, 1, 3, 5, 0, 1, 6, 2, 4, 1, 1, 1, 5, 2, 8, 7, 13, 1, 8, 8, 1, 2, 13, 0, 1,
                        1, 3, 4, 4, 0, 1, 7, 0, 1, 0, 6, 0, 2],
    'Married': ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No',
                'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes',
                'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes',
                'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No',
                'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes',
                'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes',
                'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No',
                'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes',
                'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes',
                'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes',
                'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes',
                'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'N/A', 'Yes', 'Yes', 'Yes',
                'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes',
                'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'Family_Size': [1, 4, 1, 2, 4, 5, 3, 3, 4, 4, 3, 6, 5, 3, 3, 1, 3, 2, 1, 4, 3, 0, 4, 2, 3, 4, 2, 4, 2, 2, 2, 3, 2,
                    6, 2, 1, 2, 2, 5, 2, 8, 3, 5, 6, 2, 2, 2, 2, 6, 4, 2, 1, 1, 2, 5, 2, 1, 2, 1, 2, 3, 2, 3, 2, 4, 1,
                    1, 2, 3, 2, 4, 2, 2, 4, 2, 2, 1, 3, 4, 2, 4, 4, 2, 6, 1, 5, 1, 4, 2, 5, 2, 2, 1, 2, 1, 2, 2, 1, 2,
                    0, 1, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 4, 4, 2, 3, 1, 3, 2, 2, 1, 2, 7, 4, 3, 1, 3, 2, 2, 4, 2, 2, 2,
                    3, 4, 2, 3, 2, 2, 3, 2, 4, 2, 2, 2, 4, 4, 5, 3, 2, 3, 2, 2, 3, 1, 2, 2, 1, 2, 1, 1, 2, 3, 2, 6, 8,
                    3, 2, 4, 2, 3, 1, 2, 4, 1, 2, 1, 2, 1, 2, 4, 2, 2, 1, 2, 5, 1, 3, 1, 2, 2, 4, 5, 5, 2, 3, 1, 4, 3,
                    1, 3, 3, 1, 2, 2]}

customer_data_df = pd.DataFrame(customer_research)
print(customer_data_df.shape)

# Merge DataFrames to create our working 'customer_base' dataset for exploration
customer_base = pd.concat([shoppers_df, customer_data_df], axis=1)

# Dropping duplicates for CustomerID - total: 3.
customer_base = customer_base.drop_duplicates(subset=['CustomerID'])
print(customer_base.shape)

# Assigning any missing values
avg_customer_age = customer_base['CustomerAge'].mean()
avg_customer_income = customer_base['AnnualIncome'].mean()
customer_base = customer_base.fillna({'CustomerAge': avg_customer_age, 'CustomerGender': 'Unknown',
                                      'CustomerCity': 'Unknown', 'AnnualIncome': avg_customer_income,
                                      'CreditScore': 'Unknown', 'SpendingScore': 'Unknown', 'CustomerCityID':
                                          'Unknown'})

# Removal of any remaining missing values - total: 1.
customer_base = customer_base.dropna()
print(customer_base)

# Set Index to ID
customer_base = customer_base.set_index('CustomerID')

# Numpy - to find average Customer Age which is 45.52 years old
age_array = np.array(customer_base['CustomerAge'])
print(age_array.mean())

# Determine average family size - 2 to 3 people.
family_array = np.array(customer_base['Family_Size'])
print(family_array.mean())

# Determine highest consumer income - $695'407
income_array = np.array(customer_base['AnnualIncome'].max())
print(income_array.max())

# iterrows
for location, hi in customer_base.iterrows():
    customer_base.loc[location, "CustomerPlace"] = hi["CustomerCity"].upper()
print(customer_base)

# for loop identifying how many people in capital city of Delhi
city_livers = 0
for Delhi in customer_base['CustomerCity']:
    if Delhi == 'Delhi':
        city_livers += 1
print(city_livers, 'members of our existing customer base are living in Delhi')

# Segmenting our customer_base - 18-75 year olds.
(customer_base['CustomerAge'].min())
(customer_base['CustomerAge'].max())
BabyBoomers = customer_base[(customer_base['CustomerAge'] > 57) & (customer_base['CustomerAge'] <= 75)]
GenX = customer_base[(customer_base['CustomerAge'] > 41) & (customer_base['CustomerAge'] <= 56)]
GenY = customer_base[(customer_base['CustomerAge'] > 25) & (customer_base['CustomerAge'] <= 40)]
GenZ = customer_base[(customer_base['CustomerAge'] > 6) & (customer_base['CustomerAge'] <= 24)]

# View FULL customer_base DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(customer_base)

# Visualisation - Fig.1 pandas Bar Chart displaying Average Salary by Area
avg_salary_by_area = customer_base.groupby("CustomerCity")["AnnualIncome"].mean()
avg_salary_by_area = avg_salary_by_area.plot(kind="bar", title="Average Salary by Area", rot=45)
avg_salary_by_area.plot(figsize=(8, 5))
avg_salary_by_area.figure.savefig('1. Pandas Bar Chart – Average Salary by Area.png')

# Visualisation - Fig.2 Histogram showing Gen Z v. Baby Boomer salaries i.e. Oldest v youngest segment
fig, ax = plt.subplots()
annual_inc = ax.hist(GenZ["AnnualIncome"], label="Gen Z", bins=8, histtype="step")
annual_inc = ax.hist(BabyBoomers["AnnualIncome"], label="Baby Boomers", bins=8, histtype="step")
annual_inc = ax.set_xlabel("AnnualIncome ($k)")
annual_inc = ax.set_ylabel("No. of observations")
ax.set_title('Gen Z v. Baby Boomer Annual Income')
ax.legend()
annual_inc.figure.savefig('2. Gen Z v. Baby Boomer Annual Income.png')

# Visualisation using For loop - to show Fig.3 No. of customers per city.
locations = customer_base["CustomerCity"].unique()
print(locations)
fig, ax = plt.subplots(figsize=(10, 8))
for city in locations:
    area_spread_df = customer_base[customer_base["CustomerCity"] == city]
    ax.bar(city, area_spread_df["CustomerCity"].count())
ax.set_ylabel("No. of customers")
ax.set_xticklabels(locations, rotation=45)
ax.set_title('No. of Customers per City')
plt.grid(True)
fig.savefig('3. Bar Chart No. of Customers per City.png')
plt.show()

# Visualisation using Seaborn - Fig.4 Relationship between Spending Score and Credit Score
plt.subplots(figsize=(10,6))
Spending_Credit = sns.scatterplot(x="SpendingScore", y="CreditScore", data=customer_base, hue="Profession",
                                  size="AnnualIncome", style="Married", alpha=0.9)
Spending_Credit.set_title("Relationship between Spending Score and Credit Score", y=1.01)
plt.legend(bbox_to_anchor=(1.0, 1.0), borderaxespad=0)
plt.subplots_adjust(right=0.8)
plt.savefig('4. Relationship between Spending Score and Credit Score.png', bbox_inches="tight")
plt.show()

# Visualisation using Seaborn - Fig.5 Count plot showing gender split of customer base and married split.
Gender_Marriage = sns.countplot(y="CustomerGender", data=customer_base, hue="Married")
Gender_Marriage.set_title("Customer Base Gender and Marriage split", y=1.01)
plt.savefig('5. Customer Base Gender and Marriage Split.png')
plt.show()

# Visualisation using Bokeh - Further exploration of Spending and Credit Score in relation to Annual Income.
p1 = figure(title='Relationship between Annual Income and Spending Score', x_axis_label='Spending Score',
            y_axis_label='Annual Income ($k)')
p1.circle('SpendingScore', 'AnnualIncome', source=customer_base, size=10, color='red', legend_group='CustomerCityID')
p2 = figure(title='Relationship between Annual Income and Credit Score', x_axis_label='Credit Score',
            y_axis_label='Annual Income ($k)')
p2.circle('CreditScore', 'AnnualIncome', source=customer_base, size=10, color='blue', legend_group='CustomerCityID')
chart = row(p1, p2)
p1.x_range = p2.x_range
p2.y_range = p1.y_range
hover = HoverTool(tooltips=[('CustomerID', '@CustomerID')])
output_file('Spending and Credit Score in relation to Annual Income.html')
show(chart)

