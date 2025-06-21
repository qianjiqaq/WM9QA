"""
The code aims to visualize single variables from a retail dataset.
The dataset contains sales data with features like Units Sold, Price, Discount, Category, Region, Weather Condition, and Date.
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/Users/zhangjiawei/Desktop/python_files/warwick_course/EDA/US_1_Retail_FINAL.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Histogram of Units Sold
plt.figure()
plt.hist(df['Units Sold'], bins=30)
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Boxplot of Units Sold
units = df['Units Sold']
q1 = units.quantile(0.25)
q3 = units.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5*iqr
upper = q3 + 1.5*iqr
num_low = (units < lower).sum()
num_high = (units > upper).sum()
plt.figure()
plt.boxplot(
    units,
    vert=True,
    showfliers=False,
    whis=1.5
)
plt.ylabel('Units Sold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.text(1.15, upper, f'{num_high} high outliers', va='center')
plt.text(1.15, lower, f'{num_low} low outliers', va='center')
plt.tight_layout()
plt.show()

# Histogram of Price
plt.figure()
plt.hist(df['Price'], bins=30)
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Histogram of Discount
plt.figure()
plt.hist(df['Discount'], bins=30)
plt.xlabel('Discount (USD)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Total Units Sold by Category
category_sales = df.groupby('Category')['Units Sold'].sum().sort_values(ascending=False)
plt.figure()
plt.bar(category_sales.index, category_sales.values)
plt.ylabel('Total Units Sold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Daily Total Units Sold over Time
daily_sales = df.groupby('Date')['Units Sold'].sum()
plt.figure()
plt.plot(daily_sales.index, daily_sales.values)
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.tight_layout()
plt.show()