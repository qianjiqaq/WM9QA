import pandas as pd

df = pd.read_csv('/Users/zhangjiawei/Desktop/python_files/warwick_course/US_1_Retail_AFTERDC.csv', parse_dates=['Date'])

df['month'] = df['Date'].dt.month
def month2season(m):
    if m in [3, 4, 5]:
        return 'Spring'
    elif m in [6, 7, 8]:
        return 'Summer'
    elif m in [9, 10, 11]:
        return 'Autumn'
    # 1,2,12
    else:  
        return 'Winter'

df['season'] = df['month'].apply(month2season)


df.to_csv('/Users/zhangjiawei/Desktop/python_files/warwick_course/US_1_Retail_FINAL.csv', index=False)