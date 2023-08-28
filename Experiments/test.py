import pandas as pd

# data = {'Category': ['A', 'B', 'A', 'C', 'B', 'A'],
#         'Value': [10, 20, 15, 5, 25, 30]}
df = pd.read_csv('Data/trainLabels.csv')
print(df)

# Get the list of unique categories in the 'Category' column
unique_categories = df['label'].unique().tolist()
labels = pd.factorize(df['label'])[0]+1

print(labels)

print(unique_categories)




