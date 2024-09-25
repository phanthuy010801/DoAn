
import pandas as pd
import os

df = pd.read_csv("label.csv", on_bad_lines='skip', encoding="latin-1")

PATH = 'data'

rows_to_remove = []

for i, row in df.iterrows():
    website = os.path.join(row['website'])

    if not os.path.exists(os.path.join(PATH, website)):
        print(website)
        rows_to_remove.append(i)

# Remove rows with the specified indices
df = df.drop(rows_to_remove, axis=0)

# Reset the index after removing rows
df = df.reset_index(drop=True)

# Now 'df' contains rows where the websites exist in the specified directory
print(df)

df.to_csv('label.csv', index=False)
