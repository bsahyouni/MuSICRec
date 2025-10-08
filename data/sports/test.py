import pandas as pd

# Load the data
file_path = 'New_Sports_and_Outdoors.txtsp'  # Adjust this to your file's actual path
data = pd.read_csv(file_path, sep='\t', header=None)
data.columns = ['userID', 'itemID', 'rating', 'timestamp', 'x_label']

# Sort the data
sorted_data = data.sort_values(by=['userID', 'timestamp', 'itemID'])

# Calculate the number of unique users and items
unique_users = sorted_data['userID'].nunique()
unique_items = sorted_data['itemID'].nunique()

# Print the results
print(f"Number of unique users: {unique_users}")
print(f"Number of unique items: {unique_items}")