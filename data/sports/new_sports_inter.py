import pandas as pd

# Load the interaction data
sports_inter_path = '/path/to/sports.inter'  # Update with the correct path
sports_inter = pd.read_csv('sports.inter', delimiter='\t')  # Adjust delimiter if needed

# Sort the interactions first by userID, then by itemID, and finally by timestamp
sports_inter_sorted = sports_inter.sort_values(by=['userID', 'timestamp', 'itemID'])

# Calculate the number of unique users and items
unique_users = sports_inter_sorted['userID'].nunique()
unique_items = sports_inter_sorted['itemID'].nunique()

# Print the results
print(f"Number of unique users: {unique_users}")
print(f"Number of unique items: {unique_items}")

# Group by userID and collect itemIDs into lists
grouped_interactions = sports_inter_sorted.groupby('userID')['itemID'].apply(list)

# # Write to a new text file in the same format as sports_and_outdoors.txt
# output_file_path = 'New_Sports_and_Outdoors.txt'  # Update with the desired output path
# with open(output_file_path, 'w') as f:
#     for user_id, items in grouped_interactions.iteritems():
#         sequence = ' '.join(map(str, [user_id] + items))  # Convert list of ints to a space-separated string
#         f.write(sequence + '\n')