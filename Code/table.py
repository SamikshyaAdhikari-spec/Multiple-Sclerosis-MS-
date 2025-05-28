import pandas as pd

# Define file path
csv_path = "C:/Users/samik/Documents/GitHub/MS-disease/lesion_areas1.csv"  # Adjust path accordingly

# Load the CSV file
df_lesions = pd.read_csv(csv_path)

# Extract relevant columns
# extracts only width and height of bounding box
# converts each bounding box from a string to a tuple
df_lesions["Bounding Box (w, h)"] = df_lesions["Bounding Box (x, y, w, h)"].apply(
    lambda x: tuple(map(int, x.strip("()").split(", ")[2:]))
)

# Count number of lesions with the same bounding box size
df_summary = df_lesions.groupby(["Bounding Box (w, h)"]).agg(
    {"Lesion Name": "count"}
).reset_index()

# Rename columns for clarity
df_summary.rename(columns={"Lesion Name": "Number of Lesions"}, inplace=True)

# Save the summary as a new CSV file
summary_csv_path = "C:/Users/samik/Documents/GitHub/MS-disease/lesion_summary.csv"
df_summary.to_csv(summary_csv_path, index=False)

print(f"Lesion summary saved to: {summary_csv_path}")

# Display the table
print(df_summary)
