import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/stolen_vehicles.csv")

# Convert date_stolen to datetime
df['date_stolen'] = pd.to_datetime(df['date_stolen'], errors='coerce')

# Set up a figure for multiple plots
plt.figure(figsize=(20, 25))
plot_idx = 1

# --- Objective 1: Temporal Analysis of Vehicle Theft Trends ---
monthly_thefts = df['date_stolen'].dropna().dt.to_period("M").value_counts().sort_index()
plt.subplot(3, 2, plot_idx)
monthly_thefts.plot(kind='line', marker='o')
plt.title("Monthly Stolen Vehicle Trend")
plt.xlabel("Month")
plt.ylabel("Number of Thefts")
plot_idx += 1

# --- Objective 2: Vehicle Type Theft Distribution (pie chart) ---
plt.subplot(3, 2, plot_idx)
vehicle_counts = df['vehicle_type'].value_counts().iloc[:10]
plt.pie(vehicle_counts.values, labels=vehicle_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 10 Stolen Vehicle Types (Pie Chart)")
plot_idx += 1


# --- Objective 3: Model Year vs. Theft Frequency (box plot) ---
plt.subplot(3, 2, plot_idx)
sns.boxplot(x=df['model_year'].dropna())
plt.title("Theft Frequency by Model Year (Box Plot)")
plt.xlabel("Model Year")
plot_idx += 1

# --- Objective 4: Color-Based Theft Analysis (scatter plot) ---
plt.subplot(3, 2, plot_idx)
top_colors = df['color'].value_counts().iloc[:10]
sns.scatterplot(x=top_colors.values, y=top_colors.index, s=100, color='purple')
plt.title("Top 10 Vehicle Colors in Theft Cases (Scatter Plot)")
plt.xlabel("Count")
plt.ylabel("Color")
plot_idx += 1

# --- Objective 5: Location-Based Theft Heatmap (smaller data) ---
subset_df = df[['vehicle_type', 'location_id']].dropna()
top_vehicle_types = subset_df['vehicle_type'].value_counts().nlargest(5).index
top_locations = subset_df['location_id'].value_counts().nlargest(5).index
filtered_df = subset_df[subset_df['vehicle_type'].isin(top_vehicle_types) & 
                        subset_df['location_id'].isin(top_locations)]
heatmap_data = filtered_df.pivot_table(index='vehicle_type', columns='location_id', aggfunc=len, fill_value=0)

plt.subplot(3, 2, plot_idx)
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Theft Count'})
plt.title("Heatmap of Stolen Vehicle Types by Location")
plt.xlabel("Location ID")
plt.ylabel("Vehicle Type")

plt.tight_layout()
plt.show()
