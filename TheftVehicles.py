import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, norm

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

# --- Objective 5: Location-Based Theft Heatmap (smaller data) 
subset_df = df[['vehicle_type', 'location_id']].dropna()
top_vehicle_types = subset_df['vehicle_type'].value_counts().nlargest(5).index
top_locations = subset_df['location_id'].value_counts().nlargest(5).index
filtered_df = subset_df[
    subset_df['vehicle_type'].isin(top_vehicle_types) & 
    subset_df['location_id'].isin(top_locations)
]
heatmap_data = filtered_df.pivot_table(index='vehicle_type', columns='location_id', aggfunc=len, fill_value=0)

plt.subplot(3, 2, plot_idx)
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Theft Count'})
plt.title("Heatmap of Stolen Vehicle Types by Location")
plt.xlabel("Location ID")
plt.ylabel("Vehicle Type")

plt.tight_layout()
plt.show()

# Statistical Tests

# Test 1: Chi-Square Test between 'vehicle_type' and 'color'
if 'vehicle_type' in df.columns and 'color' in df.columns:
    chi_df = df[['vehicle_type', 'color']].dropna()
    if not chi_df.empty:
        contingency = pd.crosstab(chi_df['vehicle_type'], chi_df['color'])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(contingency)
            print("\nChi-Square Test between vehicle_type and color:")
            print(f"P-Value = {p:.4f} →", "Significant" if p < 0.05 else "Not Significant")

# Test 2: Z-Test - Model Year vs Hypothetical Mean (e.g., 2010)
model_years = df['model_year'].dropna()
if len(model_years) > 30:
    hypothesized_mean = 2010
    sample_mean = model_years.mean()
    std_dev = model_years.std(ddof=1)
    sample_size = len(model_years)
    z = (sample_mean - hypothesized_mean) / (std_dev / np.sqrt(sample_size))
    p_val = 2 * (1 - norm.cdf(abs(z)))
    print(f"\nZ-Test on Model Year vs {hypothesized_mean}:")
    print(f"Z-Score = {z:.2f}, P-Value = {p_val:.4f} →", "Significant" if p_val < 0.05 else "Not Significant")
