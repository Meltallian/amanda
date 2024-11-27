import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load and clean the data
data = pd.read_csv('data.csv')

# Isolate the columns of interest
columns_of_interest = data.loc[:, '1STA':'44PER']

# Convert all columns of interest to numeric, coerce errors to NaN
columns_of_interest = columns_of_interest.apply(pd.to_numeric, errors='coerce')
# Remove rows that contain only 1s or only 4s
filtered_data = columns_of_interest[
    ~columns_of_interest.eq(1).all(axis=1) & ~columns_of_interest.eq(4).all(axis=1)
]

sex_data = data['Sexe']  # Assuming 'Sexe' column exists in the original dataset
# Align sex_data with filtered_data indices
sex_data_aligned = sex_data.loc[filtered_data.index]

# Add 'Sex' column to filtered_data
filtered_data = filtered_data.assign(Sex=sex_data_aligned.values)

# Verify the column addition
# print(filtered_data.columns)




# Define a function to check the conditions with NaN handling
def check_conditions(row):
    conditions_failed = 0
    
    # Condition 1: Check 30th and 32nd columns
    if pd.notna(row.iloc[29]) and pd.notna(row.iloc[31]):
        if ((row.iloc[29] == 1 and row.iloc[31] == 4) or (row.iloc[29] == 4 and row.iloc[31] == 1)):
            conditions_failed += 1
    
    # Condition 2: Check 23rd and 34th columns
    if pd.notna(row.iloc[22]) and pd.notna(row.iloc[33]):
        if ((row.iloc[22] == 1 and row.iloc[33] == 4) or (row.iloc[22] == 4 and row.iloc[33] == 1)):
            conditions_failed += 1
    
    # Condition 3: Check 36th and 19th columns
    if pd.notna(row.iloc[35]) and pd.notna(row.iloc[18]):
        if ((row.iloc[35] == 1 and row.iloc[18] == 4) or (row.iloc[35] == 4 and row.iloc[18] == 1)):
            conditions_failed += 1
    
    # Omit the row if it fails at least 2 out of 3 conditions
    return conditions_failed < 2

# Apply the function to filter the DataFrame
final_filtered_data = filtered_data[filtered_data.apply(check_conditions, axis=1)]
# print(filtered_data.columns)
# Rearrange columns by dimensions: STA, PER, CAD, REL, DID
dimensions = ['STA', 'PER', 'CAD', 'REL', 'DID']
# excluding sex from the sorting
sorted_columns = sorted(
    [col for col in final_filtered_data.columns if col != 'Sex'], 
    key=lambda x: dimensions.index(x[-3:])
)
final_filtered_data = final_filtered_data[sorted_columns + ['Sex']]

# print("final ?",final_filtered_data)

# Define a color mapping for each dimension
dimension_colors = {
    'STA': 'dodgerblue',
    'PER': 'tomato',
    'CAD': 'mediumseagreen',
    'REL': 'gold',
    'DID': 'violet'
}

# Assign colors to columns based on their dimension
colors = [dimension_colors[col[-3:]] for col in sorted_columns]

# Plotting the bar chart of column means
column_means = final_filtered_data.loc[:, final_filtered_data.columns != 'Sex'].mean()

plt.figure(figsize=(14, 8))
column_means.plot(kind='bar', color=colors, alpha=0.8)
plt.title('Mean Values of Each Column by Dimensions', fontsize=18, fontweight='bold')
plt.xlabel('Columns', fontsize=14)
plt.ylabel('Mean Value', fontsize=14)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.tight_layout()
# plt.show()
plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')


# Aggregate mean values by dimension
aggregate_means = {
    dim: final_filtered_data[[col for col in final_filtered_data.columns if col.endswith(dim)]].mean().mean()
    for dim in dimensions
}

# Create a list of colors for the aggregate dimensions
aggregate_colors = [dimension_colors[dim] for dim in dimensions]

# Convert the dictionary to a Series for easy plotting
aggregate_means_series = pd.Series(aggregate_means)

# Plotting the aggregate mean values by dimension
plt.figure(figsize=(10, 6))
aggregate_means_series.plot(kind='bar', color=aggregate_colors, alpha=0.8)
plt.title('Aggregate Mean Values by Dimensions', fontsize=18, fontweight='bold')
plt.xlabel('Dimensions', fontsize=14)
plt.ylabel('Aggregate Mean Value', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.tight_layout()
# plt.show()
plt.savefig('aggregated_bar_chart.png', dpi=300, bbox_inches='tight')

# print("Final:", final_filtered_data.columns)

aggregate_means_male = {
    dim: final_filtered_data[final_filtered_data['Sex'] == 'M'][
        [col for col in final_filtered_data.columns if col.endswith(dim)]
    ].mean().mean()
    for dim in dimensions
}

aggregate_means_female = {
    dim: final_filtered_data[final_filtered_data['Sex'] == 'F'][
        [col for col in final_filtered_data.columns if col.endswith(dim)]
    ].mean().mean()
    for dim in dimensions
}

# Convert the aggregated means into a DataFrame
aggregate_means_df = pd.DataFrame({
    'Male': aggregate_means_male,
    'Female': aggregate_means_female
})

# Plotting the bar chart
aggregate_means_df.plot(kind='bar', figsize=(10, 6), color=['pink', 'blue'], alpha=0.8)
plt.title('Aggregate Mean Values by Dimensions and Sex', fontsize=18, fontweight='bold')
plt.xlabel('Dimensions', fontsize=14)
plt.ylabel('Aggregate Mean Value', fontsize=14)
plt.xticks(rotation=0, fontsize=12)  # Rotate ticks for better label visibility
plt.yticks(fontsize=12)
plt.legend(title='Sex', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.tight_layout()
plt.savefig('aggregate_means_sex_barplot.png', dpi=300, bbox_inches='tight')
# plt.show()


# Radar Chart
# Prepare data for the radar chart
df = pd.DataFrame(list(aggregate_means.items()), columns=['Dimension', 'Mean'])
df['Angle'] = np.linspace(0, 2 * pi, len(df), endpoint=False)

# Append the start value to close the radar chart
df = pd.concat([df, df.iloc[[0]]]).reset_index(drop=True)

fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
ax.plot(df['Angle'], df['Mean'], 'o-', linewidth=2, label='Mean Value')
ax.fill(df['Angle'], df['Mean'], alpha=0.25)

# Set the dimension names as labels
ax.set_xticks(df['Angle'][:-1])
ax.set_xticklabels(df['Dimension'][:-1])

# Annotate values
for i, txt in enumerate(df['Mean']):
    ax.annotate(f"{txt:.2f}", (df['Angle'][i], df['Mean'][i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Mean Comparison of Dimensions', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
# plt.show()
plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')


# Clean data for one-way ANOVA

def clean_dimension(data, keyword):
    values = data.filter(like=keyword).values.flatten()
    numeric_values = pd.to_numeric(values, errors='coerce')
    return numeric_values[~pd.isna(numeric_values)]

STA = clean_dimension(data, "STA")
PER = clean_dimension(data, "PER")
CAD = clean_dimension(data, "CAD")
REL = clean_dimension(data, "REL")
DID = clean_dimension(data, "DID")

# Print number of observations
# print(f"STA: {len(STA)} observations")
# print(f"PER: {len(PER)} observations")
# print(f"CAD: {len(CAD)} observations")
# print(f"REL: {len(REL)} observations")
# print(f"DID: {len(DID)} observations")


# Conduct one-way ANOVA
f_value, p_value = stats.f_oneway(STA, PER, CAD, REL, DID)
# print(f"F-Value: {f_value}, P-Value: {p_value}")

# Detailed ANOVA table using statsmodels
anova_data = pd.DataFrame({
    'Value': np.concatenate([STA, PER, CAD, REL, DID]),
    'Group': (
        ['STA'] * len(STA) +
        ['PER'] * len(PER) +
        ['CAD'] * len(CAD) +
        ['REL'] * len(REL) +
        ['DID'] * len(DID)
    )
})

model = ols('Value ~ Group', data=anova_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA Table
print(anova_table)

# Perform Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=anova_data['Value'], groups=anova_data['Group'], alpha=0.05)
print(tukey.summary())

# Visualize Tukey's HSD results
tukey.plot_simultaneous(figsize=(8, 6))

# Reshape data for two ways ANOVA
anova_data = pd.melt(
    final_filtered_data, 
    id_vars=['Sex'], 
    value_vars=[col for col in final_filtered_data.columns if col != 'Sex'],
    var_name='Dimension', 
    value_name='Value'
)
anova_data.dropna(subset=['Value'], inplace=True)  # Remove NaN values

model = ols('Value ~ C(Sex) + C(Dimension) + C(Sex):C(Dimension)', data=anova_data).fit()
anova_results = anova_lm(model, typ=2)  # Type II ANOVA table
print(anova_results)
