import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import re
import scikit_posthocs as sp
from scipy.stats import kruskal
from bioinfokit.analys import stat
import pingouin as pg


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


############### Kruskal-Wallis on the five dimensions without other arguments ########


# 1) Kruskal–Wallis on the five dimensions
kw_stat, kw_pvalue = kruskal(STA, PER, CAD, REL, DID)
print(f"Kruskal–Wallis statistic = {kw_stat:.3f}, p-value = {kw_pvalue:.9e}")

# 2) If significant, do Dunn’s post-hoc test
# For Dunn’s, we need a single array of values plus an array of group labels:
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

if kw_pvalue < 0.05:
    print("\nDunn’s Test (post-hoc) because Kruskal–Wallis is significant:")
    dunn_results = sp.posthoc_dunn(
        anova_data, 
        val_col='Value', 
        group_col='Group', 
        p_adjust='bonferroni'  # or 'holm', etc.
    )
    print(dunn_results)
else:
    print("No significant difference found by Kruskal–Wallis (p ≥ 0.05).")


########### Taking Sex into account  ##################


######### Classes #######

def map_classe_to_group(classe_value):
    """
    Returns '9' if the value starts with 9,
            '10' if it starts with 10,
            '11' if it starts with 11,
            np.nan otherwise.
    """
    if pd.notnull(classe_value):
        # Using regex or str.startswith to check the beginning:
        if str(classe_value).startswith("9"):
            return "9"
        elif str(classe_value).startswith("10"):
            return "10"
        elif str(classe_value).startswith("11"):
            return "11"
    return np.nan

final_filtered_data['ClassGroup'] = data['Classe'].apply(map_classe_to_group)

final_filtered_data = final_filtered_data.dropna(subset=['ClassGroup'])

counts = final_filtered_data['ClassGroup'].value_counts()
print("Counts by ClassGroup:")
print(counts)

dimensions = ['STA', 'PER', 'CAD', 'REL', 'DID']

grouped_means = {}  # Dictionary to store means for each subgroup
for group_label, group_df in final_filtered_data.groupby('ClassGroup'):
    # Calculate the mean for each dimension by flattening all columns that end with that dimension
    dim_means = {}
    for dim in dimensions:
        # Filter columns that end with the dimension keyword, then compute their overall mean
        dim_cols = [col for col in group_df.columns if col.endswith(dim)]
        dim_means[dim] = group_df[dim_cols].mean().mean()
    grouped_means[group_label] = dim_means
    
# Convert to a DataFrame for easier inspection / plotting
grouped_means_df = pd.DataFrame(grouped_means)
print("Means of each dimension for each ClassGroup (9, 10, 11):")
print(grouped_means_df)


def melt_subgroup(data, subgroup_label):
    """
    Filters 'data' to only rows with ClassGroup == subgroup_label.
    Gathers all columns ending in each dimension
    into a single long-format DataFrame:
    Columns: ['Value', 'Dimension', 'ClassGroup'].
    """
    # 1) Filter to keep only this group's rows
    subgroup_data = data[data['ClassGroup'] == subgroup_label].copy()
    
    # 2) Create a list of melted DataFrames—one per dimension
    melted_parts = []
    for dim in dimensions:
        # Gather all columns that end with the dimension suffix
        dim_cols = [col for col in subgroup_data.columns if col.endswith(dim)]
        
        # Melt these columns into rows -> (variable, value)
        melted = subgroup_data[dim_cols].melt(var_name='OriginalColumn', value_name='Value')
        
        # Tag the dimension
        melted['Dimension'] = dim
        
        # We only keep the 'Value' and 'Dimension' columns
        # (Though you could keep OriginalColumn if you want to know from which question it came.)
        melted_parts.append(melted[['Value', 'Dimension']])
    
    # 3) Concatenate all dimension data
    melted_df = pd.concat(melted_parts, axis=0, ignore_index=True)
    
    # 4) Remove NaN rows
    melted_df.dropna(subset=['Value'], inplace=True)
    
    # 5) Add the subgroup label
    melted_df['ClassGroup'] = subgroup_label
    
    return melted_df

# final_filtered_data should already have the "ClassGroup" column with values '9', '10', or '11'.
melted_9 = melt_subgroup(final_filtered_data, '9')
melted_10 = melt_subgroup(final_filtered_data, '10')
melted_11 = melt_subgroup(final_filtered_data, '11')


def run_kruskal_and_dunn(melted_df, subgroup_label):
    print(f"\n=== Kruskal–Wallis for subgroup '{subgroup_label}' ===")
    
    # 1) We need to gather 'Value' arrays for each dimension
    #    e.g. all STA values in one list, all PER in another, etc.
    dimension_groups = []
    unique_dims = melted_df['Dimension'].unique()
    
    for dim in unique_dims:
        # Extract all 'Value' entries for that dimension
        vals = melted_df.loc[melted_df['Dimension'] == dim, 'Value']
        dimension_groups.append(vals.dropna())
    
    # 2) Kruskal–Wallis test
    kw_stat, kw_pvalue = kruskal(*dimension_groups)
    print(f"Kruskal–Wallis statistic = {kw_stat:.4f}, p-value = {kw_pvalue:.4e}")
    
    # 3) If significant, do Dunn’s post-hoc test
    if kw_pvalue < 0.05:
        print("\n--- Dunn’s Test (post-hoc) ---")
        # scikit-posthocs has a direct method for Dunn’s test:
        #   posthoc_dunn() expects a DataFrame, plus 'val_col' and 'group_col'
        #   specifying which columns contain numeric values and group labels.

        dunn_result = sp.posthoc_dunn(
            melted_df, 
            val_col='Value',     # Column name for numeric data
            group_col='Dimension',  # Column name for groups
            p_adjust='bonferroni'   # or 'holm', 'fdr_bh', etc.
        )
        print(dunn_result)
    else:
        print("No significant difference found by Kruskal–Wallis (p ≥ 0.05).")
        
run_kruskal_and_dunn(melted_9,  '9')
run_kruskal_and_dunn(melted_10, '10')
run_kruskal_and_dunn(melted_11, '11')

def compute_dimension_means_for_group(data, group_label):
    """
    Given a DataFrame and a class group (e.g. '9'), this function
    computes the aggregate mean of each dimension's columns within that group.
    """
    group_data = data[data['ClassGroup'] == group_label]
    means_dict = {}
    for dim in dimensions:
        dim_cols = [col for col in group_data.columns if col.endswith(dim)]
        if len(dim_cols) > 0:
            means_dict[dim] = group_data[dim_cols].mean().mean()
        else:
            # If no columns found for some reason, default to NaN
            means_dict[dim] = np.nan
    return means_dict

def plot_radar_chart(means_dict, group_label):
    """
    Plots a radar chart for the dimension means in 'means_dict'
    and saves it to a file (e.g. 'radar_chart_9.png').
    """
    # Convert the dictionary to a DataFrame with columns ['Dimension', 'Mean']
    df = pd.DataFrame(list(means_dict.items()), columns=['Dimension', 'Mean'])
    
    # Create an angle for each dimension
    df['Angle'] = np.linspace(0, 2 * pi, len(df), endpoint=False)
    
    # Append the first row at the bottom so the radar is "closed"
    df = pd.concat([df, df.iloc[[0]]]).reset_index(drop=True)
    
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
    
    # Plot the data
    ax.plot(df['Angle'], df['Mean'], 'o-', linewidth=2, label='Mean Value')
    ax.fill(df['Angle'], df['Mean'], alpha=0.25)
    
    # Set dimension labels around the circle (excluding the duplicated last row)
    ax.set_xticks(df['Angle'][:-1])
    ax.set_xticklabels(df['Dimension'][:-1])
    
    # Annotate each point with its numeric value
    for i, txt in enumerate(df['Mean']):
        ax.annotate(f"{txt:.2f}", 
                    (df['Angle'][i], df['Mean'][i]), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    # Title and legend
    plt.title(f"Mean Comparison of Dimensions for ClassGroup {group_label}",
              fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save (and/or show) the figure
    plt.savefig(f'radar_chart_{group_label}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure so it won't pop up repeatedly

# ----------------------------
# Main Execution
# ----------------------------
# Suppose 'final_filtered_data' is your cleaned DataFrame 
# which already has the "ClassGroup" column = ['9','10','11'].

# For each class group, compute dimension means and plot
for grp in ['9', '10', '11']:
    group_means = compute_dimension_means_for_group(final_filtered_data, grp)
    plot_radar_chart(group_means, grp)
    

# 1) Identify all columns ending with STA, PER, CAD, REL, DID
dimension_suffixes = ('STA', 'PER', 'CAD', 'REL', 'DID')
dimension_cols = [
    col for col in final_filtered_data.columns 
    if col.endswith(dimension_suffixes)  # ends with any of the dimension suffixes
]

# 2) Compute the "overall score" as the row-wise mean of these columns.
final_filtered_data['OverallScore'] = final_filtered_data[dimension_cols].mean(axis=1)

# 3) Optional: Inspect the first few rows
#print(final_filtered_data[['ClassGroup', 'OverallScore']])

# Suppose final_filtered_data has columns:
# 'ClassGroup' in ['9','10','11'] and 'OverallScore' (or similar).

group_9  = final_filtered_data.loc[final_filtered_data['ClassGroup'] == '9',  'OverallScore'].dropna()
group_10 = final_filtered_data.loc[final_filtered_data['ClassGroup'] == '10', 'OverallScore'].dropna()
group_11 = final_filtered_data.loc[final_filtered_data['ClassGroup'] == '11', 'OverallScore'].dropna()


# 1) Kruskal–Wallis
kw_stat, kw_pvalue = kruskal(group_9, group_10, group_11)
print(f"Kruskal–Wallis (comparing the 3 subClasses againt each other) = {kw_stat:.3f}, p-value = {kw_pvalue:.4f}.")

# 2) If significant, do Dunn’s
if kw_pvalue < 0.05:
    # We combine all scores into a single series, plus a parallel group label
    all_scores = pd.concat([group_9, group_10, group_11], ignore_index=True)
    labels = (['9'] * len(group_9)
            + ['10'] * len(group_10)
            + ['11'] * len(group_11))
    
    # We can create a small DataFrame to pass into posthoc_dunn
    df_overall = pd.DataFrame({'Score': all_scores, 'Group': labels})
    
    print("\nDunn’s Test Results:")
    dunn_result = sp.posthoc_dunn(
        df_overall, 
        val_col='Score',   # numeric column
        group_col='Group', # group labels
        p_adjust='bonferroni'
    )
    print(dunn_result)
else:
    print("No significant difference found among '9','10','11' by Kruskal–Wallis.")
    
# After the ClassGroup analysis section, add this code

print("\n===============================================")
print("ANALYZING SEX DIFFERENCES ACROSS DIMENSIONS")
print("===============================================")

# First, confirm we have data for both sexes
sex_counts = final_filtered_data['Sex'].value_counts()
print("Sample sizes by sex:")
print(sex_counts)

# Since we've already calculated aggregate means by sex earlier in the code
# (aggregate_means_male and aggregate_means_female), we can display them here
print("\nAggregate dimension means by sex:")
print(aggregate_means_df)

# Create a function to melt data by sex, similar to the ClassGroup approach
def melt_by_sex(data, sex_value):
    """
    Filters 'data' to only rows with Sex == sex_value.
    Gathers all columns ending in each dimension into a single long-format DataFrame.
    Returns: DataFrame with columns ['Value', 'Dimension', 'Sex'].
    """
    sex_data = data[data['Sex'] == sex_value].copy()
    
    melted_parts = []
    for dim in dimensions:
        dim_cols = [col for col in sex_data.columns if col.endswith(dim)]
        melted = sex_data[dim_cols].melt(var_name='OriginalColumn', value_name='Value')
        melted['Dimension'] = dim
        melted_parts.append(melted[['Value', 'Dimension']])
    
    melted_df = pd.concat(melted_parts, axis=0, ignore_index=True)
    melted_df.dropna(subset=['Value'], inplace=True)
    melted_df['Sex'] = sex_value
    
    return melted_df

# Melt data for each sex
melted_male = melt_by_sex(final_filtered_data, 'M')
melted_female = melt_by_sex(final_filtered_data, 'F')

# Analyze dimensions separately for each sex
print("\n=== Kruskal-Wallis Test for Males ===")
run_kruskal_and_dunn(melted_male, 'Male')

print("\n=== Kruskal-Wallis Test for Females ===")
run_kruskal_and_dunn(melted_female, 'Female')

# Now test if sex is a significant factor within each dimension
print("\n=== Testing Sex Effect Within Each Dimension ===")

# First, combine the melted data
all_melted = pd.concat([melted_male, melted_female], axis=0, ignore_index=True)

for dim in dimensions:
    print(f"\nDimension: {dim}")
    
    # Filter the melted data for just this dimension
    dim_data = all_melted[all_melted['Dimension'] == dim]
    
    # Split by sex
    male_values = dim_data[dim_data['Sex'] == 'M']['Value'].values
    female_values = dim_data[dim_data['Sex'] == 'F']['Value'].values
    
    # Use Mann-Whitney U test (non-parametric alternative to t-test)
    # This tests if the distribution of values differs between sexes
    u_stat, p_value = stats.mannwhitneyu(male_values, female_values, alternative='two-sided')
    
    print(f"Mann-Whitney U test: U = {u_stat:.4f}, p-value = {p_value:.4e}")
    
    if p_value < 0.05:
        print(f"Significant difference between sexes for {dim} (p < 0.05)")
        # Add male and female means for context
        male_mean = dim_data[dim_data['Sex'] == 'M']['Value'].mean()
        female_mean = dim_data[dim_data['Sex'] == 'F']['Value'].mean()
        print(f"Male mean: {male_mean:.3f}, Female mean: {female_mean:.3f}")
    else:
        print(f"No significant difference between sexes for {dim}")

# Interactive visual: radar chart comparing males and females
print("\n=== Creating Sex Comparison Radar Chart ===")

# Convert the dictionaries to DataFrames for plotting
male_df = pd.DataFrame(list(aggregate_means_male.items()), columns=['Dimension', 'Male_Mean'])
female_df = pd.DataFrame(list(aggregate_means_female.items()), columns=['Dimension', 'Female_Mean'])

# Merge the two DataFrames
radar_df = pd.merge(male_df, female_df, on='Dimension')

# Add angles for radar chart
radar_df['Angle'] = np.linspace(0, 2 * pi, len(radar_df), endpoint=False)

# Append the start value to close the radar chart
radar_df = pd.concat([radar_df, radar_df.iloc[[0]]]).reset_index(drop=True)

# Create the radar chart for sex comparison
fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 8))

# Plot male line
ax.plot(radar_df['Angle'], radar_df['Male_Mean'], 'o-', linewidth=2, color='pink', label='Male')
ax.fill(radar_df['Angle'], radar_df['Male_Mean'], color='blue', alpha=0.1)

# Plot female line
ax.plot(radar_df['Angle'], radar_df['Female_Mean'], 'o-', linewidth=2, color='blue', label='Female')
ax.fill(radar_df['Angle'], radar_df['Female_Mean'], color='pink', alpha=0.1)

# Set the dimension names as labels
ax.set_xticks(radar_df['Angle'][:-1])
ax.set_xticklabels(radar_df['Dimension'][:-1], fontsize=12)

# Annotate values
for i in range(len(radar_df) - 1):  # Skip the last duplicate point
    # Male values
    ax.annotate(
        f"{radar_df['Male_Mean'][i]:.2f}",
        (radar_df['Angle'][i], radar_df['Male_Mean'][i]), 
        textcoords="offset points", 
        xytext=(0, 10), 
        ha='center',
        color='blue'
    )
    # Female values
    ax.annotate(
        f"{radar_df['Female_Mean'][i]:.2f}",
        (radar_df['Angle'][i], radar_df['Female_Mean'][i]), 
        textcoords="offset points", 
        xytext=(0, -15), 
        ha='center',
        color='pink'
    )

plt.title('Comparison of Dimensions by Sex', fontsize=18, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig('radar_chart_sex_comparison.png', dpi=300, bbox_inches='tight')

# Finally, test if both sex and class groups interact with dimensions
# This is a more complex analysis that combines both factors
print("\n=== Testing Interaction Between Sex, Class Groups, and Dimensions ===")

# Create melted DataFrame with both Sex and ClassGroup
def melt_with_both_factors(data):
    """
    Creates a melted DataFrame with Sex, ClassGroup and Dimension information.
    Returns: DataFrame with columns ['Value', 'Dimension', 'Sex', 'ClassGroup']
    """
    melted_parts = []
    for dim in dimensions:
        dim_cols = [col for col in data.columns if col.endswith(dim)]
        subset = data[dim_cols + ['Sex', 'ClassGroup']].copy()
        melted = subset.melt(id_vars=['Sex', 'ClassGroup'], 
                           var_name='OriginalColumn', 
                           value_name='Value')
        melted['Dimension'] = dim
        melted_parts.append(melted[['Value', 'Dimension', 'Sex', 'ClassGroup']])
    
    melted_df = pd.concat(melted_parts, axis=0, ignore_index=True)
    melted_df.dropna(subset=['Value'], inplace=True)
    
    return melted_df

# Create the combined melted DataFrame
full_melted = melt_with_both_factors(final_filtered_data)

# Generate a summary table of means for all combinations
print("\n=== Summary Table: Mean Values by Dimension, Sex, and Class ===")
summary_table = full_melted.groupby(['Sex', 'ClassGroup', 'Dimension'])['Value'].agg(['mean', 'count']).reset_index()
print(summary_table)

# Test for class group effects within each dimension
print("\n===============================================")
print("ANALYZING CLASS GROUP DIFFERENCES WITHIN EACH DIMENSION")
print("===============================================")

# First, ensure we have data for the different class groups
class_counts = final_filtered_data['ClassGroup'].value_counts()
print("Sample sizes by class group:")
print(class_counts)

print("\n=== Testing Class Group Effect Within Each Dimension ===")

for dim in dimensions:
    print(f"\nDimension: {dim}")
    
    # Filter the full melted data for just this dimension
    dim_data = full_melted[full_melted['Dimension'] == dim]
    
    # Split by class group
    class_9_values = dim_data[dim_data['ClassGroup'] == '9']['Value'].values
    class_10_values = dim_data[dim_data['ClassGroup'] == '10']['Value'].values
    class_11_values = dim_data[dim_data['ClassGroup'] == '11']['Value'].values
    
    # Check if we have sufficient data in each group
    if len(class_9_values) > 0 and len(class_10_values) > 0 and len(class_11_values) > 0:
        # Use Kruskal-Wallis test (non-parametric ANOVA)
        kw_stat, kw_pvalue = kruskal(class_9_values, class_10_values, class_11_values)
        
        print(f"Kruskal-Wallis test: H = {kw_stat:.4f}, p-value = {kw_pvalue:.4e}")
        
        if kw_pvalue < 0.05:
            print(f"Significant difference between class groups for {dim} (p < 0.05)")
            
            # Create a DataFrame for Dunn's post-hoc test
            class_df = pd.DataFrame({
                'Value': np.concatenate([class_9_values, class_10_values, class_11_values]),
                'ClassGroup': ['9'] * len(class_9_values) + ['10'] * len(class_10_values) + ['11'] * len(class_11_values)
            })
            
            # Run Dunn's post-hoc test
            print("\nDunn's post-hoc test:")
            dunn_result = sp.posthoc_dunn(
                class_df,
                val_col='Value',
                group_col='ClassGroup',
                p_adjust='bonferroni'
            )
            print(dunn_result)
            
            # Add class means for context
            class_9_mean = dim_data[dim_data['ClassGroup'] == '9']['Value'].mean()
            class_10_mean = dim_data[dim_data['ClassGroup'] == '10']['Value'].mean()
            class_11_mean = dim_data[dim_data['ClassGroup'] == '11']['Value'].mean()
            print(f"Class 9 mean: {class_9_mean:.3f}, Class 10 mean: {class_10_mean:.3f}, Class 11 mean: {class_11_mean:.3f}")
        else:
            print(f"No significant difference between class groups for {dim}")
    else:
        print(f"Insufficient data for some class groups in {dim} dimension.")

# Create a visualization to compare dimensions across class groups
print("\n=== Creating Bar Chart of Dimensions by Class Groups ===")

# Calculate means for each dimension by class group
dim_class_means = full_melted.groupby(['Dimension', 'ClassGroup'])['Value'].mean().unstack()
print("Mean values by dimension and class group:")
print(dim_class_means)

# Plot this as a grouped bar chart
plt.figure(figsize=(12, 8))
dim_class_means.plot(kind='bar', figsize=(12, 8))
plt.title('Mean Dimension Scores by Class Group', fontsize=18, fontweight='bold')
plt.xlabel('Dimension', fontsize=14)
plt.ylabel('Mean Score', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Class Group', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig('dimension_by_class_comparison.png', dpi=300, bbox_inches='tight')

# Create a radar chart comparing class groups (9, 10, 11)
print("\n=== Creating Class Group Comparison Radar Chart ===")

# First, create dictionaries of aggregate means by class group
aggregate_means_9 = {
    dim: final_filtered_data[final_filtered_data['ClassGroup'] == '9'][
        [col for col in final_filtered_data.columns if col.endswith(dim)]
    ].mean().mean()
    for dim in dimensions
}

aggregate_means_10 = {
    dim: final_filtered_data[final_filtered_data['ClassGroup'] == '10'][
        [col for col in final_filtered_data.columns if col.endswith(dim)]
    ].mean().mean()
    for dim in dimensions
}

aggregate_means_11 = {
    dim: final_filtered_data[final_filtered_data['ClassGroup'] == '11'][
        [col for col in final_filtered_data.columns if col.endswith(dim)]
    ].mean().mean()
    for dim in dimensions
}

# Convert the dictionaries to DataFrames for plotting
class_9_df = pd.DataFrame(list(aggregate_means_9.items()), columns=['Dimension', 'Class_9_Mean'])
class_10_df = pd.DataFrame(list(aggregate_means_10.items()), columns=['Dimension', 'Class_10_Mean'])
class_11_df = pd.DataFrame(list(aggregate_means_11.items()), columns=['Dimension', 'Class_11_Mean'])

# Merge the DataFrames
radar_df_class = class_9_df.merge(class_10_df, on='Dimension').merge(class_11_df, on='Dimension')

# Add angles for radar chart
radar_df_class['Angle'] = np.linspace(0, 2 * pi, len(radar_df_class), endpoint=False)

# Append the start value to close the radar chart
radar_df_class = pd.concat([radar_df_class, radar_df_class.iloc[[0]]]).reset_index(drop=True)

# Create the radar chart for class group comparison
fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 8))

# Plot Class 9 line
ax.plot(radar_df_class['Angle'], radar_df_class['Class_9_Mean'], 'o-', linewidth=2, color='blue', label='Class 9')
ax.fill(radar_df_class['Angle'], radar_df_class['Class_9_Mean'], color='blue', alpha=0.1)

# Plot Class 10 line
ax.plot(radar_df_class['Angle'], radar_df_class['Class_10_Mean'], 'o-', linewidth=2, color='red', label='Class 10')
ax.fill(radar_df_class['Angle'], radar_df_class['Class_10_Mean'], color='red', alpha=0.1)

# Plot Class 11 line
ax.plot(radar_df_class['Angle'], radar_df_class['Class_11_Mean'], 'o-', linewidth=2, color='green', label='Class 11')
ax.fill(radar_df_class['Angle'], radar_df_class['Class_11_Mean'], color='green', alpha=0.1)

# Set the dimension names as labels
ax.set_xticks(radar_df_class['Angle'][:-1])
ax.set_xticklabels(radar_df_class['Dimension'][:-1], fontsize=12)

# Annotate values
for i in range(len(radar_df_class) - 1):  # Skip the last duplicate point
    # Class 9 values
    ax.annotate(
        f"{radar_df_class['Class_9_Mean'][i]:.2f}",
        (radar_df_class['Angle'][i], radar_df_class['Class_9_Mean'][i]), 
        textcoords="offset points", 
        xytext=(0, 10), 
        ha='center',
        color='blue'
    )
    
    # Class 10 values - position these slightly offset
    ax.annotate(
        f"{radar_df_class['Class_10_Mean'][i]:.2f}",
        (radar_df_class['Angle'][i], radar_df_class['Class_10_Mean'][i]), 
        textcoords="offset points", 
        xytext=(15, 0), 
        ha='left',
        color='red'
    )
    
    # Class 11 values - position these slightly offset in another direction
    ax.annotate(
        f"{radar_df_class['Class_11_Mean'][i]:.2f}",
        (radar_df_class['Angle'][i], radar_df_class['Class_11_Mean'][i]), 
        textcoords="offset points", 
        xytext=(-15, 0), 
        ha='right',
        color='green'
    )

plt.title('Comparison of Dimensions by Age Group', fontsize=18, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig('radar_chart_class_comparison.png', dpi=300, bbox_inches='tight')

# Add a summary table of dimension means by class group
print("\n=== Summary of Dimension Means by Age Group ===")
class_means_df = pd.DataFrame({
    'Class 9': pd.Series(aggregate_means_9),
    'Class 10': pd.Series(aggregate_means_10),
    'Class 11': pd.Series(aggregate_means_11)
})
print(class_means_df)


# Test for education track effects (VG vs VP) within each dimension
print("\n===============================================")
print("ANALYZING VG/VP TRACK DIFFERENCES WITHIN EACH DIMENSION")
print("===============================================")

# First, create a function to identify the education track (G or P)
def identify_track(classe_str):
    """
    Identifies whether a student is in the G track or P track based on the Classe field.
    Returns 'G' if 'G' is in the string, 'P' if 'P' is in the string, and None otherwise.
    """
    if not isinstance(classe_str, str):
        return None
        
    if 'G' in classe_str:
        return 'G'
    elif 'P' in classe_str:
        return 'P'
    else:
        return None

# Add education track column to final_filtered_data
final_filtered_data['Track'] = data['Classe'].apply(identify_track)

# Count samples in each track
track_counts = final_filtered_data['Track'].value_counts()
print("Sample sizes by education track:")
print(track_counts)

# Create melted data for dimension-by-track analysis
def melt_by_track(data):
    """
    Creates a melted DataFrame with Track and Dimension information.
    Returns: DataFrame with columns ['Value', 'Dimension', 'Track']
    """
    melted_parts = []
    
    for dim in dimensions:
        dim_cols = [col for col in data.columns if col.endswith(dim)]
        subset = data[dim_cols + ['Track']].copy()
        
        # Only use rows with valid track information
        subset = subset[subset['Track'].notna()]
        
        melted = subset.melt(id_vars=['Track'], 
                           var_name='OriginalColumn', 
                           value_name='Value')
        melted['Dimension'] = dim
        melted_parts.append(melted[['Value', 'Dimension', 'Track']])
    
    melted_df = pd.concat(melted_parts, axis=0, ignore_index=True)
    melted_df.dropna(subset=['Value'], inplace=True)
    
    return melted_df

# Create the melted DataFrame with track information
track_melted = melt_by_track(final_filtered_data)

print("\n=== Testing Track Effect Within Each Dimension ===")

for dim in dimensions:
    print(f"\nDimension: {dim}")
    
    # Filter for this dimension
    dim_data = track_melted[track_melted['Dimension'] == dim]
    
    # Split by track
    g_track_values = dim_data[dim_data['Track'] == 'G']['Value'].values
    p_track_values = dim_data[dim_data['Track'] == 'P']['Value'].values
    
    # Check if we have sufficient data
    if len(g_track_values) > 0 and len(p_track_values) > 0:
        # Use Mann-Whitney U test (non-parametric alternative to t-test)
        u_stat, p_value = stats.mannwhitneyu(g_track_values, p_track_values, alternative='two-sided')
        
        print(f"Mann-Whitney U test: U = {u_stat:.4f}, p-value = {p_value:.4e}")
        
        if p_value < 0.05:
            print(f"Significant difference between tracks for {dim} (p < 0.05)")
            # Add track means for context
            g_mean = dim_data[dim_data['Track'] == 'G']['Value'].mean()
            p_mean = dim_data[dim_data['Track'] == 'P']['Value'].mean()
            print(f"G track mean: {g_mean:.3f}, P track mean: {p_mean:.3f}")
        else:
            print(f"No significant difference between tracks for {dim}")
    else:
        print(f"Insufficient data for comparison in {dim}")

# Create dictionary of aggregate means by track
aggregate_means_g = {
    dim: final_filtered_data[final_filtered_data['Track'] == 'G'][
        [col for col in final_filtered_data.columns if col.endswith(dim)]
    ].mean().mean()
    for dim in dimensions
}

aggregate_means_p = {
    dim: final_filtered_data[final_filtered_data['Track'] == 'P'][
        [col for col in final_filtered_data.columns if col.endswith(dim)]
    ].mean().mean()
    for dim in dimensions
}

# Create a visualization to compare the tracks
print("\n=== Creating Track Comparison Radar Chart ===")

# Convert the dictionaries to DataFrames for plotting
g_df = pd.DataFrame(list(aggregate_means_g.items()), columns=['Dimension', 'G_Track_Mean'])
p_df = pd.DataFrame(list(aggregate_means_p.items()), columns=['Dimension', 'P_Track_Mean'])

# Merge the two DataFrames
radar_df_track = pd.merge(g_df, p_df, on='Dimension')

# Add angles for radar chart
radar_df_track['Angle'] = np.linspace(0, 2 * pi, len(radar_df_track), endpoint=False)

# Append the start value to close the radar chart
radar_df_track = pd.concat([radar_df_track, radar_df_track.iloc[[0]]]).reset_index(drop=True)

# Create the radar chart for track comparison
fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 8))

# Plot G track line
ax.plot(radar_df_track['Angle'], radar_df_track['G_Track_Mean'], 'o-', linewidth=2, color='green', label='VG')
ax.fill(radar_df_track['Angle'], radar_df_track['G_Track_Mean'], color='green', alpha=0.1)

# Plot P track line
ax.plot(radar_df_track['Angle'], radar_df_track['P_Track_Mean'], 'o-', linewidth=2, color='purple', label='VP')
ax.fill(radar_df_track['Angle'], radar_df_track['P_Track_Mean'], color='purple', alpha=0.1)

# Set the dimension names as labels
ax.set_xticks(radar_df_track['Angle'][:-1])
ax.set_xticklabels(radar_df_track['Dimension'][:-1], fontsize=12)

# Annotate values
for i in range(len(radar_df_track) - 1):
    # G track values
    ax.annotate(
        f"{radar_df_track['G_Track_Mean'][i]:.2f}",
        (radar_df_track['Angle'][i], radar_df_track['G_Track_Mean'][i]), 
        textcoords="offset points", 
        xytext=(0, 10), 
        ha='center',
        color='green'
    )
    # P track values
    ax.annotate(
        f"{radar_df_track['P_Track_Mean'][i]:.2f}",
        (radar_df_track['Angle'][i], radar_df_track['P_Track_Mean'][i]), 
        textcoords="offset points", 
        xytext=(0, -15), 
        ha='center',
        color='purple'
    )

plt.title('Comparison of Dimensions by Education Track (VG vs VP)', fontsize=18, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig('radar_chart_track_comparison.png', dpi=300, bbox_inches='tight')

# Add a summary table of dimension means by track
print("\n=== Summary of Dimension Means by Education Track ===")
track_means_df = pd.DataFrame({
    'G Track': pd.Series(aggregate_means_g),
    'P Track': pd.Series(aggregate_means_p)
})
print(track_means_df)