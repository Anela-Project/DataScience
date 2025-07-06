import numpy as np  # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # suppress warnings
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE  # oversampling to handle imbalance
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, plot_confusion_matrix
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\USER\PycharmProjects\recommendation_system\transaction_dataset.csv")
df.head()

print(f"Number of rows in DataFrame is {df.shape[0]}")
print(f"Number of columns in DataFrame is {df.shape[1]}")
print(df.isnull().sum())  # check missing values by column
print(df['FLAG'])  # display target variable

# Drop irrelevant or redundant columns
df.drop(columns=['Unnamed: 0', 'Index', 'Address'],  inplace=True)

# Clean column names by replacing spaces with underscores
df.columns = df.columns.str.strip().str.replace(r'\b\s+\b', '_', regex=True)
numeric = df.select_dtypes(include=['number']).columns
categories = df.select_dtypes(include='object').columns.tolist()

# Identify and drop constant variance features
constant_var = [i for i in numeric if df[i].var() == 0]
print(f"Number of features that have constant value is {len(constant_var)}")
print(constant_var)
df.drop(columns=constant_var, inplace=True)

# Drop highly correlated features to reduce redundancy
drop_cols = [
    'total_ether_sent_contracts',
    'ERC20_max_val_rec',
    'ERC20_avg_val_sent',
    'ERC20_max_val_sent',
    'ERC20_uniq_rec_token_name',
    'ERC20_min_val_sent',
    'avg_value_sent_to_contract',
    'ERC20_avg_val_rec',
    'total_transactions (including_tnx_to_create_contract'
]
df.drop(drop_cols, axis=1, inplace=True)

# Calculate correlation matrix and identify highly correlated pairs
corr_matrix = df.corr(numeric_only=True)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = upper.stack().sort_values(ascending=False)
high_corr = high_corr[high_corr > 0.8]
print("Highly correlated pairs (correlation > 0.8):")
print(high_corr)

# Visualize correlation heatmap to understand relationships between features
plt.figure(figsize=(10, 7))
numeric = df.select_dtypes(include=['number']).columns
corr = df[numeric].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, annot=True, vmin=-1, vmax=1, fmt=".1f")
plt.show()

# Plot pie chart of target class distribution to observe imbalance
pie, ax = plt.subplots(figsize=[15,10])
labels = ['Non-fraud', 'Fraud']
colors = ['#f9ae35', '#f64e38']
plt.pie(x = df['FLAG'].value_counts(), autopct='%.2f%%', explode=[0.02]*2, labels=labels, pctdistance=0.5, textprops={'fontsize': 14}, colors = colors)
plt.title('Target distribution')
plt.show()

# Visualize distributions of selected features with boxplots
b=20
columns = df.columns
fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout =True)
plt.subplots_adjust(wspace = 0.7, hspace=0.8)
plt.suptitle("Distribution of features",y=0.95, size=18, weight='bold')

ax = sns.boxplot(ax = axes[0,0], data=df, x=columns[1])
ax.set_title(f'Distribution of {columns[1]}')

ax1 = sns.boxplot(ax = axes[0,1], data=df, x=columns[2])
ax1.set_title(f'Distribution of {columns[2]}')

ax2 = sns.boxplot(ax = axes[1,0], data=df, x=columns[3])
ax2.set_title(f'Distribution of {columns[3]}')

ax3 = sns.boxplot(ax = axes[1,1], data=df, x=columns[4])
ax3.set_title(f'Distribution of {columns[4]}')
plt.show()

# Check and display rows with missing values
print(f"Number of rows that has at least one missing value: {df.isnull().any(axis=1).sum()}")
missing_mask = df.isnull().any(axis=1)
print(df.loc[missing_mask, 'FLAG'].value_counts())

# Prepare dataset without missing values for modeling
print(df[~missing_mask].shape)
sub_df = df[~missing_mask]
X = sub_df.drop(columns='FLAG', axis=1)
y = sub_df['FLAG']

# Split data into training and test sets with stratification on target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
print(f"Shape of X_train is {X_train.shape}")
print(f"Shape of y_train is {y_train.shape}")
print(f"Shape of X_test is {X_test.shape}")
print(f"Shape of y_test is {y_test.shape}")

# Initialize and apply target encoding to categorical columns
encoder = TargetEncoder(cols=categories)
X_train_encoded = encoder.fit_transform(X_train, y_train)  # fit on train data
X_test_encoded = encoder.transform(X_test)  # transform test data with same encoding

# Use SMOTE to oversample minority class in training set
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_encoded, y_train)

# Display class distribution before and after oversampling
non_fraud = sum(y_train == 0)
fraud = sum(y_train == 1)

no = sum(y_train_resampled == 0)
yes = sum(y_train_resampled == 1)

print(f'BEFORE OVERSAMPLING \n \tNon-frauds: {non_fraud} \n \tFauds: {fraud}')
print(f'AFTER OVERSAMPLING \n \tNon-frauds: {no} \n \tFauds: {yes}')

# Train XGBoost classifier on balanced data
xgb_c = xgb.XGBClassifier(random_state=42)
xgb_c.fit(X_train_resampled, y_train_resampled)

# Predict and evaluate model on test set
predictions_xgb = xgb_c.predict(X_test_encoded)

print(classification_report(y_test, predictions_xgb))
print(confusion_matrix(y_test, predictions_xgb))
plot_confusion_matrix(xgb_c, X_test_encoded, y_test)
