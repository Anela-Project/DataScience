import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE  #oversampling
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, plot_confusion_matrix
import pickle
#IsolationForest is an algorithm from scikit-learn for anomaly detection (detecting outliers or rare events).
#from sklearn.ensemble import IsolationForest
#TSNE stands for t-distributed Stochastic Neighbor Embedding. It’s a technique for dimensionality reduction and data visualization.
#from sklearn.manifold import TSNE


#  _________________________________   READ FILE_________________________    #


df = pd.read_csv(r"C:\Users\USER\PycharmProjects\recommendation_system\transaction_dataset.csv")
df.head()

print(f"Number of rows in DataFrame is {df.shape[0]}")
print(f"Number of columns in DataFrame is {df.shape[1]}")
print(df.isnull().sum())
# Show number of values in each class in percent
df['FLAG'].value_counts(normalize=True) * 100
print(df['FLAG'])



# ___________________PREPROCESS ____________________________________________  #
df.drop(columns=['Unnamed: 0', 'Index', 'Address'],  inplace=True)

df.columns = df.columns.str.strip().str.replace(r'\b\s+\b', '_', regex=True)
numeric = df.select_dtypes(include=['number']).columns
categories = df.select_dtypes(include='object').columns.tolist()

## Columns that have constant values:
constant_var = [i for i in numeric if df[i].var() == 0]
print(f"Number of features that have constant value is {len(constant_var)}")
print(constant_var)
# Drop constant variance features
df.drop(columns=constant_var, inplace=True)
print("emrat e kolonave: " + df.columns)
# Drop any of hgh correlation features
drop_cols = [
    'total_ether_sent_contracts',
    'ERC20_max_val_rec',
    'ERC20_avg_val_sent',
    'ERC20_avg_val_sent',  # same as above, consider dropping once
    'ERC20_max_val_sent',
    'ERC20_max_val_sent',  # duplicate, drop once
    'ERC20_uniq_rec_token_name',
    'ERC20_avg_val_sent',  # repeated again, drop once
    'ERC20_min_val_sent',
    'total_ether_sent_contracts',  # repeated, drop once
    'avg_value_sent_to_contract',
    'ERC20_avg_val_rec',
    'ERC20_avg_val_rec',  # duplicate, drop once
    'total_transactions (including_tnx_to_create_contract'
]
df.drop(drop_cols, axis=1, inplace=True)

corr_matrix = df.corr(numeric_only=True)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr = upper.stack().sort_values(ascending=False)
high_corr = high_corr[high_corr > 0.8]

# Display results
print("Highly correlated pairs (correlation > 0.8):")
print(high_corr)
# -----------------------------PLOTS ______________________________#
plt.figure(figsize=(10, 7))
numeric = df.select_dtypes(include=['number']).columns
corr = df[numeric].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, annot=True, vmin=-1, vmax=1, fmt=".1f")
plt.show()

pie, ax = plt.subplots(figsize=[15,10])
labels = ['Non-fraud', 'Fraud']
colors = ['#f9ae35', '#f64e38']
plt.pie(x = df['FLAG'].value_counts(), autopct='%.2f%%', explode=[0.02]*2, labels=labels, pctdistance=0.5, textprops={'fontsize': 14}, colors = colors)
plt.title('Target distribution')
plt.show()

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








#____________________________ MODELLING ____________________________________#4

#PREPARE TRAIN AND TEST DATASET
print(f"Number of rows that has at least one missing value: {df.isnull().any(axis=1).sum()}")
missing_mask = df.isnull().any(axis=1)
# Split the data into train and test set
print(df.loc[missing_mask, 'FLAG'].value_counts())

print(df[~missing_mask].shape)
sub_df = df[~missing_mask]
X = sub_df.drop(columns='FLAG', axis=1)
y = sub_df['FLAG']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
print(f"Shape of X_train is {X_train.shape}")
print(f"Shape of y_train is {y_train.shape}")
print(f"Shape of X_test is {X_test.shape}")
print(f"Shape of y_test is {y_test.shape}")


# Target encoding
encoder = TargetEncoder(cols=categories)
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)


sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_encoded, y_train)


#Initialize encoder to apply target encoding on selected categorical columns
#Learn category → target mean mapping from training data and transform X_train
#Apply the same mapping to the test data without re-fitting


# Target distribution before SMOTE
non_fraud = 0
fraud = 0

for i in y_train:
    if i == 0:
        non_fraud +=1
    else:
        fraud +=1

# Target distribution after SMOTE
no = 0
yes = 1

for j in y_train_resampled:
    if j == 0:
        no +=1
    else:
        yes +=1


print(f'BEFORE OVERSAMPLING \n \tNon-frauds: {non_fraud} \n \tFauds: {fraud}')
print(f'AFTER OVERSAMPLING \n \tNon-frauds: {no} \n \tFauds: {yes}')


#  ______________________________XGBOOST MODEL ____________________________________
xgb_c = xgb.XGBClassifier(random_state=42)
xgb_c.fit(X_train_resampled, y_train_resampled)
predictions_xgb = xgb_c.predict(X_test_encoded)

print(classification_report(y_test, predictions_xgb))
print(confusion_matrix(y_test, predictions_xgb))
plot_confusion_matrix(xgb_c, X_test_encoded, y_test)
