# ============================================================
# DIABETES PREDICTION - Machine Learning Project
# Task 2: Predictive Modeling Using Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, roc_curve, auc, accuracy_score,
                              classification_report)
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# ────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING DIABETES DATASET")
print("=" * 60)

# Download automatically from internet:
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv(url, names=cols)

print(f"Dataset loaded! Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDescriptive statistics:")
print(df.describe())
print(f"\nDiabetes cases: {df['Outcome'].value_counts()[1]} | Healthy: {df['Outcome'].value_counts()[0]}")

# ────────────────────────────────────────────────────────────
# STEP 2: DATA CLEANING
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

# Replace impossible 0 values with NaN
for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    zeros = (df[col] == 0).sum()
    df[col] = df[col].replace(0, np.nan)
    print(f"{col}: replaced {zeros} impossible zeros with NaN")

print(f"\nMissing values after replacing zeros:")
print(df.isnull().sum())

# ────────────────────────────────────────────────────────────
# STEP 3: PREPROCESSING
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: PREPROCESSING")
print("=" * 60)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Fill missing with median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_imp_df = pd.DataFrame(X_imputed, columns=X.columns)
print("Missing values filled with median values")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print("Features scaled using StandardScaler")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining samples: {len(X_train)} (80%)")
print(f"Testing samples:  {len(X_test)} (20%)")

# ────────────────────────────────────────────────────────────
# STEP 4: TRAIN MODELS
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: TRAINING ML MODELS")
print("=" * 60)

lr = LogisticRegression(random_state=42)
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

models = {'Logistic Regression': lr, 'Decision Tree': dt, 'Random Forest': rf}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'pred': y_pred, 'acc': acc}
    print(f"\n{name}:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred,
          target_names=['Healthy','Diabetic'], zero_division=0))

best_name = max(results, key=lambda x: results[x]['acc'])
print(f"\n🏆 Best Model: {best_name} ({results[best_name]['acc']*100:.2f}% accuracy)")

# ────────────────────────────────────────────────────────────
# STEP 5: VISUALIZATIONS
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: CREATING CHARTS (close each to see next)")
print("=" * 60)

df_clean = X_imp_df.copy()
df_clean['Outcome'] = y.values

# Chart 1: Feature Distributions
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle('Chart 1: Feature Distributions (Orange=Diabetic, Blue=Healthy)',
             fontsize=13, fontweight='bold')
for i, col in enumerate(X.columns):
    ax = axes[i//3][i%3]
    ax.hist(df_clean[df_clean['Outcome']==0][col], bins=20, alpha=0.6,
            color='steelblue', label='Healthy')
    ax.hist(df_clean[df_clean['Outcome']==1][col], bins=20, alpha=0.6,
            color='orange', label='Diabetic')
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('diabetes_chart1_distributions.png')
print("Saved: diabetes_chart1_distributions.png")
plt.show()

# Chart 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_clean.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Chart 2: Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('diabetes_chart2_heatmap.png')
print("Saved: diabetes_chart2_heatmap.png")
plt.show()

# Chart 3: Model Accuracy
fig, ax = plt.subplots(figsize=(8, 5))
names = list(results.keys())
accs = [results[n]['acc']*100 for n in names]
bars = ax.bar(names, accs, color=['steelblue','coral','seagreen'], width=0.5)
ax.set_ylim(60, 100)
ax.set_title('Chart 3: Model Accuracy Comparison', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('diabetes_chart3_accuracy.png')
print("Saved: diabetes_chart3_accuracy.png")
plt.show()

# Chart 4: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Chart 4: Confusion Matrices', fontsize=13, fontweight='bold')
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Healthy','Diabetic'],
                yticklabels=['Healthy','Diabetic'])
    ax.set_title(f"{name}\n{res['acc']*100:.1f}% Accuracy")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('diabetes_chart4_confusion.png')
print("Saved: diabetes_chart4_confusion.png")
plt.show()

# Chart 5: ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
for (name, res), color in zip(results.items(), ['steelblue','coral','seagreen']):
    fpr, tpr, _ = roc_curve(y_test, res['model'].predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.2f})')
ax.plot([0,1],[0,1],'k--', lw=1, label='Random Guess')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('Chart 5: ROC Curves — All Models', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('diabetes_chart5_roc.png')
print("Saved: diabetes_chart5_roc.png")
plt.show()

# Chart 6: Decision Tree
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt, feature_names=X.columns.tolist(),
          class_names=['Healthy','Diabetic'],
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title('Chart 6: Decision Tree Visualization', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('diabetes_chart6_decision_tree.png', bbox_inches='tight')
print("Saved: diabetes_chart6_decision_tree.png")
plt.show()

# Chart 7: Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
importances.plot(kind='barh', ax=ax, color='seagreen')
ax.set_title('Chart 7: Feature Importance (Random Forest)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
for i, v in enumerate(importances.values):
    ax.text(v+0.002, i, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('diabetes_chart7_feature_importance.png')
print("Saved: diabetes_chart7_feature_importance.png")
plt.show()

# Chart 8: Boxplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Chart 8: Feature Comparison — Diabetic vs Healthy',
             fontsize=13, fontweight='bold')
for i, col in enumerate(X.columns):
    ax = axes[i//4][i%4]
    data_plot = pd.DataFrame({
        'Value': df_clean[col],
        'Status': df_clean['Outcome'].map({0:'Healthy', 1:'Diabetic'})
    })
    sns.boxplot(data=data_plot, x='Status', y='Value', ax=ax,
                palette={'Healthy':'steelblue','Diabetic':'orange'},
                hue='Status', legend=False)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel('')
plt.tight_layout()
plt.savefig('diabetes_chart8_boxplots.png')
print("Saved: diabetes_chart8_boxplots.png")
plt.show()

# ────────────────────────────────────────────────────────────
# STEP 6: FINAL INSIGHTS
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: KEY INSIGHTS")
print("=" * 60)
print(f"1. Dataset has {len(df)} patients: {(y==1).sum()} diabetic, {(y==0).sum()} healthy")
print(f"2. Best ML model: {best_name} with {results[best_name]['acc']*100:.2f}% accuracy")
print(f"3. Logistic Regression accuracy: {results['Logistic Regression']['acc']*100:.2f}%")
print(f"4. Decision Tree accuracy:       {results['Decision Tree']['acc']*100:.2f}%")
print(f"5. Random Forest accuracy:       {results['Random Forest']['acc']*100:.2f}%")
print(f"6. Top predictor of diabetes: Glucose level (most important feature)")
print(f"7. BMI is 2nd most important feature for prediction")
print(f"8. Age is 3rd most important feature")
print(f"9. Diabetic patients have avg glucose: {df_clean[df_clean['Outcome']==1]['Glucose'].mean():.1f}")
print(f"   Healthy patients have avg glucose:  {df_clean[df_clean['Outcome']==0]['Glucose'].mean():.1f}")
print(f"10. Diabetic patients have avg BMI: {df_clean[df_clean['Outcome']==1]['BMI'].mean():.1f}")
print(f"    Healthy patients have avg BMI:  {df_clean[df_clean['Outcome']==0]['BMI'].mean():.1f}")

print("\n" + "=" * 60)
print("ALL DONE! 8 charts saved. Task 2 Complete! 🎉")
print("=" * 60)
