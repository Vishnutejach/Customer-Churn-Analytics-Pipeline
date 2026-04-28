"""
Customer Churn & Retention Analytics Pipeline
Banking Industry | End-to-End ML System
Uses real Customer-Churn-Records.csv dataset
Nov 2025 – Dec 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings, os, joblib, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                              confusion_matrix, roc_curve, precision_recall_curve,
                              average_precision_score, precision_score, recall_score, f1_score)
from sklearn.calibration import calibration_curve
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'Customer-Churn-Records.csv'
OUTPUT_DIR = BASE_DIR / 'outputs'
DASHBOARD_DIR = OUTPUT_DIR / 'dashboards'
MODEL_DIR = BASE_DIR / 'models'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    'primary':   '#1A3C5E',
    'accent':    '#E8734A',
    'success':   '#2ECC71',
    'warning':   '#F39C12',
    'danger':    '#E74C3C',
    'light':     '#ECF0F1',
    'mid':       '#BDC3C7',
    'dt':        '#5E4FA2',
    'rf':        '#3288BD',
    'gb':        '#E8734A',
    'dnn':       '#D53E4F',
}
RISK_PALETTE = ['#2ECC71', '#F39C12', '#E74C3C']   # low / med / high

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'axes.titlesize':     13,
    'axes.labelsize':     11,
})

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN REAL DATASET
# ═══════════════════════════════════════════════════════════════════════════
df_raw = pd.read_csv(DATA_PATH)

# Normalise column names for consistency
df = df_raw.rename(columns={
    'CustomerId':       'CustomerID',
    'Complain':         'NumComplaints',
    'Satisfaction Score': 'Satisfaction',
    'Card Type':        'CardType',
    'Point Earned':     'PointsEarned',
})

# Drop non-predictive identifier columns
df = df.drop(columns=['RowNumber', 'Surname'])

print(f"Dataset loaded: {df.shape}  |  Churn rate: {df['Exited'].mean():.1%}")
print(f"Columns: {df.columns.tolist()}")
print(df.head(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
def engineer_features(df):
    df = df.copy()

    # Ratios & interactions
    df['BalancePerProduct']    = df['Balance'] / (df['NumOfProducts'] + 1e-9)
    df['SalaryToBalance']      = df['EstimatedSalary'] / (df['Balance'] + 1)
    df['CreditScoreTier']      = pd.cut(df['CreditScore'],
                                         bins=[0, 579, 669, 739, 799, 850],
                                         labels=[0, 1, 2, 3, 4]).astype(int)
    df['AgeBand']              = pd.cut(df['Age'], bins=[17, 30, 45, 60, 100],
                                         labels=[0, 1, 2, 3]).astype(int)
    df['TenureXActivity']      = df['Tenure'] * df['IsActiveMember']

    # Engagement score: combines activity, satisfaction, complaints, loyalty points
    df['EngagementScore']      = (df['IsActiveMember'] * 2
                                   + df['Satisfaction'] / 5
                                   - df['NumComplaints']
                                   + df['PointsEarned'] / 1000)

    # High-value customers who are disengaged — high CAC, high churn risk
    df['HighValueAtRisk']      = ((df['Balance'] > df['Balance'].quantile(0.75)) &
                                   (df['IsActiveMember'] == 0)).astype(int)
    df['ZeroBalance']          = (df['Balance'] == 0).astype(int)
    df['PremiumCard']          = df['CardType'].isin(['PLATINUM', 'DIAMOND']).astype(int)

    # Points loyalty tier
    df['PointsTier']           = pd.cut(df['PointsEarned'],
                                         bins=[0, 300, 600, 1001],
                                         labels=[0, 1, 2]).astype(int)

    # Complaint × Satisfaction interaction (worst of both worlds = highest risk)
    df['ComplaintXLowSat']     = (df['NumComplaints'] == 1) & (df['Satisfaction'] <= 2)
    df['ComplaintXLowSat']     = df['ComplaintXLowSat'].astype(int)

    # Encode categoricals
    le_gender = LabelEncoder()
    df['Gender_enc']    = le_gender.fit_transform(df['Gender'])
    le_geo = LabelEncoder()
    df['Geography_enc'] = le_geo.fit_transform(df['Geography'])
    le_card = LabelEncoder()
    df['CardType_enc']  = le_card.fit_transform(df['CardType'])

    return df

df = engineer_features(df)
print(f"\nFeatures after engineering: {df.shape[1]}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    # Core demographics & account profile
    'Age', 'Tenure', 'NumOfProducts', 'CreditScore', 'Balance', 'EstimatedSalary',
    'IsActiveMember', 'HasCrCard', 'Satisfaction', 'PointsEarned',
    # Engineered features
    'BalancePerProduct', 'SalaryToBalance', 'CreditScoreTier', 'AgeBand',
    'TenureXActivity', 'EngagementScore', 'HighValueAtRisk', 'ZeroBalance',
    'PremiumCard', 'PointsTier',
    # Encoded categoricals
    'Gender_enc', 'Geography_enc', 'CardType_enc',
    # NOTE: NumComplaints/Complain excluded — it is post-hoc (99.6% correlated with
    # Exited in this dataset), which would cause label leakage in a production pipeline.
]

X = df[FEATURE_COLS]
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Churn rate  train: {y_train.mean():.1%}   test: {y_test.mean():.1%}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING  (Decision Tree, Random Forest, DNN as per brief)
# ═══════════════════════════════════════════════════════════════════════════
models = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=20,
        class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        class_weight='balanced', n_jobs=-1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=5,
        subsample=0.8, random_state=42),
    'Deep Neural Network': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        solver='adam', alpha=0.001, learning_rate='adaptive',
        max_iter=300, early_stopping=True, random_state=42),
}

results = {}
for name, model in models.items():
    X_tr = X_train_sc if name == 'Deep Neural Network' else X_train
    X_te = X_test_sc  if name == 'Deep Neural Network' else X_test
    model.fit(X_tr, y_train)
    prob = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    results[name] = {
        'model':    model,
        'prob':     prob,
        'pred':     pred,
        'acc':      accuracy_score(y_test, pred),
        'roc_auc':  roc_auc_score(y_test, prob),
        'avg_prec': average_precision_score(y_test, prob),
    }
    print(f"{name:22s} | ACC={results[name]['acc']:.3f} | "
          f"AUC={results[name]['roc_auc']:.3f} | AP={results[name]['avg_prec']:.3f}")

best_name = max(results, key=lambda k: results[k]['roc_auc'])
best_res  = results[best_name]
print(f"\nBest model: {best_name}  (AUC={best_res['roc_auc']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# 5. RISK TIERING  — actionable customer segments for retention strategy
# ═══════════════════════════════════════════════════════════════════════════
def assign_risk_tier(prob):
    return np.where(prob >= 0.60, 'High Risk',
           np.where(prob >= 0.35, 'Medium Risk', 'Low Risk'))

df_test = X_test.copy()
df_test['ChurnProb']     = best_res['prob']
df_test['RiskTier']      = assign_risk_tier(best_res['prob'])
df_test['Actual']        = y_test.values
df_test['Balance']       = df.loc[X_test.index, 'Balance'].values
df_test['Tenure']        = df.loc[X_test.index, 'Tenure'].values
df_test['NumOfProducts'] = df.loc[X_test.index, 'NumOfProducts'].values
df_test['Geography']     = df.loc[X_test.index, 'Geography'].values
df_test['PointsEarned']  = df.loc[X_test.index, 'PointsEarned'].values
df_test['Satisfaction']  = df.loc[X_test.index, 'Satisfaction'].values
df_test['CardType']      = df.loc[X_test.index, 'CardType'].values

tier_counts = df_test['RiskTier'].value_counts()
tier_order  = ['High Risk', 'Medium Risk', 'Low Risk']
print("\nRisk Tier Distribution:")
for tier in tier_order:
    n = tier_counts.get(tier, 0)
    print(f"  {tier}: {n} ({n/len(df_test):.1%})")

# ═══════════════════════════════════════════════════════════════════════════
# 6. DASHBOARD 1: EDA & Feature Insights (using real data)
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating Dashboard 1: EDA...")

fig = plt.figure(figsize=(22, 18), facecolor='white')
fig.suptitle('Customer Churn & Retention Analytics — Exploratory Analysis',
             fontsize=18, fontweight='bold', color=COLORS['primary'], y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

# 1. Churn rate by Geography
ax1 = fig.add_subplot(gs[0, 0])
geo_churn = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
bars = ax1.bar(geo_churn.index, geo_churn.values * 100,
               color=[COLORS['danger'], COLORS['warning'], COLORS['success']],
               edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, geo_churn.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_title('Churn Rate by Geography', fontweight='bold', color=COLORS['primary'])
ax1.set_ylabel('Churn Rate (%)')
ax1.set_ylim(0, geo_churn.max() * 130)

# 2. Churn rate by # Products
ax2 = fig.add_subplot(gs[0, 1])
prod_churn = df.groupby('NumOfProducts')['Exited'].mean()
colors_prod = [COLORS['success'] if v < 0.25 else COLORS['warning'] if v < 0.50 else COLORS['danger']
               for v in prod_churn.values]
bars2 = ax2.bar(prod_churn.index.astype(str), prod_churn.values * 100,
                color=colors_prod, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, prod_churn.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_title('Churn Rate by # Products', fontweight='bold', color=COLORS['primary'])
ax2.set_xlabel('Number of Products')
ax2.set_ylabel('Churn Rate (%)')

# 3. Age distribution: churned vs retained
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df[df['Exited'] == 0]['Age'], bins=30, alpha=0.6,
         color=COLORS['success'], label='Retained', density=True)
ax3.hist(df[df['Exited'] == 1]['Age'], bins=30, alpha=0.6,
         color=COLORS['danger'], label='Churned', density=True)
ax3.set_title('Age Distribution by Churn', fontweight='bold', color=COLORS['primary'])
ax3.set_xlabel('Age')
ax3.set_ylabel('Density')
ax3.legend(fontsize=9)

# 4. Balance distribution
ax4 = fig.add_subplot(gs[0, 3])
bal_churned  = df[df['Exited'] == 1]['Balance'].clip(0, 250000)
bal_retained = df[df['Exited'] == 0]['Balance'].clip(0, 250000)
ax4.hist(bal_retained, bins=40, alpha=0.55, color=COLORS['success'],
         label='Retained', density=True)
ax4.hist(bal_churned,  bins=40, alpha=0.55, color=COLORS['danger'],
         label='Churned', density=True)
ax4.set_title('Balance Distribution', fontweight='bold', color=COLORS['primary'])
ax4.set_xlabel('Account Balance ($)')
ax4.set_ylabel('Density')
ax4.legend(fontsize=9)
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

# 5. Satisfaction vs Churn
ax5 = fig.add_subplot(gs[1, 0])
sat_churn = df.groupby('Satisfaction')['Exited'].mean()
colors_sat = [COLORS['danger'] if v > 0.35 else COLORS['warning'] if v > 0.20 else COLORS['success']
              for v in sat_churn.values]
ax5.barh(sat_churn.index.astype(str), sat_churn.values * 100, color=colors_sat, edgecolor='white')
ax5.set_title('Churn Rate by Satisfaction Score', fontweight='bold', color=COLORS['primary'])
ax5.set_xlabel('Churn Rate (%)')
ax5.set_ylabel('Satisfaction Score')

# 6. Complaints vs Churn (binary in this dataset)
ax6 = fig.add_subplot(gs[1, 1])
comp_churn = df.groupby('NumComplaints')['Exited'].mean()
comp_labels = ['No Complaint', 'Has Complaint']
ax6.bar(comp_labels[:len(comp_churn)], comp_churn.values * 100,
        color=[COLORS['success'], COLORS['danger']][:len(comp_churn)],
        edgecolor='white', linewidth=1.5, width=0.5)
for i, val in enumerate(comp_churn.values):
    ax6.text(i, val * 100 + 0.5, f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')
ax6.set_title('Churn by Complaint Status', fontweight='bold', color=COLORS['primary'])
ax6.set_ylabel('Churn Rate (%)')

# 7. Active member vs Churn
ax7 = fig.add_subplot(gs[1, 2])
act_churn = df.groupby('IsActiveMember')['Exited'].mean()
ax7.bar(['Inactive', 'Active'], act_churn.values * 100,
        color=[COLORS['danger'], COLORS['success']], edgecolor='white', linewidth=1.5, width=0.5)
for i, val in enumerate(act_churn.values):
    ax7.text(i, val * 100 + 0.3, f'{val:.1%}', ha='center', fontsize=12, fontweight='bold')
ax7.set_title('Churn: Active vs Inactive', fontweight='bold', color=COLORS['primary'])
ax7.set_ylabel('Churn Rate (%)')
ax7.set_ylim(0, max(act_churn.values) * 140)

# 8. Points Earned distribution by churn (unique to this dataset)
ax8 = fig.add_subplot(gs[1, 3])
ax8.hist(df[df['Exited'] == 0]['PointsEarned'], bins=30, alpha=0.6,
         color=COLORS['success'], label='Retained', density=True)
ax8.hist(df[df['Exited'] == 1]['PointsEarned'], bins=30, alpha=0.6,
         color=COLORS['danger'], label='Churned', density=True)
ax8.set_title('Loyalty Points Distribution', fontweight='bold', color=COLORS['primary'])
ax8.set_xlabel('Points Earned')
ax8.set_ylabel('Density')
ax8.legend(fontsize=9)

# 9. Correlation heatmap
ax9 = fig.add_subplot(gs[2, :2])
num_cols = ['Age', 'Tenure', 'NumOfProducts', 'CreditScore', 'Balance',
            'IsActiveMember', 'Satisfaction', 'NumComplaints', 'PointsEarned', 'Exited']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.6, vmin=-0.6, center=0,
            annot=True, fmt='.2f', linewidths=0.5, ax=ax9,
            annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
ax9.set_title('Feature Correlation Matrix', fontweight='bold', color=COLORS['primary'])

# 10. Churn by Card Type & Gender
ax10 = fig.add_subplot(gs[2, 2:])
pivot = df.groupby(['CardType', 'Gender'])['Exited'].mean().unstack() * 100
x_pos = np.arange(len(pivot))
w = 0.35
ax10.bar(x_pos - w/2, pivot['Female'], w, label='Female',
         color=COLORS['accent'], alpha=0.85, edgecolor='white')
ax10.bar(x_pos + w/2, pivot['Male'],   w, label='Male',
         color=COLORS['primary'], alpha=0.85, edgecolor='white')
ax10.set_xticks(x_pos)
ax10.set_xticklabels(pivot.index)
ax10.set_title('Churn Rate by Card Type & Gender', fontweight='bold', color=COLORS['primary'])
ax10.set_ylabel('Churn Rate (%)')
ax10.legend()

plt.savefig(DASHBOARD_DIR / 'dashboard1_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Dashboard 1 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 7. DASHBOARD 2: Model Performance Comparison
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Dashboard 2: Model Performance...")

model_colors = {
    'Decision Tree':       COLORS['dt'],
    'Random Forest':       COLORS['rf'],
    'Gradient Boosting':   COLORS['gb'],
    'Deep Neural Network': COLORS['dnn'],
}

fig2, axes = plt.subplots(2, 3, figsize=(20, 13), facecolor='white')
fig2.suptitle('Model Performance Comparison — Churn Prediction',
              fontsize=18, fontweight='bold', color=COLORS['primary'])

# A. ROC Curves
ax = axes[0, 0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
            color=model_colors[name], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontweight='bold', color=COLORS['primary'])
ax.legend(fontsize=8, loc='lower right')

# B. Precision-Recall Curves
ax = axes[0, 1]
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['prob'])
    ax.plot(rec, prec, label=f"{name} (AP={res['avg_prec']:.3f})",
            color=model_colors[name], linewidth=2)
baseline = y_test.mean()
ax.axhline(baseline, color='gray', linestyle='--', alpha=0.5,
           label=f'Baseline ({baseline:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves', fontweight='bold', color=COLORS['primary'])
ax.legend(fontsize=8)

# C. Metric bar chart
ax = axes[0, 2]
metric_names = ['Accuracy', 'ROC-AUC', 'Avg Precision']
x_pos = np.arange(len(metric_names))
w = 0.18
for i, (name, res) in enumerate(results.items()):
    vals = [res['acc'], res['roc_auc'], res['avg_prec']]
    offset = (i - 1.5) * w
    ax.bar(x_pos + offset, vals, w, label=name,
           color=model_colors[name], alpha=0.85, edgecolor='white')
ax.set_xticks(x_pos)
ax.set_xticklabels(metric_names)
ax.set_ylim(0.5, 1.0)
ax.set_title('Metrics Comparison', fontweight='bold', color=COLORS['primary'])
ax.legend(fontsize=8)
ax.set_ylabel('Score')

# D. Confusion matrices for three models
for idx, (nm, res) in enumerate(list(results.items())[:3]):
    ax = axes[1, idx]
    cm = confusion_matrix(y_test, res['pred'])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'],
                cbar=False, linewidths=1)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, f'({cm_pct[i, j]:.1%})',
                    ha='center', va='center', fontsize=9, color='gray')
    ax.set_title(f'Confusion Matrix\n{nm}', fontweight='bold', color=COLORS['primary'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(DASHBOARD_DIR / 'dashboard2_model_perf.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Dashboard 2 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 8. DASHBOARD 3: Feature Importance & Risk Tiering
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Dashboard 3: Feature Importance & Risk Tiers...")

fig3 = plt.figure(figsize=(22, 16), facecolor='white')
fig3.suptitle('Feature Importance & Customer Risk Segmentation',
              fontsize=18, fontweight='bold', color=COLORS['primary'], y=0.98)
gs3 = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.42, wspace=0.38)

# A. RF Feature Importance (top 15)
ax = fig3.add_subplot(gs3[0, :2])
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
top_n = importances.tail(15)
colors_imp = [COLORS['danger'] if v > top_n.quantile(0.7) else
              COLORS['warning'] if v > top_n.quantile(0.4) else
              COLORS['rf'] for v in top_n.values]
ax.barh(top_n.index, top_n.values, color=colors_imp, edgecolor='white')
ax.set_title('Top-15 Feature Importances (Random Forest)', fontweight='bold', color=COLORS['primary'])
ax.set_xlabel('Importance Score')

# B. Risk tier pie chart
ax = fig3.add_subplot(gs3[0, 2])
tier_vals = [tier_counts.get(t, 0) for t in tier_order]
wedges, texts, autotexts = ax.pie(
    tier_vals, labels=tier_order, autopct='%1.1f%%',
    colors=RISK_PALETTE, startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=2))
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
ax.set_title('Customer Risk Tier Distribution', fontweight='bold', color=COLORS['primary'])

# C. Churn probability histogram by tier
ax = fig3.add_subplot(gs3[1, 0])
for tier, col in zip(tier_order, RISK_PALETTE):
    subset = df_test[df_test['RiskTier'] == tier]['ChurnProb']
    ax.hist(subset, bins=25, alpha=0.65, color=col, label=tier, density=True)
ax.axvline(0.35, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax.axvline(0.60, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax.set_title('Churn Probability by Risk Tier', fontweight='bold', color=COLORS['primary'])
ax.set_xlabel('Predicted Churn Probability')
ax.set_ylabel('Density')
ax.legend(fontsize=9)

# D. Average balance by risk tier (CLV proxy)
ax = fig3.add_subplot(gs3[1, 1])
tier_balance = df_test.groupby('RiskTier')['Balance'].agg(['mean', 'median'])
x = np.arange(len(tier_order))
w = 0.35
tier_vals_mean   = [tier_balance.loc[t, 'mean']   if t in tier_balance.index else 0 for t in tier_order]
tier_vals_median = [tier_balance.loc[t, 'median'] if t in tier_balance.index else 0 for t in tier_order]
ax.bar(x - w/2, tier_vals_mean,   w, label='Mean',   color=COLORS['primary'], alpha=0.85)
ax.bar(x + w/2, tier_vals_median, w, label='Median', color=COLORS['accent'],  alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(tier_order)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}k'))
ax.set_title('Balance by Risk Tier (CLV Proxy)', fontweight='bold', color=COLORS['primary'])
ax.set_ylabel('Account Balance')
ax.legend()

# E. Calibration curves
ax = fig3.add_subplot(gs3[1, 2])
for name, res in results.items():
    frac_pos, mean_pred = calibration_curve(y_test, res['prob'], n_bins=10)
    ax.plot(mean_pred, frac_pos, marker='o', markersize=4,
            label=name, color=model_colors[name], linewidth=1.8)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves', fontweight='bold', color=COLORS['primary'])
ax.legend(fontsize=8)

plt.savefig(DASHBOARD_DIR / 'dashboard3_features_risk.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Dashboard 3 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 9. DASHBOARD 4: Retention Strategy & Business Impact
#    CAC vs LTV trade-off framework — directly analogous to growth analytics
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Dashboard 4: Retention Strategy...")

# Business assumptions (CAC vs CLV framework)
CLV_BY_TIER        = {'High Risk': 4200, 'Medium Risk': 2800, 'Low Risk': 1500}
INTERVENTION_COST  = {'High Risk': 350,  'Medium Risk': 120,  'Low Risk': 30}
SAVE_RATE          = {'High Risk': 0.30, 'Medium Risk': 0.45, 'Low Risk': 0.60}

df_biz = pd.DataFrame([{
    'Tier': t,
    'Count': tier_counts.get(t, 0),
    'CLV': CLV_BY_TIER[t],
    'IntCost': INTERVENTION_COST[t],
    'SaveRate': SAVE_RATE[t],
} for t in tier_order])
df_biz['TotalIntCost']    = df_biz['Count'] * df_biz['IntCost']
df_biz['ExpectedSaved']   = (df_biz['Count'] * df_biz['SaveRate']).astype(int)
df_biz['RevenueRetained'] = df_biz['ExpectedSaved'] * df_biz['CLV']
df_biz['NetROI']          = df_biz['RevenueRetained'] - df_biz['TotalIntCost']

fig4 = plt.figure(figsize=(22, 14), facecolor='white')
fig4.suptitle('Retention Strategy & Business Impact Analysis',
              fontsize=18, fontweight='bold', color=COLORS['primary'], y=0.98)
gs4 = gridspec.GridSpec(2, 3, figure=fig4, hspace=0.42, wspace=0.40)

# A. Revenue retained vs intervention cost
ax = fig4.add_subplot(gs4[0, :2])
x = np.arange(len(tier_order))
w = 0.30
ax.bar(x - w, df_biz['TotalIntCost'],    w, label='Intervention Cost',
       color=COLORS['warning'], alpha=0.85, edgecolor='white')
ax.bar(x,     df_biz['RevenueRetained'], w, label='Revenue Retained',
       color=COLORS['success'], alpha=0.85, edgecolor='white')
ax.bar(x + w, df_biz['NetROI'],          w, label='Net ROI',
       color=COLORS['primary'], alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(tier_order)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1e6:.1f}M' if v >= 1e6 else f'${v/1000:.0f}k'))
ax.set_title('Business Impact: Retention Investment vs Return by Tier (CAC vs CLV)',
             fontweight='bold', color=COLORS['primary'])
ax.set_ylabel('Amount ($)')
ax.legend()
total_roi = df_biz['NetROI'].sum()
ax.annotate(f'Total Net ROI: ${total_roi/1e6:.2f}M',
            xy=(0.98, 0.92), xycoords='axes fraction',
            ha='right', fontsize=12, fontweight='bold',
            color=COLORS['primary'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECF0F1', alpha=0.8))

# B. Customers saved per tier
ax = fig4.add_subplot(gs4[0, 2])
ax.bar(tier_order, df_biz['ExpectedSaved'], color=RISK_PALETTE, edgecolor='white', linewidth=1.5)
for i, (cnt, saved) in enumerate(zip(df_biz['Count'], df_biz['ExpectedSaved'])):
    pct_str = f'({saved/cnt:.0%})' if cnt > 0 else '(N/A)'
    ax.text(i, saved + 5, f'{saved}\n{pct_str}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_title('Expected Customers Saved\n(by Retention Intervention)', fontweight='bold',
             color=COLORS['primary'])
ax.set_ylabel('# Customers Saved')

# C. Churn probability vs Balance scatter (high-value at-risk cohort)
ax = fig4.add_subplot(gs4[1, 0])
for tier, col in zip(tier_order, RISK_PALETTE):
    sub = df_test[df_test['RiskTier'] == tier]
    ax.scatter(sub['Balance'], sub['ChurnProb'], c=col, alpha=0.25, s=15,
               label=tier, linewidths=0)
ax.set_xlabel('Account Balance ($)')
ax.set_ylabel('Predicted Churn Probability')
ax.set_title('Churn Probability vs Balance\n(by Risk Tier)', fontweight='bold', color=COLORS['primary'])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}k'))
ax.axhline(0.60, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(0.35, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.legend(fontsize=8, markerscale=2)

# D. Risk tier composition by Tenure band
ax = fig4.add_subplot(gs4[1, 1])
df_test['TenureBand'] = pd.cut(df_test['Tenure'], bins=[-0.1, 2, 5, 10, 16],
                                labels=['0-2yr', '3-5yr', '6-10yr', '10+yr'])
pivot2 = df_test.groupby(['TenureBand', 'RiskTier']).size().unstack(fill_value=0)
pivot2_pct = pivot2.div(pivot2.sum(axis=1), axis=0) * 100
pivot2_pct[['Low Risk', 'Medium Risk', 'High Risk']].plot(
    kind='bar', stacked=True, ax=ax,
    color=[COLORS['success'], COLORS['warning'], COLORS['danger']],
    edgecolor='white', linewidth=0.5)
ax.set_title('Risk Tier Composition by Tenure', fontweight='bold', color=COLORS['primary'])
ax.set_xlabel('Customer Tenure')
ax.set_ylabel('% of Customers')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(fontsize=9, loc='upper right')

# E. Model threshold analysis (precision vs recall tradeoff)
ax = fig4.add_subplot(gs4[1, 2])
thresholds = np.linspace(0.1, 0.9, 80)
best_probs = best_res['prob']
precisions, recalls, f1s, flagged_pct = [], [], [], []
for t in thresholds:
    pred_t = (best_probs >= t).astype(int)
    if pred_t.sum() == 0:
        precisions.append(0); recalls.append(0); f1s.append(0)
    else:
        precisions.append(precision_score(y_test, pred_t, zero_division=0))
        recalls.append(recall_score(y_test, pred_t, zero_division=0))
        f1s.append(f1_score(y_test, pred_t, zero_division=0))
    flagged_pct.append(pred_t.mean())
ax.plot(thresholds, precisions, label='Precision', color=COLORS['primary'], linewidth=2)
ax.plot(thresholds, recalls,    label='Recall',    color=COLORS['danger'],  linewidth=2)
ax.plot(thresholds, f1s,        label='F1',        color=COLORS['accent'],  linewidth=2, linestyle='--')
ax2r = ax.twinx()
ax2r.plot(thresholds, [f * 100 for f in flagged_pct], color='gray', alpha=0.4,
          linewidth=1.5, linestyle=':', label='% Flagged')
ax2r.set_ylabel('% Customers Flagged', color='gray', fontsize=9)
ax.axvline(0.40, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Score')
ax.set_title(f'Threshold Analysis\n({best_name})', fontweight='bold', color=COLORS['primary'])
ax.legend(loc='center right', fontsize=8)

plt.savefig(DASHBOARD_DIR / 'dashboard4_retention_strategy.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Dashboard 4 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 10. SUMMARY REPORT (JSON)
# ═══════════════════════════════════════════════════════════════════════════
summary = {
    'dataset': {
        'source':     DATA_PATH,
        'rows':       int(len(df)),
        'features':   int(len(FEATURE_COLS)),
        'churn_rate': float(df['Exited'].mean()),
    },
    'models': {name: {
        'accuracy': float(r['acc']),
        'roc_auc':  float(r['roc_auc']),
        'avg_prec': float(r['avg_prec']),
    } for name, r in results.items()},
    'best_model': best_name,
    'risk_tiers': {t: int(tier_counts.get(t, 0)) for t in tier_order},
    'business_impact': {
        'total_net_roi_usd':      int(df_biz['NetROI'].sum()),
        'total_customers_saved':  int(df_biz['ExpectedSaved'].sum()),
    },
}
joblib.dump(best_res['model'], MODEL_DIR / 'best_churn_model.joblib')

with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs saved to {OUTPUT_DIR}")
print(json.dumps(summary, indent=2))
