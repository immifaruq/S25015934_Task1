import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Outcomes Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Predicting Customer Outcomes")
st.markdown("Upload your dataset, explore the data, and compare ML models.")

# ── 1. Data Upload ────────────────────────────────────────────────────────────
st.header("1. Load Dataset")
uploaded_file = st.file_uploader("dataset.csv", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()  # Halt execution until a file is uploaded

df = pd.read_csv(uploaded_file, encoding='latin1')
st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head(), use_container_width=True)

# ── 2. Data Quality ───────────────────────────────────────────────────────────
st.header("2. Data Quality")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    st.dataframe(missing, use_container_width=True)

with col2:
    st.subheader("Duplicate Rows")
    dupes = df.duplicated().sum()
    st.metric("Duplicate Rows", dupes)
    st.subheader("Dataset Shape")
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])

st.subheader("Descriptive Statistics")
st.dataframe(df.describe(), use_container_width=True)

# ── 3. Exploratory Data Analysis ──────────────────────────────────────────────
st.header("3. Exploratory Data Analysis")

# Histograms
st.subheader("Feature Distributions")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
n_cols = 4
n_rows = int(np.ceil(len(numeric_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=15, color='#3498db', edgecolor='black')
    axes[i].set_title(col, fontsize=9)
    axes[i].tick_params(labelsize=7)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Distribution of Features", fontsize=14)
plt.tight_layout()
st.pyplot(fig)          # ✅ Correct Streamlit call
plt.close(fig)

# Boxplots
st.subheader("Boxplot – Outlier Detection")
fig, ax = plt.subplots(figsize=(20, 8))
df[numeric_cols].boxplot(ax=ax)
ax.set_xticklabels(numeric_cols, rotation=90)
ax.set_title("Boxplot of Features")
plt.tight_layout()
st.pyplot(fig)          # ✅
plt.close(fig)

# Target distribution
if 'Target' in df.columns:
    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    df['Target'].value_counts().plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], edgecolor='black')
    ax.set_title("Distribution of Target Variable")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig)      # ✅
    plt.close(fig)

# Correlation heatmap
st.subheader("Correlation Matrix")
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Spectral', linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix of Numeric Features")
plt.tight_layout()
st.pyplot(fig)          # ✅
plt.close(fig)

# ── 4. Preprocessing ──────────────────────────────────────────────────────────
st.header("4. Preprocessing & Model Training")

if 'Target' not in df.columns:
    st.error("No 'Target' column found in the dataset.")
    st.stop()

# Map target
df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})
df.dropna(subset=['Target'], inplace=True)

X = df.drop(columns=['Target'])
y = df['Target']

# Keep only numeric features (drop any remaining non-numeric columns)
X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"**Training set:** {X_train.shape}  |  **Test set:** {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 5. Train Models ───────────────────────────────────────────────────────────
with st.spinner("Training models – this may take a moment…"):

    # XGBoost
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)

    # AdaBoost
    ada_model = AdaBoostClassifier(random_state=42)
    ada_model.fit(X_train_scaled, y_train)
    y_pred_ada = ada_model.predict(X_test_scaled)

    # Logistic Regression
    logreg_model = LogisticRegression(random_state=42, max_iter=1000)
    logreg_model.fit(X_train_scaled, y_train)
    y_pred_logreg = logreg_model.predict(X_test_scaled)

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    y_pred_dt = dt_model.predict(X_test_scaled)

st.success("All models trained!")

# ── 6. Results ────────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1 Score':  f1_score(y_true, y_pred, zero_division=0),
        'AUC':       roc_auc_score(y_true, y_prob),
    }

results = {
    'XGBoost':            get_metrics(y_test, y_pred_xgb,   xgb_model.predict_proba(X_test_scaled)[:, 1]),
    'AdaBoost':           get_metrics(y_test, y_pred_ada,   ada_model.predict_proba(X_test_scaled)[:, 1]),
    'Logistic Regression':get_metrics(y_test, y_pred_logreg,logreg_model.predict_proba(X_test_scaled)[:, 1]),
    'Decision Tree':      get_metrics(y_test, y_pred_dt,    dt_model.predict_proba(X_test_scaled)[:, 1]),
}

comparison_df = pd.DataFrame(results).T
st.header("5. Model Comparison")
st.dataframe(comparison_df.style.highlight_max(axis=0, color='#d4edda').format("{:.4f}"),
             use_container_width=True)

# Bar charts per metric
st.subheader("Performance Charts")
palettes = ['viridis', 'plasma', 'cividis', 'magma']
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

cols = st.columns(2)
for i, metric in enumerate(metrics_to_plot):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=comparison_df.index,
        y=comparison_df[metric],
        palette=palettes[i % len(palettes)],
        ax=ax
    )
    ax.set_title(f"Model Comparison – {metric}", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylabel(f"{metric} Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=30)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=9
        )
    plt.tight_layout()
    cols[i % 2].pyplot(fig)   # ✅
    plt.close(fig)

# ROC Curves
st.subheader("ROC Curves")
fig, ax = plt.subplots(figsize=(8, 6))
model_map = {
    'XGBoost': (xgb_model, y_pred_xgb),
    'AdaBoost': (ada_model, y_pred_ada),
    'Logistic Regression': (logreg_model, y_pred_logreg),
    'Decision Tree': (dt_model, y_pred_dt),
}
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for (name, (model, _)), color in zip(model_map.items(), colors):
    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, lw=2)
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve – All Models")
ax.legend(loc='lower right')
plt.tight_layout()
st.pyplot(fig)              # ✅
plt.close(fig)

# Confusion Matrices
st.subheader("Confusion Matrices")
preds = [y_pred_xgb, y_pred_ada, y_pred_logreg, y_pred_dt]
names = ['XGBoost', 'AdaBoost', 'Logistic Regression', 'Decision Tree']
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, pred, name in zip(axes, preds, names):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Dropout', 'Graduate'],
                yticklabels=['Dropout', 'Graduate'])
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
st.pyplot(fig)              # ✅
plt.close(fig)

# Classification Reports
st.subheader("Classification Reports")
for pred, name in zip(preds, names):
    with st.expander(f"📋 {name} – Full Classification Report"):
        report = classification_report(y_test, pred,
                                       target_names=['Dropout', 'Graduate'])
        st.text(report)