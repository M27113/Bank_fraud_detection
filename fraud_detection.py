import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, recall_score
import joblib

# --- Load dataset ---
df = pd.read_csv("transactions.csv")

# --- Handle missing values ---
df['Amount'].fillna(df['Amount'].median(), inplace=True)
df['CardHolderAge'].fillna(df['CardHolderAge'].median(), inplace=True)
df['Location'].fillna(df['Location'].mode()[0], inplace=True)

# --- Features and target ---
X = df[['Amount','Time','Location','MerchantCategory','CardHolderAge']]
y = df['IsFraud']

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocessing ---
numeric_features = ['Amount','Time','CardHolderAge']
categorical_features = ['Location','MerchantCategory']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# --- Pipelines ---
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# --- Train models ---
rf_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)

# --- Predictions ---
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:,1]

y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:,1]

# --- Confusion matrices ---
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_lr = confusion_matrix(y_test, y_pred_lr)

# --- Classification reports ---
cr_rf = classification_report(y_test, y_pred_rf, target_names=['Not Fraud','Fraud'])
cr_lr = classification_report(y_test, y_pred_lr, target_names=['Not Fraud','Fraud'])

# --- ROC curves ---
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(5,4))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_rf:.2f})')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_lr:.2f})')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("roc_curve_comparison.png")
plt.close()

# --- Recall curves ---
plt.figure(figsize=(5,4))
plt.plot(tpr_rf, label=f'Random Forest Recall')
plt.plot(tpr_lr, label=f'Logistic Regression Recall')
plt.xlabel("Threshold Index")
plt.ylabel("Recall")
plt.title("Recall Comparison")
plt.legend()
plt.savefig("recall_comparison.png")
plt.close()

# --- Confusion matrix plots ---
plt.figure(figsize=(4,4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Fraud','Fraud'], yticklabels=['Not Fraud','Fraud'])
plt.title("Random Forest Confusion Matrix")
plt.savefig("confusion_matrix_rf.png")
plt.close()

plt.figure(figsize=(4,4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Oranges", xticklabels=['Not Fraud','Fraud'], yticklabels=['Not Fraud','Fraud'])
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("confusion_matrix_lr.png")
plt.close()

# --- Classification report tables ---
def save_classification_report_image(y_true, y_pred, model_name, filename):
    report_dict = classification_report(y_true, y_pred, target_names=['Not Fraud','Fraud'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    fig_width = max(6, report_df.shape[1] * 1.5)
    fig_height = max(2, report_df.shape[0] * 0.8)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.axis('off')
    tbl = ax.table(cellText=np.round(report_df.values,2), colLabels=report_df.columns, rowLabels=report_df.index, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(0.8,1.8)
    plt.title(f"{model_name} Classification Report")
    plt.savefig(filename)
    plt.close()

save_classification_report_image(y_test, y_pred_rf, "Random Forest", "classification_report_rf.png")
save_classification_report_image(y_test, y_pred_lr, "Logistic Regression", "classification_report_lr.png")

# --- Feature Importance ---
importances = rf_pipeline.named_steps['classifier'].feature_importances_
num_features = numeric_features
cat_features = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features = np.concatenate([num_features, cat_features])
feat_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,4))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# --- Fraud vs Non-Fraud Amount Distribution ---
plt.figure(figsize=(6,4))
sns.histplot(df, x='Amount', hue='IsFraud', bins=30, kde=False, palette=['green','red'])
plt.title("Transaction Amount Distribution - Fraud vs Non-Fraud")
plt.xlabel("Transaction Amount")
plt.ylabel("Count")
plt.savefig("amount_distribution.png")
plt.close()

# --- Manual Transaction Prediction ---
def manual_fraud_input(transaction, model_pipeline=rf_pipeline):
    df_trans = pd.DataFrame([transaction])
    prob = model_pipeline.predict_proba(df_trans)[0][1]
    pred = model_pipeline.predict(df_trans)[0]
    print(f"Fraud Probability: {prob:.2f}")
    print(f"Prediction: {'Fraud' if pred==1 else 'Not Fraud'}")
    return prob, pred

transaction_example = {
    'Amount': 514.72,
    'Time': 23833,
    'Location': 'Chicago',
    'MerchantCategory': 'Electronics',
    'CardHolderAge': 52
}

manual_fraud_input(transaction_example)

# --- Save trained pipeline ---
joblib.dump(rf_pipeline, "rf_model.pkl")
joblib.dump(preprocessor, "preprocessing_pipeline.pkl")

# --- Manual Transaction Image ---
plt.figure(figsize=(4,3))
prob_manual, _ = manual_fraud_input(transaction_example)
plt.bar(['Not Fraud','Fraud'], [1-prob_manual, prob_manual], color=['green','red'])
plt.title("Manual Transaction Prediction")
plt.ylabel("Probability")
plt.savefig("manual_transaction.png")
plt.close()

# --- Batch Prediction Example ---
batch_df = X_test.copy()
batch_df['Fraud_Probability'] = y_prob_rf
batch_df['Prediction'] = y_pred_rf
batch_df['Flag'] = np.where(batch_df['Prediction']==1, 'Fraud Alert','Safe')
fig, ax = plt.subplots(figsize=(10,2))
ax.axis('off')
tbl = ax.table(cellText=batch_df.head(4).values, colLabels=batch_df.columns, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1,1.5)
plt.title("Batch Prediction Example")
plt.savefig("batch_prediction.png")
plt.close()


