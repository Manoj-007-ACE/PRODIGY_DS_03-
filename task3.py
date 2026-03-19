import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# Prodigy InfoTech - Data Science Internship
# Task 3: Decision Tree Classifier - Bank Marketing Dataset
# ============================================================

df = pd.read_csv(r"C:\prodigy_task3\bank-additional-full.csv", sep=";")
print("Dataset loaded! Shape:", df.shape)

# ==================== DATA PREPROCESSING ====================

# Encode all text columns to numbers
le = LabelEncoder()
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Split features and target
X = df_encoded.drop("y", axis=1)
y = df_encoded["y"]

# Split into train and test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ==================== BUILD DECISION TREE ====================

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ==================== MODEL EVALUATION ====================

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

# ==================== PLOT 1: Decision Tree ====================

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns.tolist(),
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree - Bank Marketing Prediction", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(r"C:\prodigy_task3\plot1_decision_tree.png", dpi=150)
plt.show()
print("Plot 1 saved!")

# ==================== PLOT 2: Confusion Matrix ====================

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"])
plt.title("Confusion Matrix", fontsize=15, fontweight="bold")
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig(r"C:\prodigy_task3\plot2_confusion_matrix.png", dpi=150)
plt.show()
print("Plot 2 saved!")

# ==================== PLOT 3: Feature Importance ====================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
colors = sns.color_palette("Blues_d", len(importance))
bars = plt.barh(importance["Feature"], importance["Importance"],
                color=colors, edgecolor="white")
plt.title("Top 10 Most Important Features", fontsize=15, fontweight="bold")
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(r"C:\prodigy_task3\plot3_feature_importance.png", dpi=150)
plt.show()
print("Plot 3 saved!")

# ==================== PLOT 4: Target Distribution ====================

plt.figure(figsize=(7, 5))
ax = sns.countplot(x="y", data=df, palette=["#E24B4A", "#1D9E75"])
ax.bar_label(ax.containers[0], fontsize=12, fontweight="bold")
plt.title("Target Distribution (Subscribed?)", fontsize=15, fontweight="bold")
plt.xticks([0, 1], ["No", "Yes"], fontsize=12)
plt.xlabel("")
plt.ylabel("Number of Customers", fontsize=12)
plt.tight_layout()
plt.savefig(r"C:\prodigy_task3\plot4_target_distribution.png", dpi=150)
plt.show()
print("Plot 4 saved!")

# ==================== PLOT 5: Age vs Subscription ====================

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="age", hue="y", bins=30,
             palette=["#E24B4A", "#1D9E75"], alpha=0.7)
plt.title("Age Distribution by Subscription", fontsize=15, fontweight="bold")
plt.xlabel("Age", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.legend(["No", "Yes"], fontsize=11)
plt.tight_layout()
plt.savefig(r"C:\prodigy_task3\plot5_age_subscription.png", dpi=150)
plt.show()
print("Plot 5 saved!")

# ==================== PLOT 6: Job vs Subscription ====================

job_counts = df.groupby(["job", "y"]).size().unstack()
job_counts.plot(kind="bar", figsize=(12, 6),
                color=["#E24B4A", "#1D9E75"], edgecolor="white")
plt.title("Subscription by Job Type", fontsize=15, fontweight="bold")
plt.xlabel("Job", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.xticks(rotation=30, ha="right", fontsize=10)
plt.legend(["No", "Yes"], fontsize=11)
plt.tight_layout()
plt.savefig(r"C:\prodigy_task3\plot6_job_subscription.png", dpi=150)
plt.show()
print("Plot 6 saved!")

print(f"\nAll 6 plots saved in C:\\prodigy_task3\\")
print(f"\nFinal Model Accuracy: {accuracy*100:.2f}%")