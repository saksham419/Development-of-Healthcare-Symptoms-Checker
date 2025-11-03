# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# import pickle

# # Load data
# df = pd.read_csv("ml_training_data.csv")

# # Prepare features & target
# X = df.drop('disease_name', axis=1)
# y = LabelEncoder().fit_transform(df['disease_name'])

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = MultinomialNB()
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# # Save model
# pickle.dump(model, open("symptom_checker_model.pkl", "wb"))
# print("🧠 Model saved as symptom_checker_model.pkl")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report
# import joblib

# # ✅ Step 1: Load the prepared dataset
# df = pd.read_csv("ml_training_data.csv")

# print("🔹 Dataset loaded successfully!")
# print(df.head())

# # ✅ Step 2: Split features (X) and target (y)
# X = df.drop(columns=['disease_name'])
# y = df['disease_name']

# # ✅ Step 3: Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # ✅ Step 4: Initialize and train the model
# model = MultinomialNB()
# model.fit(X_train, y_train)

# # ✅ Step 5: Evaluate the model
# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
# print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# # ✅ Step 6: Save the model
# joblib.dump(model, "disease_prediction_model.pkl")
# print("\n💾 Model saved as 'disease_prediction_model.pkl' successfully!")

# # ✅ Step 7: Save the symptom order (for later use in predictions)
# joblib.dump(list(X.columns), "symptom_list.pkl")
# print("💾 Symptom list saved as 'symptom_list.pkl' successfully!")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report
# import joblib

# # Load dataset
# df = pd.read_csv("ml_training_data.csv")
# df.dropna(inplace=True)

# # Strip disease name spaces if any
# df["disease_name"] = df["disease_name"].str.strip()

# # Split features and target
# X = df.drop(columns=["disease_name"])
# y = df["disease_name"]

# print("\n✅ Data loaded successfully. Checking class distribution:")
# print(df['disease_name'].value_counts())
# print("\nTotal samples:", len(df))
# print("Unique diseases:", df['disease_name'].nunique())

# # Stratified split ensures all diseases appear in train & test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42, stratify=y
# )

# print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)} samples")

# # Train model
# model = MultinomialNB()
# model.fit(X_train, y_train)

# # Evaluate model
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)

# print(f"\n🎯 Model Accuracy: {acc * 100:.2f}%")
# print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# # Save model
# joblib.dump(model, "disease_prediction_model.pkl")
# joblib.dump(list(X.columns), "symptom_list.pkl")
# print("\n💾 Model and symptom list saved successfully!")



# ================================================
# 📘 Phase 3 - AI Model Training for Disease Prediction
# ================================================

import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# ==================================================
# Step 1: Load Dataset
# ==================================================
print("📥 Loading dataset...")
df = pd.read_csv("ml_training_data.csv")

# Show dataset info
print("\n✅ Data loaded successfully. Checking class distribution:")
print(df['disease_name'].value_counts())

print(f"\nTotal samples: {len(df)}")
print(f"Unique diseases: {df['disease_name'].nunique()}")

# ==================================================
# Step 2: Balance Dataset (avoid stratify errors)
# ==================================================
# Make sure every disease has at least 5 samples
balanced_df = df.groupby('disease_name', group_keys=False).apply(
    lambda x: resample(x, replace=True, n_samples=max(5, len(x)), random_state=42)
)

print("\n✅ After balancing:")
print(balanced_df['disease_name'].value_counts())

df = balanced_df.reset_index(drop=True)

# ==================================================
# Step 3: Encode Symptoms for Model Input
# ==================================================
# Ensure symptom column is a list (split by comma if string)
df['symptoms'] = df['symptoms'].apply(lambda x: x.split(',') if isinstance(x, str) else x)

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['symptoms'])
y = df['disease_name']

# ==================================================
# Step 4: Split Dataset
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\n📊 Dataset split completed successfully!")

# ==================================================
# Step 5: Train Model
# ==================================================
print("\n🤖 Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=10,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ==================================================
# Step 6: Evaluate Model
# ==================================================
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ==================================================
# Step 7: Save Model and Symptom List
# ==================================================
with open("disease_prediction_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("symptom_list.pkl", "wb") as symptom_file:
    pickle.dump(mlb.classes_, symptom_file)

print("\n💾 Model saved as 'disease_prediction_model.pkl' successfully!")
print("💾 Symptom list saved as 'symptom_list.pkl' successfully!")
print("\n✅ Phase 3 Completed Successfully! 🚀")
