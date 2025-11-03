# import pandas as pd

# # Load data
# df = pd.read_csv("disease_symptom_dataset.csv")

# # Create one-hot encoded symptom table
# symptoms_list = sorted(df['symptom_name'].unique())
# data = []

# for disease in df['disease_name'].unique():
#     row = {'disease_name': disease}
#     disease_symptoms = df[df['disease_name'] == disease]['symptom_name'].values
#     for symptom in symptoms_list:
#         row[symptom] = 1 if symptom in disease_symptoms else 0
#     data.append(row)

# ml_data = pd.DataFrame(data)
# ml_data.to_csv("ml_training_data.csv", index=False)

# print("✅ Machine Learning formatted dataset created successfully!")
# print(ml_data.head())

import pandas as pd

# ✅ Step 1: Load the raw dataset
df = pd.read_csv("disease_symptom_dataset.csv")

print("🔹 Original Data:")
print(df.head())

# ✅ Step 2: Get the unique list of symptoms
symptoms_list = sorted(df['symptom_name'].unique())

# ✅ Step 3: Create a one-hot encoded table (disease × symptoms)
data = []

for disease in df['disease_name'].unique():
    row = {'disease_name': disease}
    disease_symptoms = df[df['disease_name'] == disease]['symptom_name'].values
    for symptom in symptoms_list:
        row[symptom] = 1 if symptom in disease_symptoms else 0
    data.append(row)

# ✅ Step 4: Convert into DataFrame
ml_data = pd.DataFrame(data)

print("\n✅ Transformed Data (ML Format):")
print(ml_data.head())

# ✅ Step 5: Save ML-ready dataset
ml_data.to_csv("ml_training_data.csv", index=False)
print("\n💾 File saved as 'ml_training_data.csv' successfully!")
