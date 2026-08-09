import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

print("🤖 1. Loading the hospital dataset into memory...")
df = pd.read_csv("app_and_ml/data/PPG_Dataset.csv", header=None)

# 1. THE FEATURES (X): We give the AI the first 1000 milliseconds of every patient's wave.
X = df.iloc[:, :1000]

# 2. THE TARGET (Y): We generate a mock "Vascular Age" based on how erratic the wave is. 
# (In a real hospital dataset, this would be a column typed by a doctor).
print("🧠 2. Formatting the Features (X) and Targets (Y)...")
# Scale the wave variance across a complete human longevity spectrum (18 to 110 years old)
raw_var = X.var(axis=1)
y = 18 + ((raw_var - raw_var.min()) / (raw_var.max() - raw_var.min())) * 92

# We hide 20% of the patients from the AI so we can test it like a pop-quiz later.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. THE BRAIN: We are using a Random Forest (An AI made of 100 decision trees)
print("🌲 3. Training the Random Forest AI (This might take 5 to 10 seconds)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)

# .fit() is the command that actually makes the AI learn!
model.fit(X_train, y_train)

# 4. THE POP-QUIZ: Let's see how smart it got.
accuracy = model.score(X_test, y_test)
print(f"🎯 4. AI Training Complete! Test Accuracy Score: {accuracy * 100:.2f}%")

# 5. SAVE THE BRAIN: We save the trained AI as a file so the Streamlit website can use it later.
print("💾 5. Saving the AI brain to a file...")
with open("app_and_ml/vascular_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Success! 'vascular_model.pkl' has been created.")