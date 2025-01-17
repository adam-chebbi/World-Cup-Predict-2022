import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Ensure the file exists in the working directory
file_path = 'Coupe Du Monde Stats 2022.xlsx'
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file '{file_path}' was not found. Please place it in the working directory or provide the correct path.")

# Load datasets
groups_data = pd.read_excel(file_path, sheet_name='2022_world_cup_groups')
matches_2022_data = pd.read_excel(file_path, sheet_name='2022_world_cup_matches')
world_cup_matches_data = pd.read_excel(file_path, sheet_name='world_cup_matches')

# Prepare data for training
# Select relevant columns
historical_matches = world_cup_matches_data[['Home Team', 'Away Team', 'Home Goals', 'Away Goals', 'Winning Team']].copy()

# Add a target column
historical_matches['Result'] = historical_matches['Winning Team'].apply(
    lambda x: 'Draw' if pd.isna(x) else ('Home Win' if x == historical_matches['Home Team'] else 'Away Win')
)

# Encode categorical variables
encoder = LabelEncoder()
historical_matches['Home Team Encoded'] = encoder.fit_transform(historical_matches['Home Team'])
historical_matches['Away Team Encoded'] = encoder.transform(historical_matches['Away Team'])
historical_matches['Result Encoded'] = encoder.fit_transform(historical_matches['Result'])

# Features and target
X = historical_matches[['Home Team Encoded', 'Away Team Encoded']]
y = historical_matches['Result Encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Prepare data for prediction
# Ensure all teams in the 2022 data exist in the encoder
missing_teams = set(matches_2022_data['Home Team']).union(matches_2022_data['Away Team']) - set(encoder.classes_)
if missing_teams:
    raise ValueError(f"The following teams are missing from the encoder: {missing_teams}")

matches_2022_data['Home Team Encoded'] = encoder.transform(matches_2022_data['Home Team'])
matches_2022_data['Away Team Encoded'] = encoder.transform(matches_2022_data['Away Team'])
prediction_features = matches_2022_data[['Home Team Encoded', 'Away Team Encoded']]

# Predict results for the remaining matches
matches_2022_data['Predicted Result'] = model.predict(prediction_features)
matches_2022_data['Predicted Result'] = matches_2022_data['Predicted Result'].map(
    {idx: label for idx, label in enumerate(encoder.classes_)}
)

# Save predictions
def save_predictions(matches):
    output_file = 'predicted_world_cup_results.csv'
    matches[['Home Team', 'Away Team', 'Predicted Result']].to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'.")

save_predictions(matches_2022_data)

# Instructions for GitHub
print("\n\nTo use this script, upload all required files to the same directory and ensure dependencies are installed. Ensure the dataset is named 'Coupe Du Monde Stats 2022.xlsx'.\n")
