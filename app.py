import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Title and Introduction
st.title("⚽ FIFA 21 Rating Predictor")
st.write("""
### Project Overview
This app uses **Machine Learning** to predict a player's Overall Rating based on their stats. 
Use the sidebar to adjust the values and see how the rating changes!
""")

# 2. Load and Prepare Data
@st.cache_data # This makes it fast by caching the data
def load_data():
    df = pd.read_csv('players_21.csv')
    
    # We will use the top influential skills we found in our analysis
    # "Reactions" was the #1 predictor!
    features = ['movement_reactions', 'skill_ball_control', 'passing', 
                'dribbling', 'shooting', 'physic', 'defending', 'attacking_crossing']
    
    # Clean data: Drop rows where these specific stats are missing
    df = df.dropna(subset=features + ['overall'])
    return df, features

df, feature_names = load_data()

# 3. Train the Model 
X = df[feature_names]
y = df['overall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Sidebar Inputs (The "Interactive" Part)
st.sidebar.header("Player Stats")

def user_input_features():
    # We create a slider for each feature
    # The range is 0 to 100, default is the average (50)
    reactions = st.sidebar.slider('Reactions (The Secret Stat!)', 0, 100, 50)
    control = st.sidebar.slider('Ball Control', 0, 100, 50)
    passing = st.sidebar.slider('Passing', 0, 100, 50)
    dribbling = st.sidebar.slider('Dribbling', 0, 100, 50)
    shooting = st.sidebar.slider('Shooting', 0, 100, 50)
    physic = st.sidebar.slider('Physical', 0, 100, 50)
    defending = st.sidebar.slider('Defending', 0, 100, 50)
    crossing = st.sidebar.slider('Crossing', 0, 100, 50)
    
    data = {
        'movement_reactions': reactions,
        'skill_ball_control': control,
        'passing': passing,
        'dribbling': dribbling,
        'shooting': shooting,
        'physic': physic,
        'defending': defending,
        'attacking_crossing': crossing
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 5. Display Prediction
st.subheader("Predicted Rating")
prediction = model.predict(input_df)

# Show the result as a big metric
st.metric(label="Overall Rating", value=f"{prediction[0]:.0f}")

# 6. Show Feature Importance Chart 
st.write("---")
st.subheader("Why this rating?")
st.write("The model values **Reactions** and **Ball Control** the most.")

import matplotlib.pyplot as plt
import seaborn as sns

importance = model.feature_importances_
fig, ax = plt.subplots()
sns.barplot(x=importance, y=feature_names, ax=ax, palette="viridis")
plt.title("Feature Importance in this Model")
st.pyplot(fig)