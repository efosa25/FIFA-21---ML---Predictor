# FIFA 21 Player Rating Predictor ⚽

## Project Overview
This project uses Machine Learning (Random Forest Regressor) to predict a football player's "Overall Rating" based on their in-game stats. It analyzes data from the FIFA 21 video game to determine which technical skills—like Reactions and Ball Control—matter most.

## 🚀 New Feature: Interactive Web App
I have transformed the static analysis into a **real-time dashboard** using Streamlit.

![Dashboard Preview](dashboard_preview.png)
![Dashboard Preview](dashboard_preview2.png)

* **Interactive Sliders:** Users can adjust specific player stats (Speed, Shooting, Defending, etc.) to see how they impact the overall score.
* **Instant Prediction:** The ML model runs in real-time to generate a rating (0-100).
* **Dynamic Visuals:** Includes a live bar chart showing which features are driving the current prediction.

## Key Findings (Data Science)
* **Model Accuracy:** The Random Forest model achieved a Mean Absolute Error (MAE) of **0.20** (on a 100-point scale).
* **Feature Importance:** After removing financial data (Value/Wage) to prevent data leakage, the model revealed that **Movement Reactions** is the #1 predictor of a high rating, followed by Ball Control.

## Technologies Used
* **Python** 
* **Streamlit** (Web Dashboard)
* **Scikit-Learn** (Machine Learning)
* **Pandas** (Data Engineering/ Manipulation)
* **Seaborn / Matplotlib** (Data Visualization)

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn streamlit seaborn matplotlib
3. To run the Analysis: Open fifa_ml.ipynb in Jupyter/VS Code.
4. To launch the Dashboard: Run the following command in your terminal:
    ```bash
    streamlit run app.py