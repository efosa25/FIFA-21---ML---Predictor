# FIFA 21 Player Rating Predictor ⚽

## Project Overview
This project uses Machine Learning (Random Forest Regressor) to predict a football player's "Overall Rating" based on their in-game stats. It was built using Python, Scikit-Learn, and Pandas.

## Key Findings
* **Model Accuracy:** The initial model achieved a Mean Absolute Error (MAE) of **0.20**.
* **Feature Analysis:** After removing financial data (Value/Wage) to prevent data leakage, the model revealed that **Movement Reactions** is the most critical skill for determining a player's rating, followed by Ball Control and Defending.

## Technologies Used
* Python 
* Scikit-Learn (Random Forest)
* Pandas (Data Manipulation)
* Matplotlib (Visualization)

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas scikit-learn matplotlib`
3. Run the Jupyter Notebook `fifa_ml.ipynb`.