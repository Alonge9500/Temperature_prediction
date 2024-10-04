# Temperature Prediction Project

## Introduction
This project aims to predict Temperature, specifically the mean temperature, for Munich, Germany, using various machine learning models. The dataset spans from 2000 to 2010 and includes multiple weather features. The project culminates in a Streamlit app that provides temperature predictions using two models.

## Data Source
The data is sourced from the European Climate Assessment & Dataset (ECA&D). It contains daily weather observations for 18 European cities, but this project focuses on Munich.

## Project Structure

### 1. Data Loading and Exploration
- Loaded the dataset using pandas
- Explored basic statistics and data distribution
- Visualized relationships between variables using scatter plots and heatmaps

### 2. Data Preprocessing
- Selected relevant features for Munich
- Removed highly correlated features (temp_min and temp_max)
- Normalized data using MinMaxScaler
- Applied Principal Component Analysis (PCA) for dimensionality reduction

### 3. Data Splitting
- Split data into training (60%), validation (20%), and test (20%) sets

### 4. Model Building
Four regression models were implemented:
- Random Forest
- XGBoost
- Linear Regression
- Support Vector Regression

Models were trained on both the original scaled data and PCA-transformed data.

### 5. Model Evaluation
Models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²) score

### 6. Hyperparameter Tuning
- Performed Grid Search for Random Forest to find optimal hyperparameters

## Results

### Model Performance (Without PCA)
1. Random Forest:
   - MAE: 2.4959
   - RMSE: 3.1665
   - R²: 0.8301

2. XGBoost:
   - MAE: 2.6170
   - RMSE: 3.3581
   - R²: 0.8089

3. Linear Regression:
   - MAE: 3.4387
   - RMSE: 4.2395
   - R²: 0.6954

4. Support Vector Regression:
   - MAE: 2.6361
   - RMSE: 3.3463
   - R²: 0.8102

### Model Performance (With PCA)
1. Random Forest:
   - MAE: 2.7303
   - RMSE: 3.4670
   - R²: 0.7963

2. XGBoost:
   - MAE: 2.9698
   - RMSE: 3.7910
   - R²: 0.7564

3. Linear Regression:
   - MAE: 3.7925
   - RMSE: 4.7001
   - R²: 0.6256

4. Support Vector Regression:
   - MAE: 2.7256
   - RMSE: 3.4302
   - R²: 0.8006

### Best Model: Random Forest
After hyperparameter tuning:
- Best Parameters: {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}
- Performance on Validation Set:
  - MAE: 2.4680
  - RMSE: 3.1470
  - R²: 0.8321

## Conclusion
The Random Forest model performed best, explaining about 83% of the variance in mean temperature predictions for Munich. The model's RMSE of approximately 3.15°C indicates a reasonable level of accuracy for weather prediction tasks. The PCA results show slightly lower performance, suggesting that the original features contain important information for prediction.

## Deployment
The project is deployed as a Streamlit app, which uses both the Random Forest and XGBoost models to provide a predicted temperature range.

### How to Run the Streamlit App
1. Ensure you have Streamlit installed: `pip install streamlit`
2. Navigate to the project directory
3. Run the command: `streamlit run app.py`
4. The app will open in your default web browser
5. Input the required weather parameters
6. The app will display predicted temperature ranges from both models

## Future Work
- Incorporate time series analysis techniques
- Experiment with ensemble methods
- Collect and integrate more recent data
- Expand the model to predict other weather variables
- Enhance the Streamlit app with more features and visualizations

## Dependencies
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- streamlit

## How to Run the Full Project
1. Ensure all dependencies are installed
2. Load the dataset
3. Run the preprocessing steps
4. Train the models
5. Evaluate performance
6. Use the saved Random Forest and XGBoost models for predictions in the Streamlit app

The final models are saved as 'random_forest_model_main.pkl' and 'xgb_model.pkl' in the 'pickles' folder.
#### Video Walkthrorugh of deployment
<video src=''></video>



