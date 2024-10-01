# Temperature_prediction

# Language_Identification

## Project Structure


```plaintext
|──data_folder
|   |──extracted_data.csv
|──pickles
|   |──language_label_mapping.pkl
|   |──tfidf_vectorizer.pkl
|   |──final_naivebayes_model.pkl
|──scripts
|   |──__init__.py
|   |──cleaning.py
|   |──preprocess.py
|   |──model_building.py
|   |──hyperparameter_tuning.py
|   |──testing1.py
|   |──testing2.py
|──language_identification_notebook.ipynb
|──data.csv
|──language_identification_app.py
|──README.md
|──requirements.txt
|
```


## About Data
* Data Name : Wili 2018
* Source: Kaggle [https://www.kaggle.com/datasets/sharansmenon/wili-2018?select=data.csv]
* The main language data. Contains about 200k instances for 235 languages

# STEPS
### Load Data
### Data Cleaning
* Remove Escape strings at the end of the Data

### Data Preprocessing
* Select only data label to Afrikaans, Spanish, German and Alemannic German only
* Feature Extraction (TF-IDF- Term Frequency-Inverse Document Frequency)

### Model Training and Evaluation
* Split Data
  - Splitted Data to Train, Test and Validation set 60%, 20% and 20% Respectively
* Train Data on 4 Model

```plaintext
Random Forest Model Evaluation:
Mean Absolute Error (MAE): 2.4876
Root Mean Square Error (RMSE): 3.1607
R-squared (R²): 0.8307
--------------------------------------------------
XGBoost Model Evaluation:
Mean Absolute Error (MAE): 2.6170
Root Mean Square Error (RMSE): 3.3581
R-squared (R²): 0.8089
--------------------------------------------------
Linear Regression Model Evaluation:
Mean Absolute Error (MAE): 3.4387
Root Mean Square Error (RMSE): 4.2395
R-squared (R²): 0.6954
--------------------------------------------------
Support Vector Regression Model Evaluation:
Mean Absolute Error (MAE): 2.6361
Root Mean Square Error (RMSE): 3.3463
R-squared (R²): 0.8102
```

```plaintext
### RESULT with PCA
Random Forest Model Evaluation:
Mean Absolute Error (MAE): 2.7396
Root Mean Square Error (RMSE): 3.4623
R-squared (R²): 0.7968
--------------------------------------------------
XGBoost Model Evaluation:
Mean Absolute Error (MAE): 2.9698
Root Mean Square Error (RMSE): 3.7910
R-squared (R²): 0.7564
--------------------------------------------------
Linear Regression Model Evaluation:
Mean Absolute Error (MAE): 3.7925
Root Mean Square Error (RMSE): 4.7001
R-squared (R²): 0.6256
--------------------------------------------------
Support Vector Regression Model Evaluation:
Mean Absolute Error (MAE): 2.7256
Root Mean Square Error (RMSE): 3.4302
R-squared (R²): 0.8006
```
- Base on the Reason Below we will be selecting RANDOM FOREST REGRESSOR for this Project
  - *Results*: Naive bayes happen to return the best result for MAE, RMSE, RSQUARED

### Hyperparameter Tuning
To further increase the metrics result of our naive bayes model we use GridSearch CV(Cross Validation)
To further improve the model and we got the following Result

- Comparing the two results above we notice a little significant improvement
```plaintext
Grid Search Hyperparameter Tuning for Random Forest completed in 5705.76 seconds
Best Params for Random Forest: {'max_depth': 30, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}
Random Forest Model Evaluation On Testing Subset:
Mean Squared Error: 2.5558
Root Mean Squared Error: 3.2471
R-squared: 0.8227
Random Forest Model Evaluation On Validation Subset:
Mean Absolute Error: 2.4816
Root Mean Squared Error: 3.1474
R-squared: 0.8321
Random Forest Model Saved to Pickle Folder
  ```



### Deployment
- The Random FOREST MODEL was serialized using pickle and the model was deploy using streamlit
#### Running The App
To Run the app use the command below
  
`streamlit run l.py`

### Unit Test
* There are two testing file in the scripts folder, testing1.py and testing2.py
* testing1.py was design with unittest approach
  To run testing 1 use the command below
  `python run testing1.py`
* testing2.py was design with pytest approach
  `pytest testing2.py` provided you have pytest installed

#### Video Walkthrorugh of deployment
<video src=''></video>



