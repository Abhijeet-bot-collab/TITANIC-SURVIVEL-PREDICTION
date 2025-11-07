
# Titanic Survival Prediction

## Project Overview

This project analyzes the historical Titanic dataset to predict passenger survival using various machine learning techniques. The process includes exploratory data analysis, handling missing values, feature engineering, data preprocessing, training and evaluating multiple classification models, hyperparameter tuning, and model interpretation. The goal is to identify the key factors influencing survival and build a predictive model.

## How to Run the Code

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd titanic-survival-prediction
    ```
3.  **Install Required Libraries:**
    Make sure you have Python installed. Install the necessary libraries using pip:
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
    ```
4.  **Obtain the Dataset:**
    Download the `train.csv` and `test.csv` files from the Kaggle Titanic competition ([https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)) and place them in a directory named `PROJECT HUB/project/TITANIC SURVIVAL PREDICTION (ML)` within your Google Drive, or modify the file paths in the notebook accordingly.
5.  **Run the Jupyter Notebook:**
    Open and run the `Titanic_Survival_Prediction.ipynb` notebook (or the equivalent notebook containing the code) in a Jupyter environment (like Jupyter Notebook, JupyterLab, or Google Colab). Execute the cells sequentially.

The notebook contains the full workflow, from data loading and cleaning to model training, evaluation, interpretation, and submission file generation.

## Results Summary

- **Key Factors for Survival:** Analysis revealed that 'Sex' (female), 'Pclass' (higher classes), and 'Title' (e.g., 'Mrs', 'Miss', 'Master') were the most significant predictors of survival.
- **Best Performing Model:** The XGBoost classifier, after hyperparameter tuning, achieved the best performance with a cross-validation accuracy of approximately 83.5%.
- **Feature Importance:** The tuned XGBoost model highlighted 'Title_Mr', 'Sex_female', and 'Pclass_3' as the most important features for prediction.
- **Submission:** A `submission.csv` file is generated, containing 'PassengerId' and the predicted 'Survived' status for the test dataset.

## Files Generated

- `train_cleaned.csv`: The preprocessed training data.
- `test_cleaned.csv`: The preprocessed test data.
- `best_model.joblib`: The trained XGBoost model saved using joblib.
- `submission.csv`: The submission file in the format required by the Kaggle competition.
