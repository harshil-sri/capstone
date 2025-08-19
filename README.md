Automated Salary Prediction Model
A machine learning project designed to predict new hire salaries, aiming to promote fairness and reduce subjective bias in compensation decisions. This tool provides a data-driven approach for HR departments to standardize salary offers based on historical data.

📜 Project Overview
This project addresses the business challenge of inconsistent and potentially biased salary offers during the hiring process. By leveraging a dataset of historical applicant profiles, this model learns the complex relationships between a candidate's qualifications, experience, and their expected salary (CTC).

The core objective is to provide an objective, data-driven tool that can recommend a fair salary, helping to ensure pay equity and consistency for candidates with comparable qualifications. The model was built using a Random Forest Regressor, which has proven to be highly effective for this prediction task.

✨ Key Features
Data-Driven Salary Prediction: Recommends salaries based on historical data, not subjective judgment.

Bias Reduction: Helps mitigate unconscious bias in compensation.

Standardized Process: Creates a consistent and repeatable method for determining salary offers.

High Accuracy: The model demonstrates a very high level of accuracy in its predictions.

⚙️ Model & Technical Details
The prediction model is a Random Forest Regressor implemented using Python's scikit-learn library.

The process involves:

Data Preprocessing: Cleaning the dataset by handling missing values, removing irrelevant identifiers, and converting categorical data (like 'Department', 'Role') into a numerical format using one-hot encoding.

Model Training: The cleaned data is split into training (80%) and testing (20%) sets. The Random Forest model is then trained on the training data.

Evaluation: The model's performance is measured against the unseen test data to ensure its predictions are accurate and reliable.

📊 Model Performance
The model achieved excellent performance metrics, indicating a high degree of accuracy and reliability:

R-squared (R²) Score: 0.9996

This means the model can explain 99.96% of the variance in the salary data, which is an outstanding fit.

Mean Absolute Error (MAE): 11609.15

On average, the model's salary prediction is off by approximately $11,609.

Root Mean Squared Error (RMSE): 22999.39

This is another measure of error, more sensitive to large mistakes.

🚀 How to Run the Project
To run this project on your local machine, follow these steps.

Prerequisites
Make sure you have Python and the following libraries installed:

Pandas

NumPy

Scikit-learn

You can install them using pip:

pip install pandas numpy scikit-learn

Installation & Usage
Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

Place the dataset:
Ensure the dataset file, expected_ctc.csv, is in the same directory as the Python script.

Run the script:
Execute the script from your terminal to train the model and see the performance evaluation.

python "major project.py"

The script will load the data, train the model, and print the final performance metrics to the console.
