# Student_feedback_ML_Project


Below is My Live Application Link

https://studentfeedbackmlproject-itca8kqd3bethjudejpqys.streamlit.app/


📊 Employee Feedback Prediction App (Streamlit Deployment)

This repository hosts a machine learning-powered web application designed to predict employee job feedback categories (e.g., 'Good', 'Average', 'Poor') based on key demographic and employment features. The primary goal of this project is to demonstrate the end-to-end deployment of a pre-trained model using Streamlit, enabling real-time, interactive predictions.

The model was trained on the Employee_clean.csv dataset and is persisted as a pickled file (New.pkl).

🌟 Features

Interactive Input: A user-friendly interface built with Streamlit allows users to input employee features (Age, Experience, Salary, Department).

Real-time Prediction: Uses a pre-trained New.pkl model to instantly generate predictions.

Model Agnostic: The core deployment logic can be easily adapted to any scikit-learn compatible classification model.

Dependency Management: Includes a requirements.txt file for easy environment setup and deployment.

⚙️ Technologies Used

Python: The core language for the application.

Streamlit: For creating the interactive web application interface.

scikit-learn: For the underlying machine learning model (New.pkl).

Pandas & NumPy: For data handling and input structuring.

🚀 How to Run Locally

Follow these steps to get a local copy of the project running on your machine.

Prerequisites

You need Python installed on your system.

1. Clone the repository

git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
cd your-repo-name


2. Prepare the Model and Data Files

Ensure the following two files are placed directly in the project root directory:

New.pkl (The pickled machine learning model)

Employee_clean.csv (The training data file, used for context)

3. Setup the Environment

Install the required Python packages using the provided requirements.txt file:

pip install -r requirements.txt


4. Run the Streamlit App

Execute the main application file:

streamlit run app.py


The application will automatically open in your default web browser at http://localhost:8501.

📈 Model and Data Information

The model predicts the Feedback class based on the following features:

Feature

Type

Encoding in App

Age

Numerical (Slider)

Direct Input

Experience

Numerical (Slider)

Direct Input

Salary

Numerical (Input Box)

Direct Input

Department

Categorical (Select Box)

One-hot encoded (0, 1, 2, 3, 4) in the model

The department mapping used in the application is:

0: Sales

1: IT

2: HR

3: Finance

4: R&D
(Note: If your model uses a different encoding, you must update the DEPARTMENT_MAP in app.py.)
