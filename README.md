# Personal Fitness Tracker - README

## Overview

The **Personal Fitness Tracker** is a Streamlit-based web application that predicts calories burned during exercise based on personal attributes. It utilizes a **Random Forest Regressor** model trained on fitness data.

## Features

- **User Input Form**: Enter age, BMI, duration, heart rate, body temperature, and gender.
- **Data Preprocessing**: Merges and cleans exercise and calorie datasets.
- **Machine Learning Model**: Predicts calories burned using a trained **Random Forest Regressor**.
- **Insights Section**: Compares user data with historical data and provides fitness insights.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fitness-tracker.git
   cd fitness-tracker
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Data Requirements

- `calories.csv` and `exercise.csv` must be placed in the project directory.
- The datasets should include **User_ID, Age, Gender, Weight, Height, Duration, Heart Rate, Body Temperature, and Calories**.

## Usage

1. Enter your fitness details in the sidebar.
2. View predicted calories burned.
3. Get fitness insights and compare results with similar data.

## Dependencies

- `streamlit`
- `numpy`
- `pandas`
- `scikit-learn`

## Future Enhancements

- Add more fitness-related parameters.
- Integrate real-time activity tracking.

---

🚀 **Developed with Streamlit and Machine Learning**