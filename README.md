# 🏏 IPL Cricket Score Predictor

A Machine Learning powered web application that predicts the final score of the first innings in an IPL match. The model considers current match conditions like wickets, overs, and recent performance to generate accurate predictions.

## 📸 Demo
## 🚀 Features
* **Real-time Prediction:** Predicts the final score based on the current situation (Overs, Wickets, Current Score).
* **Advanced Logic:** Uses "Runs in last 5 overs" and exact ball-by-ball calculation for higher accuracy.
* **Interactive UI:** Built with **Streamlit**, featuring a dynamic video background.
* **High Accuracy:** Uses **XGBoost Regressor**, achieving an R2 score of ~96% on test data.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-learn, XGBoost
* **Data Processing:** Pandas, NumPy
* **Dataset:** IPL Ball-by-Ball Dataset (2008 - 2024)

## 📂 Project Structure
```text
ipl-score-predictor/
│
├── data/
│   ├── matches.csv        # Raw match summaries
│   ├── deliveries.csv     # Ball-by-ball data
│   └── cleaned_data.csv   # Processed data for training
│
├── background.mp4         # Background video for the app
├── pipe.pkl               # Trained Machine Learning Model
├── preprocess.py          # Script for data cleaning & feature engineering
├── train.py               # Script for model training (XGBoost)
├── app.py                 # Main Streamlit Application
└── README.md              # Project Documentation




⚙️ Installation & Setup
1. Clone the Repository
Bash

git clone https://github.com/Aashiskr/Cricket_score_prediction.git
cd IPL-Score-Predictor
2. Install Dependencies
Make sure you have Python installed. Run the following command:

Bash

pip install pandas numpy scikit-learn streamlit xgboost
3. Data Preparation (Optional)
If you want to retrain the model from scratch:

Download matches.csv and deliveries.csv from Kaggle.

Place them inside the data/ folder.

Run the preprocessing script:

Bash

python preprocess.py
4. Train the Model
To generate a fresh pipe.pkl file:

Bash

python train.py
5. Run the App
Launch the web interface:

Bash

streamlit run app.py
📊 How It Works
Data Extraction: The app loads historical IPL data.

Processing: It calculates key metrics like Current Run Rate, Wickets Left, and Runs in Last 5 Overs.

Prediction: The XGBoost model analyzes these metrics against 15+ years of match history to predict the final total.

🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.
