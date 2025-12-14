import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# 1. Load Clean Data
df = pd.read_csv('data/cleaned_data.csv')

# 2. Split Input (X) and Output (y)
X = df.drop(columns=['total_runs_x'])
y = df['total_runs_x']

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("Training Started... This might take a minute.")

# 3. Create Processing Pipeline
# Convert Team Names/Cities to Numbers using OneHotEncoding
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# 4. Create Model Pipeline
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', StandardScaler()),
    ('step3', XGBRegressor(n_estimators=1000, learning_rate=0.2, max_depth=12, random_state=1))
])

# 5. Train
pipe.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipe.predict(X_test)
print(f"Model Accuracy (R2 Score): {r2_score(y_test, y_pred)}")

# 7. Save Model
pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("Success! Model saved as 'pipe.pkl'")
