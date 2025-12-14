import pandas as pd

# 1. Load Data
match_df = pd.read_csv('data/matches.csv')
delivery_df = pd.read_csv('data/deliveries.csv')

print("Original Match Data:", match_df.shape)
print("Original Delivery Data:", delivery_df.shape)

# 2. Calculate Total Score per Inning
total_score_df = delivery_df.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()

# We only predict for the 1st Innings
total_score_df = total_score_df[total_score_df['inning'] == 1]

# Merge total score back to match data
match_df = match_df.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

# 3. Clean Team Names
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

# Replace old team names with current ones
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

# Filter to keep only consistent teams
match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

# 4. Merge Matches and Deliveries
match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]
delivery_df = match_df.merge(delivery_df, on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 1]

# 5. Feature Engineering

# Current Score (Cumulative Sum)
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()

# Wickets (Cumulative Sum)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna(0)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == 0 else 1)
delivery_df['wickets'] = delivery_df.groupby('match_id')['player_dismissed'].cumsum()

# --- CRITICAL FIX: Exact Overs Calculation ---
# Convert ball number into fraction (e.g., 0.1 becomes 0.166)
delivery_df['overs'] = delivery_df['over'] + (delivery_df['ball'] / 6)

# Runs in Last 5 Overs (Rolling Window)
groups = delivery_df.groupby('match_id')
match_ids = delivery_df['match_id'].unique()
last_five = []

for id in match_ids:
    # Sum of last 30 balls (5 overs)
    last_five.extend(groups.get_group(id)['total_runs_y'].rolling(window=30).sum().values.tolist())

delivery_df['runs_last_5'] = last_five

# 6. Final Data Selection
final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'current_score', 'wickets', 'overs', 'runs_last_5', 'total_runs_x']]

# Remove missing values (first 5 overs of every match will be NaN)
final_df.dropna(inplace=True)

# Shuffle data to remove bias
final_df = final_df.sample(final_df.shape[0])

print("Final Data Shape:", final_df.shape)
print(final_df.head())

# 7. Save Cleaned Data
final_df.to_csv('data/cleaned_data.csv', index=False)
print("Success! Data saved to 'data/cleaned_data.csv'")
