import pandas as pd
import pickle as pk
import time


def drop_duplicates_and_count(df):
    initial_count = len(df)
    df = df.dropna(subset=['sub_category'])
    final_count = len(df)
    # drop out rate
    drop_out_rate = (initial_count - final_count) / initial_count * 100

    # most of the "Moving Target" records are waypoint, which have limited activity semantics.
    # You can also keep them if you want to include more detailed movement information.
    df = df[df['sub_category'] != 'Moving Target'] # around 100 records in the whole dataset.
    print(f"Initial rows: {initial_count}, After dropping null values in 'sub_category': {final_count}, Drop Number: {initial_count - final_count}, Drop Out Rate: {drop_out_rate:.2f}%")
    return df


# here, we set the interval to 10 minutes, you can adjust it as needed
def keep_first_visit_per_interval(df):

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date
    # map the timestamp to the corresponding time interval, e.g., 10-minute intervals
    df['time_interval'] = df['timestamp'].dt.floor('10min')

    df = df.sort_values(by=['uid', 'visit_order'])

    df = df.drop_duplicates(subset=['uid', 'time_interval'], keep='first')

    df = df.reset_index(drop=True)

    # drop the visit order column, as it is no longer meaningful after keeping only the first visit per interval
    # drop the timestamp column, as it is no longer needed after mapping to time intervals
    df = df.drop(columns=['visit_order', "timestamp"], errors='ignore')

    return df


def get_poi_id_text_label(df):
    df = df.copy()
    poi_df = df[["poi_id", "sub_category"]].drop_duplicates()
    poi_id_mapping = {}

    for sub_cat in poi_df['sub_category'].unique():
        sub_cat_pois = poi_df[poi_df['sub_category'] == sub_cat]['poi_id'].unique()
        sub_cat_pois = sorted(list(sub_cat_pois))
        new_ids = [f"{sub_cat} #{i}" for i in range(len(sub_cat_pois))]
        poi_id_mapping.update(dict(zip(sub_cat_pois, new_ids)))

    df['poi_id'] = df['poi_id'].map(poi_id_mapping)
    return df


def generate_trajectory_prompts(df):
    """
    This function takes a preprocessed DataFrame with columns ['uid', 'poi_id', 'sub_category', 'date', 'time_interval']
    and generates a dictionary mapping each user ID (uid) to a list of daily activity prompts. Each prompt describes the
    activities of that day in a human-readable format.
    e.g., "Activities at 2026-03-01: Convenience Store #1 at 08:00, Cafe #2 at 09:10, Restaurant #5 at 12:00."
    """
    df = df.copy()

    # 1. make sure the data is sorted by user and time
    df = df.sort_values(by=['uid', 'time_interval'])

    # 2. extract the time part from the time interval for better readability in the prompt
    df['time_str'] = df['time_interval'].dt.strftime('%H:%M:%S')

    # 3. filter out consecutive visits to the same POI, only keep the first one in a sequence of identical POIs
    is_poi_changed = df['poi_id'] != df.groupby(['uid', 'date'])['poi_id'].shift(1)
    df = df[is_poi_changed]

    # 4. concatenate the POI label and time into a single string for each event, e.g., "Restaurant #5 at 12:00"
    df['event_str'] = df['poi_id'] + " at " + df['time_str']

    # 5. aggregate by user and date, concatenate the events of the day into a single string separated by commas
    df["uid"] = "user_" + df["uid"].astype(str)
    daily_traj = df.groupby(['uid', 'date'])['event_str'].apply(lambda x: ', '.join(x)).reset_index()

    # 6. construct the prompt for each day, e.g., "Activities at 2026-03-01: Convenience Store #1 at 08:00, Cafe #2 at 09:10, Restaurant #5 at 12:00."
    daily_traj['prompt'] = "Activities at " + daily_traj['date'].astype(str) + ": " + daily_traj['event_str'] + "."

    # 7. aggregate the daily prompts into a list for each user, resulting in a dictionary {uid: [prompt1, prompt2, ...]}
    daily_text_trajs = daily_traj.groupby('uid')['prompt'].apply(list).to_dict()

    return daily_text_trajs


def run_preprocessing(df, dataset_name):
    start_time = time.time()
    print(f"Processing {dataset_name} dataset...")
    df = drop_duplicates_and_count(df)
    df = keep_first_visit_per_interval(df)
    df = get_poi_id_text_label(df)
    # keep only the necessary columns for the final output
    keep_columns = ['uid', 'poi_id', 'sub_category', 'date', 'time_interval']
    text_trajs = generate_trajectory_prompts(df[keep_columns])
    end_time = time.time()
    print(f"Number of users in {dataset_name} dataset: {len(text_trajs)}")
    print(f"Finished processing {dataset_name} dataset in {end_time - start_time:.2f} seconds.")
    print("-" * 150)
    return  text_trajs


df1=  pd.read_csv("anonymized_olympic.csv")
df2 = pd.read_csv("anonymized_covid.csv")
df3 = pd.read_csv("anonymized_typhoon.csv")
df4 = pd.read_csv("anonymized_normal_day.csv")

print("-" * 150)
text_trajs_olympic = run_preprocessing(df1, "olympic")
text_trajs_covid = run_preprocessing(df2, "covid")
text_trajs_typhoon = run_preprocessing(df3, "typhoon")
text_trajs_normal_day =  run_preprocessing(df4, "normal_day")


# save the processed prompts to pickle files for later use
pk.dump(text_trajs_olympic, open("activities_list_coordinate_olympic", "wb"))
pk.dump(text_trajs_covid, open("activities_list_coordinate_covid", "wb"))
pk.dump(text_trajs_typhoon, open("activities_list_coordinate_typhoon", "wb"))
pk.dump(text_trajs_normal_day, open("activities_list_coordinate_normal_day", "wb"))


import pandas as pd

files = [
    'anonymized_covid.csv',
    'anonymized_normal_day.csv',
    'anonymized_olympic.csv',
    'anonymized_typhoon.csv',
]

dfs = [pd.read_csv(f, usecols=['sub_category']) for f in files]
combined = pd.concat(dfs, ignore_index=True)
result = combined.drop_duplicates().sort_values('sub_category').reset_index(drop=True)
result.to_csv('subcategories.csv', index=False)
print(f"共 {len(result)} 条唯一 sub_category")