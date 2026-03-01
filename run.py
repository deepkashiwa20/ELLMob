from simulator.traj_generator import *
from simulator.person_anonymized import *
import pickle
import csv
from datetime import datetime, date, timedelta

def ensure_dates(activities: list, dates: list) -> list:
    existing = [
        m.group(1)
        for entry in activities
        for m in [re.search(r'Activities at (\d{4}-\d{2}-\d{2})', entry)]
        if m
    ]
    updated = activities.copy()
    for d in dates:
        if d not in existing:
            updated.append(f'Activities at {d}: ')
    def extract_date(entry: str) -> str:
        m = re.search(r'Activities at (\d{4}-\d{2}-\d{2})', entry)
        return m.group(1) if m else ''
    updated.sort(key=extract_date)
    return updated

def filter_train(item_list):
    """Return activity entries whose date falls from 2020 02 07 through 2020 04 06 inclusive."""
    filtered = []
    for item in item_list:
        m = re.search(r"Activities at (\d{4})-(\d{2})-(\d{2}):", item)
        if not m:
            continue
        year, month, day = map(int, m.groups())
        dt = datetime(year, month, day)
        start = datetime(2020, 2, 7)
        end   = datetime(2020, 4, 6)
        if start <= dt <= end:
            filtered.append(item)
    return filtered


def filter_test(item_list):
    """Return activity entries whose date falls from 2020 02 07 through 2020 04 06 inclusive."""
    filtered = []
    for item in item_list:
        m = re.search(r"Activities at (\d{4})-(\d{2})-(\d{2}):", item)
        if not m:
            continue
        year, month, day = map(int, m.groups())
        if year == 2020 and month == 4 and 7 <= day <= 13:
            filtered.append(item)
    return filtered

if __name__ == "__main__":
    planner = DayPlanner()
    routinelist = open("trajectory_data.pkl", 'rb')
    routine = pickle.load(routinelist)
    file_path = 'user_list.csv'

    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        user_list = [row[0] for row in reader]
    trajectories = {}
    start_date = date(2020, 4, 7)
    num_days = 7
    days_to_check = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]
    for key in user_list:
        P = Person(key)
        print(key)
        print("current:" + str(len(trajectories)))
        P.test_routine_list = ensure_dates(filter_test(routine[key]), days_to_check)
        P.train_routine_list = filter_train(routine[key])
        trajectories[str(P.name)] = planner.plan_new_day(P)
        P.name = key
        with open('result.pkl', 'wb') as f:
            pickle.dump(trajectories, f)
    print("done")
