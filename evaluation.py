import numpy as np
import scipy.stats
import pickle as pk
from math import sin, cos, asin, sqrt, radians
import pickle
import re
from datetime import datetime
from logger import get_logger
import logging
import argparse
import re
from typing import List, Tuple, Union

with open("data/subcategory_to_topscategory.pickle", "rb") as f:
    sub_to_top = pickle.load(f)


class DataLoader(object):
    def __init__(self):
        pass

    @staticmethod
    def clean_and_extract_locations(route):
        try:
            route = route.replace("  ", " ")
            route = route.replace(" ,", ",")
            route = route.replace(" .", ".")
            route = route.strip('.')
        except:
            route = route[0]
            route = route.replace("  ", " ")
            route = route.replace(" ,", ",")
            route = route.replace(" .", ".")
            route = route.strip('.')
        return route

    @staticmethod
    def load_location_map(data_path):
        logger.debug("Loading location map file...")
        location_map = {}
        data = pk.load(open(data_path, "rb"))
        for location_name in data:
            lng, lat = float(data[location_name][1][2]), float(data[location_name][1][3])
            location_map[location_name] = (lng, lat)
        logger.debug(f"Loaded {len(location_map)} locations from {data_path}")
        return location_map

    @staticmethod
    def load_category_map(data_path):
        logger.debug("Loading category map file...")
        category_mapping = {}
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                cat = line.strip()
                if cat:
                    category_mapping[cat] = idx
        logger.debug(f"Loaded {len(category_mapping)} categories from {data_path}")
        return category_mapping

    @staticmethod
    def load_trajectory_data(data_path):
        """
        Load trajectory data, it could be generated or real trajectory data. \n
        Args:
            data_path: str, path to the trajectory data file
        Returns:
            fake_trajs: list, generated trajectories
            real_trajs: list, real trajectories
        """
        logger.debug("Loading and cleaning generation and real trajectory data...")
        genlist = pickle.load(open(data_path, 'rb'))
        reallist = pickle.load(open("groundtruth.pkl", 'rb'))

        success_users = set()
        all_fake = {}
        all_real = {}
        for m in genlist:
            if m in reallist:
                for date, text_traj in genlist[m]['results'].items():
                    dt = datetime.strptime(date, '%Y-%m-%d')
                    # please adjust the date range according to the actual date range of your data, here we only keep the trajectories in April 7-13, 2020 for covid event dataset.
                    if dt.month != 4 or (dt.month == 4 and (dt.day < 7 or dt.day > 13)):
                        continue
                    all_fake[f"{m}_{date}"] = text_traj

                for date, text_traj in reallist[m]['reals'].items():

                    dt = datetime.strptime(date, '%Y-%m-%d')
                    if dt.month != 4 or (dt.month == 4 and (dt.day < 7 or dt.day > 13)):
                        continue
                    all_real[f"{m}_{date}"] = text_traj

                success_users.add(m)

        fake_trajs = [DataLoader.clean_and_extract_locations(all_fake[k]) for k in all_fake.keys()]
        real_trajs = [DataLoader.clean_and_extract_locations(all_real[k]) for k in all_real.keys()]

        logger.debug(
            f"Succcessfully loaded {len(success_users)}/{len(genlist)} users' {len(all_fake)} generated trajectories and {len(all_real)} real trajectories \n")

        return fake_trajs, real_trajs


class Evaluation(object):

    def __init__(self, locations_map, categories_map, logger=None):

        self.logger = logger
        self.locations_map = locations_map
        self.categories_map = categories_map

    @staticmethod
    def parse_activities(activities_str):
        pattern = re.compile(
            r'(?:^|,)\s*' 
            r'(.+? at \d{1,2}:\d{2}(?::\d{2})?\.?)'
        )
        return pattern.findall(activities_str)

    def extract_category_seq_single(self, trajs):
        all_categories = []
        no_hash_count = 0
        no_hash_examples = []
        for trajectory_str in trajs:
            header, sep, activities_part = trajectory_str.partition(':')
            if not sep:
                continue
            matches = self.parse_activities(activities_part)

            categories = []
            for match in matches:
                if '#' in match:
                    category_part = match.split('#')[0]
                    categories.append(category_part)
                else:
                    categories.append(match)
            all_categories.extend(categories)
        return all_categories

    def extract_duration_seq(self, p):
        def to_seconds(ts):
            parts = ts.split(':')
            try:
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    return h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    h, m = map(int, parts)
                    return h * 3600 + m * 60
            except:
                self.logger.error(f"Invalid time format: {ts}")
                pass
            return 0

        # Regex to match HH:MM or HH:MM:SS
        time_pattern = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?)")
        d = []
        for u in p:
            # Create a raw text containing all events for this day
            if isinstance(u, str):
                raw = u
            elif isinstance(u, (list, tuple)):
                raw = ", ".join(str(x) for x in u)
            else:
                raw = str(u)
            # Extract timestamps in order
            times = [to_seconds(m.group(1)) for m in time_pattern.finditer(raw)]
            if len(times) < 2:
                # self.logger.warning("Not enough timestamps to compute differences.")
                continue
            # Compute consecutive differences
            diffs = [times[i] - times[i - 1] for i in range(1, len(times))]
            d.append(diffs)

        # Flatten and scale differences
        flat_scaled = [round(diff) for day in d for diff in day]
        negative_count = sum(1 for diff in flat_scaled if diff < 0)
        if negative_count > 0:
            self.logger.debug(
                f"Found {negative_count} negative time differences in {len(flat_scaled)} total time differences.")
        else:
            self.logger.debug(f"No negative time differences found in {len(flat_scaled)} total time differences.")

        return flat_scaled

    def extract_lnglat_seq(self, trajectory_str, data_type="real"):
        """
        e.g.: trajectory = "Activities at 2020-09-01: Shipping, Freight, and Material Transportation Service#1111 at 14:20, Other Place#22222 at 16:00"
        data_type: str, "real" or "generated"
        """
        activities_str = trajectory_str.split(": ")[1]
        activities = self.parse_activities(activities_str)
        # 首先构建所有有效点
        valid_points = []
        for activity in activities:
            # 解析地点和时间
            if activity and "at" in activity:
                loc_name, time = activity.rsplit(" at ", 1)
                if loc_name in self.locations_map:
                    lng, lat = self.locations_map[loc_name]
                    valid_points.append((lng, lat))

        return valid_points

    def arr_to_distribution(self, arr, Min, Max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """
        num_of_samples = len(arr)
        distribution, base = np.histogram(arr, bins=bins, range=(Min, Max))  # base 记录了 bin 的边界
        max_part = np.array([len(arr[arr > Max])], dtype='int64')
        min_part = np.array([len(arr[arr < Min])], dtype='int64')
        distribution = np.hstack((distribution, max_part))
        distribution = np.hstack((min_part, distribution))

        assert num_of_samples == distribution.sum(), f"num_of_samples: {num_of_samples}, distribution.sum(): {distribution.sum()}"
        return distribution, base[:-1]

    def get_js_divergence(self, p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum() + 1e-9)
        p2 = p2 / (p2.sum() + 1e-9)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
        return js

    def geodistance(self, lng1, lat1, lng2, lat2):
        """
        Calculate the greater circle distance in `km` between two points on the earth
        Args:
            lng1 (float): longitude of the first point
            lat1 (float): latitude of the first point
            lng2 (float): longitude of the second point
            lat2 (float): latitude of the second point

        Returns:
            distance (float): distance of the two points in km
        """
        lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance = 2 * asin(sqrt(a)) * 6371 * 1000
        distance = round(distance / 1000, 3)
        return distance

    # SD
    def calc_distance_one_step_jsd(self, real_trajs, generated_trajs, max_distance=100, granularity=0.5):
        """
        calculate the Jensen-Shanon Divergence of  `one step distance` between generated and real trajectories
        Args:
            real_trajs (list): ground truth trajs
            generated_trajs (list): generated trajs
            max_distance (float): maximum distance for last bins, default is `100 km`
            granularity (float): granularity of the distance bins, in `km`, default is `0.5 km`
        Returns:
            JSD (float): Jensen-Shanon Divergence
        """
        self.logger.debug("Calculating SD (distance one step) ...")
        generated_trajs = [self.extract_lnglat_seq(traj, data_type="generated") for traj in generated_trajs]
        real_trajs = [self.extract_lnglat_seq(traj, data_type="real") for traj in real_trajs]
        generated_trajs = [traj for traj in generated_trajs if len(traj) > 1]
        real_trajs = [traj for traj in real_trajs if len(traj) > 1]

        generated_distances = [self.geodistance(lnglat[0], lnglat[1], traj[index][0], traj[index][1]) for traj in
                               generated_trajs for index, lnglat in enumerate(traj[1:])]
        real_distances = [self.geodistance(lnglat[0], lnglat[1], traj[index][0], traj[index][1]) for traj in real_trajs
                          for index, lnglat in enumerate(traj[1:])]


        MIN = 0
        MAX = max_distance
        bins = int((MAX - MIN) / granularity)
        r_list, _ = self.arr_to_distribution(np.array(real_distances), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(generated_distances), MIN, MAX, bins)
        JSD = self.get_js_divergence(r_list, f_list)
        self.logger.debug(
            f"Number of bins: {bins}, with spaital granularity of {granularity} km and max distance of {max_distance} km")
        self.logger.debug(
            f"Min/Max one step distance of generated trajs: {min(generated_distances)}km, {max(generated_distances)}km")
        self.logger.debug(f"Min/Max one step distance of real trajs: {min(real_distances)}km, {max(real_distances)}km")
        self.logger.debug(f"SD (distance one step): {JSD:.4f}\n")

        return JSD

    # SI
    def calc_duration_jsd(self, reals, generations, granularity=10.0):
        """
        calculate the Jensen-Shanon Divergence of `duration` between generated and real trajectories
        Args:
            reals (list): real trajectories
            generations (list): generated trajectories
            granularity (float): granularity of the time bins, in `minutes`
        Returns:
            JSD (float): Jensen-Shanon Divergence
        """

        self.logger.debug("Calculating SI (duration JSD) ...")
        g = self.extract_duration_seq(generations)
        r = self.extract_duration_seq(reals)
        g = [x / 60 for x in g]
        r = [x / 60 for x in r]

        MIN = 0
        MAX = 24 * 60  # 1440 minutes in a day
        bins = int((MAX - MIN) / granularity)
        self.logger.debug(f"Number of bins per day: {bins}, with time granularity of {granularity} minutes")

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        g_list, _ = self.arr_to_distribution(np.array(g), MIN, MAX, bins)
        JSD = self.get_js_divergence(r_list, g_list)

        self.logger.debug(f"SI (duration JSD): {JSD:.4f}\n")
        return JSD


    def calc_category_jsd(self, generated_trajs, real_trajs):
        """
        calculate the Jensen-Shanon Divergence of `category` between generated and real trajectories
        Args:
            generated_trajs (list): generated trajectories
            real_trajs (list): real trajectories
        Returns:
            JSD (float): Jensen-Shanon Divergence
        """
        self.logger.debug("Calculating CI (category JSD) ...")

        generated_categories = self.extract_category_seq_single(generated_trajs)
        real_categories = self.extract_category_seq_single(real_trajs)
        self.logger.debug(f"Number of category entries in generated trajs: {len(generated_categories)}")
        self.logger.debug(f"Number of category entries in real trajs: {len(real_categories)}")

        all_tops = sorted({
            top
            for tops in sub_to_top.values()
            for top in tops
            if isinstance(top, str)
        })
        top_index = {top: i for i, top in enumerate(all_tops)}

        def count_top_categories(sub_cats):
            counts = np.zeros(len(top_index), dtype=int)
            for sub in sub_cats:
                sub = sub.strip()
                if sub == "Cafe":
                    sub = "Café"
                elif sub == "Pet Cafe":
                    sub = "Pet Café"
                tops = sub_to_top.get(sub)
                if not tops:
                    continue
                top = tops[0]
                counts[top_index[top]] += 1
            return counts

        top_counts1 = count_top_categories(generated_categories)
        top_counts2 = count_top_categories(real_categories)

        print(top_counts1)
        print(top_counts2)
        JSD = self.get_js_divergence(top_counts1, top_counts2)

        self.logger.debug(f"Category JSD: {JSD:.4f}\n")

        return JSD

    # Spatial Grid Activity JSD
    def calc_sg_act_jsd(self, generated_trajs, real_trajs, grid_count=100, lon_range=(139.50, 139.90),
                        lat_range=(35.50, 35.82)):
        """
        calculate the Jensen-Shanon Divergence of `spatial grid activity` between generated and real trajectories
        Args:
            p1 (list): generated trajectories
            p2 (list): real trajectories
            grid_count (int): number of grids, default is 10000
            lon_range (tuple): longitude range, default is (139.50, 139.90)
            lat_range (tuple): latitude range, default is (35.50, 35.82)
        Returns:
            JSD (float): Jensen-Shanon Divergence
        """

        def map_visits_to_grid(p):
            grid_side = int(np.sqrt(grid_count))
            counts = np.zeros(grid_count + 1, dtype=int)
            for lon, lat in p:
                if lon_range[0] <= lon <= lon_range[1] and lat_range[0] <= lat <= lat_range[1]:
                    x_norm = (lon - lon_range[0]) / (lon_range[1] - lon_range[0])
                    y_norm = (lat - lat_range[0]) / (lat_range[1] - lat_range[0])
                    gx = min(int(x_norm * grid_side), grid_side - 1)
                    gy = min(int(y_norm * grid_side), grid_side - 1)
                    idx = gy * grid_side + gx
                else:
                    idx = grid_count
                counts[idx] += 1
            return list(counts)
        self.logger.debug("Calculating SDG (Spatial distribution (grid scale) JSD) ...")

        generated_points = [self.extract_lnglat_seq(traj, data_type="generated") for traj in generated_trajs]
        real_points = [self.extract_lnglat_seq(traj, data_type="real") for traj in real_trajs]
        generated_points = [point for traj in generated_points for point in traj]
        real_points = [point for traj in real_points for point in traj]

        self.logger.debug(f"Number of points in generated trajs: {len(generated_points)}")
        self.logger.debug(f"Number of points in real trajs: {len(real_points)}")

        self.logger.debug(
            f"Grid count: {grid_count}, with grid size of {int(np.sqrt(grid_count))}x{int(np.sqrt(grid_count))}")

        p1 = np.array(map_visits_to_grid(generated_points))[:grid_count]
        p2 = np.array(map_visits_to_grid(real_points))[:grid_count]
        top25 = np.argsort(p2)[-25:][::-1]
        selected_p1 = p1[top25]
        selected_p2 = p2[top25]
        JSD = self.get_js_divergence(selected_p1, selected_p2)
        self.logger.debug(f"Spatial Grid JSD: {JSD:.4f}\n")
        return JSD


def double_filter_multi_visit_trajectories(
        fake_trajs: List[List[str]],
        real_trajs: List[List[str]]
) -> Tuple[List[List[str]], List[List[str]]]:
    filtered_fake = []
    filtered_real = []
    for fake, real in zip(fake_trajs, real_trajs):
        if isinstance(real, str) and real.count('#') > 1:
            filtered_real.append(real)
    for fake, real in zip(fake_trajs, real_trajs):
        if isinstance(fake, str) and fake.count('#') > 1:
            filtered_fake.append(fake)
    return filtered_fake, filtered_real


def single_filter_multi_visit_trajectories(fake_trajs, real_trajs):
    filtered_fake = []
    filtered_real = []

    for real in real_trajs:
        if isinstance(real, str) and real.count('#') >= 1:
            filtered_real.append(real)

    for fake in fake_trajs:
        if isinstance(fake, str) and fake.count('#') >= 1:
            filtered_fake.append(fake)

    return filtered_fake, filtered_real

def label_visit_flags(
        fake_trajs: List[Union[str, List[str]]],
        real_trajs: List[Union[str, List[str]]]
) -> Tuple[List[int], List[int]]:

    def label(traj: Union[str, List[str]]) -> int:
        text = " ".join(traj) if isinstance(traj, list) else traj
        return 1 if '#' in text else 0

    pred_flags = [label(t) for t in fake_trajs]
    true_flags = [label(t) for t in real_trajs]

    return pred_flags, true_flags


def compute_classification_metrics(
        true_flags: List[int],
        pred_flags: List[int]
) -> dict:
    TP = TN = FP = FN = 0
    for t, p in zip(true_flags, pred_flags):
        if t == 1 and p == 1:
            TP += 1
        elif t == 0 and p == 0:
            TN += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 1 and p == 0:
            FN += 1

    # 计算指标
    total = len(true_flags)
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--log_level', type=str, default='DEBUG', help='log level')
    argparser.add_argument('--model', type=str, default='ours', help='model name')
    argparser.add_argument('--year', type=str, default='2020', help='year')
    args = argparser.parse_args()

    if args.log_level == 'DEBUG':
        logger = get_logger("llm_generator", log_file="debug.log", level=logging.DEBUG)
    if args.log_level == 'INFO':
        logger = get_logger("llm_generator", log_file="debug.log", level=logging.INFO)

    trajdata_path = "result_first.pkl"
    location_map = DataLoader.load_location_map("data/location_map_covid.pkl")
    category_map = DataLoader.load_category_map("data/subcategories.csv")
    fake_trajs, real_trajs = DataLoader.load_trajectory_data(trajdata_path)


    doublecheckin_fake_trajs, doublecheckin_real_trajs = double_filter_multi_visit_trajectories(fake_trajs, real_trajs)
    onecheckin_fake_trajs, onecheckin_real_trajs = single_filter_multi_visit_trajectories(fake_trajs, real_trajs)

    evaluator = Evaluation(location_map, category_map, logger=logger)
    duration_jsd = evaluator.calc_duration_jsd(doublecheckin_real_trajs, doublecheckin_fake_trajs, granularity=10)
    distance_step_jsd = evaluator.calc_distance_one_step_jsd(doublecheckin_real_trajs, doublecheckin_fake_trajs,
                                                             max_distance=100, granularity=1)
    category_jsd = evaluator.calc_category_jsd(onecheckin_fake_trajs, onecheckin_real_trajs)
    sg_act_jsd = evaluator.calc_sg_act_jsd(onecheckin_fake_trajs, onecheckin_real_trajs, grid_count=100,
                                           lon_range=(139.50, 139.90), lat_range=(35.50, 35.82))

    pred_flags, true_flags = label_visit_flags(fake_trajs, real_trajs)
    metrics = compute_classification_metrics(true_flags, pred_flags)

    logger.info("********** RESULTS SUMMARY ***********")
    logger.info(f"Model: {args.model}, Year: {args.year}")
    logger.info(f"SI (One step Duration JSD): \t {duration_jsd:.4f}")
    logger.info(f"SD (One step distance JSD): \t {distance_step_jsd:.4f}")
    logger.info(f"CI (Category JSD): \t\t {category_jsd:.4f}")
    logger.info(f"SG (Spatial Grid JSD): \t {sg_act_jsd:.4f}")
    logger.info(f"Go out or not: \t {metrics}")
    logger.info("**************************************")