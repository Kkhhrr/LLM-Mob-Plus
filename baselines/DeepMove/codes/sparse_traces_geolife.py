import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import datetime

# trackintel
from trackintel.io.dataset_reader import read_geolife
from trackintel.preprocessing.triplegs import generate_trips
from trackintel.analysis.tracking_quality import temporal_tracking_quality, _split_overlaps
import trackintel as ti


# --- Helper Functions (inlined from utils.py) ---

def _filter_user(df, min_thres, mean_thres):
    consider = df.loc[df["quality"] != 0]
    if (consider["quality"].min() > min_thres) and (consider["quality"].mean() > mean_thres):
        return df

def _get_tracking_quality(df, window_size):
    weeks = (df["finished_at"].max() - df["started_at"].min()).days // 7
    start_date = df["started_at"].min().date()

    quality_list = []
    # construct the sliding week gdf
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        # the total df for this time window
        cAll_gdf = df.loc[(df["started_at"] >= curr_start) & (df["finished_at"] < curr_end)]
        if cAll_gdf.shape[0] == 0:
            continue
        total_sec = (curr_end - curr_start).total_seconds()

        quality_list.append([i, cAll_gdf["duration"].sum() / total_sec])
    ret = pd.DataFrame(quality_list, columns=["timestep", "quality"])
    ret["user_id"] = df["user_id"].unique()[0]
    return ret

def calculate_user_quality(sp, trips, file_path, quality_filter):
    trips["started_at"] = pd.to_datetime(trips["started_at"]).dt.tz_localize(None)
    trips["finished_at"] = pd.to_datetime(trips["finished_at"]).dt.tz_localize(None)
    sp["started_at"] = pd.to_datetime(sp["started_at"]).dt.tz_localize(None)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"]).dt.tz_localize(None)

    # merge trips and staypoints
    print("starting merge", sp.shape, trips.shape)
    sp["type"] = "sp"
    trips["type"] = "tpl"
    df_all = pd.concat([sp, trips])
    df_all = _split_overlaps(df_all, granularity="day")
    df_all["duration"] = (df_all["finished_at"] - df_all["started_at"]).dt.total_seconds()
    print("finished merge", df_all.shape)
    print("*" * 50)

    if "min_thres" in quality_filter:
        end_period = datetime.datetime(2017, 12, 26)
        df_all = df_all.loc[df_all["finished_at"] < end_period]

    print(len(df_all["user_id"].unique()))

    # get quality
    total_quality = temporal_tracking_quality(df_all, granularity="all")
    # get tracking days
    total_quality["days"] = (
        df_all.groupby("user_id").apply(lambda x: (x["finished_at"].max() - x["started_at"].min()).days).values
    )
    # filter based on days
    user_filter_day = (
        total_quality.loc[(total_quality["days"] > quality_filter["day_filter"])]
        .reset_index(drop=True)["user_id"]
        .unique()
    )

    sliding_quality = (
        df_all.groupby("user_id")
        .apply(_get_tracking_quality, window_size=quality_filter["window_size"])
        .reset_index(drop=True)
    )

    filter_after_day = sliding_quality.loc[sliding_quality["user_id"].isin(user_filter_day)]

    if "min_thres" in quality_filter:
        # filter based on quanlity
        filter_after_day = (
            filter_after_day.groupby("user_id")
            .apply(_filter_user, min_thres=quality_filter["min_thres"], mean_thres=quality_filter["mean_thres"])
            .reset_index(drop=True)
            .dropna()
        )

    filter_after_user_quality = filter_after_day.groupby("user_id", as_index=False)["quality"].mean()

    print("final selected user", filter_after_user_quality.shape[0])
    filter_after_user_quality.to_csv(file_path, index=False)
    return filter_after_user_quality["user_id"].values

# --- DeepMove Specific Helper Functions ---

def convert_time_to_id(t):
    """
    将 datetime 对象转换为 DeepMove 使用的 48-bin 时间 ID。
    工作日: 0-23, 周末: 24-47
    """
    if t.weekday() in [0, 1, 2, 3, 4]:  # 周一到周五
        return t.hour
    else:  # 周六、周日
        return t.hour + 24

# --- Main Processing Logic ---

def process_geolife_for_deepmove(epsilon=50, num_samples=2, hour_gap=12, session_min_len=5, user_min_sessions=10):
    """
    完整处理流程：从原始 Geolife 数据生成 DeepMove 可用的 geolife.pk 文件。
    """
    # --- 1. 路径设置 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "geolife")
    raw_geolife_path = os.path.join(data_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    
    # --- 2. 读取数据并生成位置 ---
    print("Step 1: Reading raw Geolife data and generating locations...")
    
    sp_time_file = os.path.join(data_dir, "sp_time_temp_geolife.csv")
    if Path(sp_time_file).is_file():
        print(f"Loading pre-generated staypoints from {sp_time_file}...")
        sp_time = pd.read_csv(sp_time_file, parse_dates=['started_at', 'finished_at'])
        locations_file = os.path.join(data_dir, f"locations_geolife.csv")
        locs = pd.read_csv(locations_file).set_index('location_id')
    else:
        pfs, _ = read_geolife(raw_geolife_path, print_progress=True)
        pfs, sp = pfs.as_positionfixes.generate_staypoints(
            gap_threshold=24 * 60, include_last=True, print_progress=True, dist_threshold=200, time_threshold=30, n_jobs=-1
        )
        sp = sp.as_staypoints.create_activity_flag(method="time_threshold", time_threshold=25)
        
        quality_path = os.path.join(data_dir, "quality")
        quality_file = os.path.join(quality_path, "geolife_slide_filtered.csv")
        if not os.path.exists(quality_path):
            os.makedirs(quality_path)
            
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
        sp, tpls, trips = generate_trips(sp, tpls, add_geometry=False)
        quality_filter = {"day_filter": 50, "window_size": 10}
        valid_user = calculate_user_quality(sp.copy(), trips.copy(), quality_file, quality_filter)
        sp = sp.loc[sp["user_id"].isin(valid_user)]
        sp = sp.loc[sp["is_activity"] == True]

        sp, locs = sp.as_staypoints.generate_locations(
            epsilon=epsilon, num_samples=num_samples, distance_metric="haversine", agg_level="dataset", n_jobs=-1
        )
        sp = sp.loc[~sp["location_id"].isna()].copy()
        locs = locs[~locs.index.duplicated(keep="first")]
        filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]
        filtered_locs.as_locations.to_csv(os.path.join(data_dir, f"locations_geolife.csv"))
        
        sp = sp[["user_id", "started_at", "finished_at", "location_id"]]
        sp_time = sp.sort_values(by=['user_id', 'started_at']).reset_index(drop=True)
        sp_time.to_csv(sp_time_file, index=False)
        
    print("Step 1 finished.")

    # --- 3. 构建会话 (Session) ---
    print("Step 2: Building sessions for each user...")
    sessions = {}
    
    for user_id, user_df in tqdm(sp_time.groupby('user_id')):
        user_df = user_df.sort_values('started_at')
        
        user_sessions = {}
        session_id_counter = 0
        current_session = []
        
        for i in range(len(user_df)):
            loc = int(user_df.iloc[i]['location_id'])
            time_obj = user_df.iloc[i]['started_at']
            
            current_session.append([loc, convert_time_to_id(time_obj)])
            
            if i < len(user_df) - 1:
                time_gap_hours = (user_df.iloc[i+1]['started_at'] - user_df.iloc[i]['finished_at']).total_seconds() / 3600
                if time_gap_hours > hour_gap:
                    if len(current_session) >= session_min_len:
                        user_sessions[session_id_counter] = current_session
                        session_id_counter += 1
                    current_session = []
            else:
                if len(current_session) >= session_min_len:
                    user_sessions[session_id_counter] = current_session
        
        if len(user_sessions) >= user_min_sessions:
             sessions[user_id] = user_sessions

    print(f"Step 2 finished. Found {len(sessions)} valid users.")

    # --- 4. 构建 DeepMove 所需的最终数据结构 ---
    print("Step 3: Building final data structure for DeepMove...")
    
    user_ids = list(sessions.keys())
    user_enc = {old_uid: new_uid for new_uid, old_uid in enumerate(user_ids)}
    
    all_locs = set()
    for user_id in sessions:
        for sess_id in sessions[user_id]:
            for loc, _ in sessions[user_id][sess_id]:
                all_locs.add(loc)
    loc_ids = list(all_locs)
    loc_enc = {old_loc_id: new_loc_id for new_loc_id, old_loc_id in enumerate(loc_ids, 1)}
    loc_enc['unk'] = 0

    data_neural = {}
    uid_list = {}
    vid_list = {original_id: [new_id, 0] for original_id, new_id in loc_enc.items() if original_id != 'unk'}
    vid_list['unk'] = [0, -1]
    
    for old_uid, user_sessions in tqdm(sessions.items()):
        new_uid = user_enc[old_uid]
        
        encoded_sessions = {}
        for sess_id, session_data in user_sessions.items():
            encoded_session = []
            for loc, tim in session_data:
                new_loc = loc_enc[loc]
                encoded_session.append([new_loc, tim])
                vid_list[loc][1] += 1
            encoded_sessions[sess_id] = encoded_session
            
        session_keys = list(encoded_sessions.keys())
        split_point = int(len(session_keys) * 0.8)
        train_ids = session_keys[:split_point]
        test_ids = session_keys[split_point:]
        
        if not train_ids or not test_ids:
            continue
            
        data_neural[new_uid] = {
            'sessions': encoded_sessions,
            'train': train_ids,
            'test': test_ids
        }
        uid_list[old_uid] = [new_uid, len(encoded_sessions)]

    vid_lookup = {}
    for old_loc, new_loc in loc_enc.items():
        if old_loc != 'unk' and old_loc in locs.index:
            point = locs.loc[old_loc].geometry
            vid_lookup[new_loc] = [point.x, point.y]
    
    parameters = {
        'hour_gap': hour_gap,
        'session_min_len': session_min_len,
        'user_min_sessions': user_min_sessions,
        'epsilon': epsilon,
        'num_samples': num_samples
    }
    
    final_dataset = {
        'data_neural': data_neural,
        'vid_list': vid_list,
        'uid_list': uid_list,
        'vid_lookup': vid_lookup,
        'parameters': parameters
    }
    
    print(f"Final valid user count: {len(data_neural)}")
    print(f"Final valid location count: {len(vid_list) - 1}")
    
    # --- 5. 保存为 .pk 文件 ---
    output_path = os.path.join(data_dir, "geolife.pk")
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
        
    print(f"Step 3 finished. Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Geolife dataset for DeepMove model.")
    parser.add_argument("--epsilon", type=int, default=50, help="Epsilon for DBSCAN to detect locations (in meters).")
    parser.add_argument("--num_samples", type=int, default=2, help="Minimum number of samples for a location.")
    parser.add_argument("--hour_gap", type=int, default=12, help="Minimum time gap in hours to define a new session.")
    parser.add_argument("--session_min_len", type=int, default=5, help="Minimum number of points in a valid session.")
    parser.add_argument("--user_min_sessions", type=int, default=10, help="Minimum number of sessions for a valid user.")
    args = parser.parse_args()

    process_geolife_for_deepmove(
        epsilon=args.epsilon, 
        num_samples=args.num_samples,
        hour_gap=args.hour_gap,
        session_min_len=args.session_min_len,
        user_min_sessions=args.user_min_sessions
    )