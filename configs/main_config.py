from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, Optional, Tuple, Union
import numpy as np

from pathlib import Path
d = Path().resolve()#.parent
general_path = str(d) + "/standalone_examples/Aloha"

@dataclass
class MainConfig:
    tuning: bool = False
    headless: bool = False
    reward_mode: int = 1
    cup_usd_path: str = general_path + "/assets/objects/bowl.usd"
    jetbot_asset_path: str = general_path + "/assets/aloha/ALOHA_with_sensor_02.usd"
    # room_usd_path: str = general_path + "/assets/scenes/sber_kitchen/sber_kitchen_ft.usd"
    room_usd_path: str = general_path + "/assets/scenes/scenes_sber_kitchen_for_BBQ/kitchen.usd"
    camera_paths: str = "/jetbot/fl_link4/visuals/realsense/husky_rear_left"
    goal_image_path: str = general_path + '/img/goal.png'
    train_log_dir: str = general_path + "/models/SAC"
    load_policy: str = general_path + "/models/SAC/last_chance7_410000_steps"
    log_path: str = general_path + "/log.txt"
    camera_usd_local_path = general_path + '/assets/aloha/aloloha_v03_cameras.usd'
    camera_image_saved_path = general_path + "/img/"
    training_mode:int = 0