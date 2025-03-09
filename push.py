from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

dataset = LeRobotDataset(repo_id="Pi-robot/put_meat_clean", root="/mnt/vdb/lerobot/so100_up7")
# dataset.configure()
dataset.push_to_hub()