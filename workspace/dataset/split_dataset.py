import re
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile

from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.parent

@dataclass
class Data:
    path: Path
    name: str

    def __hash__(self):
        return hash(self.name)

def split_dataset():
    X_DIR = ROOT_DIR / "MS-SNSD" / "NoisySpeech_training"

    X = list(X_DIR.glob("*.wav"))

    x_train, x_val = train_test_split(X, test_size=0.2, random_state=42)

    y_train = set()
    y_val = set()

    data_dest = ROOT_DIR / "workspace" / "dataset" / "data"

    X_train_dir = data_dest / "train" / "input"
    Y_train_dir = data_dest / "train" / "output"
    X_val_dir = data_dest / "val" / "input"
    Y_val_dir = data_dest / "val" / "output"

    for file in X_train_dir.glob("*"):
        file.unlink()
    for file in Y_train_dir.glob("*"):
        file.unlink()
    for file in X_val_dir.glob("*"):
        file.unlink()
    for file in Y_val_dir.glob("*"):
        file.unlink()

    try:
        X_train_dir.rmdir()
        Y_train_dir.rmdir()
        X_val_dir.rmdir()
        Y_val_dir.rmdir()
    except FileNotFoundError:
        pass

    X_train_dir.mkdir(parents=True, exist_ok=True)
    Y_train_dir.mkdir(parents=True, exist_ok=True)
    X_val_dir.mkdir(parents=True, exist_ok=True)
    Y_val_dir.mkdir(parents=True, exist_ok=True)

    number_regex = re.compile(r"\d+\.\d+_\w+")
    clean_regex = re.compile(r"clnsp\d+")

    def get_file(file):
        out_name = number_regex.search(file.name).group()
        name = clean_regex.search(file.name).group()
        return Data(
            path=ROOT_DIR / "MS-SNSD" / "CleanSpeech_training" / f"{name}.wav",
            name=f"{out_name}.wav"
        )

    for data in x_train:
        y_file = get_file(data)
        y_train.add(y_file)
    for data in x_val:
        y_file = get_file(data)
        y_val.add(y_file)

    for data in tqdm(x_train, desc="Copying noisy train data"):
        copyfile(data, X_train_dir / data.name)
    for data in tqdm(y_train, desc="Copying clean train data"):
        copyfile(data.path, Y_train_dir / data.name)
    for data in tqdm(x_val, desc="Copying noisy val data"):
        copyfile(data, X_val_dir / data.name)
    for data in tqdm(y_val, desc="Copying clean val data"):
        copyfile(data.path, Y_val_dir / data.name)


split_dataset()
