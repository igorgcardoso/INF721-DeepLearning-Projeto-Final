import re
from pathlib import Path
from shutil import copyfile

from tqdm import tqdm

ROOT = Path(__file__).parent

data_dir = ROOT / "data"
clean_dir = data_dir / "clean"

train_dir = data_dir / "train"
train_noisy_dir = train_dir / "input"
train_clean_dir = train_dir / "output"

# test_dir = data_dir / "test"

val_dir = data_dir / "val"
val_noisy_dir = val_dir / "input"
val_clean_dir = val_dir / "output"

number_regex = re.compile(r"\d+\.\d+_\w+")
clean_regex = re.compile(r"clnsp\d+")

train_noisy = sorted(train_noisy_dir.glob("*.wav"))
val_noisy = sorted(val_noisy_dir.glob("*.wav"))

# file = train_noisy[0]

# print(file.stem)
# print(clean_regex.search(file.stem).group())
# print(number_regex.search(file.stem).group())

for file in tqdm(train_noisy):
    filename = file.stem
    file_number = number_regex.search(filename).group()
    clean_name = clean_regex.search(filename).group()
    clean_file = clean_dir / f"{clean_name}.wav"
    copyfile(clean_file, train_clean_dir / f"{file_number}.wav")

for file in tqdm(val_noisy):
    filename = file.stem
    file_number = number_regex.search(filename).group()
    clean_name = clean_regex.search(filename).group()
    clean_file = clean_dir / f"{clean_name}.wav"
    copyfile(clean_file, val_clean_dir / f"{file_number}.wav")

