from pathlib import Path

data_dir="./dataset"

for x in Path(data_dir).iterdir():
    print(x)


for x in Path(data_dir).rglob('*'):
    if x.is_dir():  # 检查是否为目录
        print(x)