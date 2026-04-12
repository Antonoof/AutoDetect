import os
import gc
import torch
import random
import shutil

from glob import glob
from ultralytics import YOLO

class AutoDetect:

    def __init__(
            self,
            train, val,
            model='x',
            warmup=False,
            inference_speed=-1,
            seed=42
    ):
        self.model = model
        self.target_dir = 'result'
        os.makedirs(self.target_dir, exist_ok=True)

        self.set_seed(seed)
        self.create_yaml(train, val)
        print("✔️ yaml file created! Ready to learn.")

    def set_seed(self, seed):
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def parse_classes(self, folders):
        classes = set()
        for folder in folders:
            txt_files = glob(os.path.join(folder, 'labels', '*.txt'))

            for txt_path in txt_files:
                try:
                    with open(txt_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) >= 1:
                                class_id = int(parts[0])
                                classes.add(class_id)
                except Exception as e:
                    print(f"Error {txt_path}: {e}")

        return len(classes), [f'class_{i}' for i in range(len(classes))]

    def create_yaml(self, train, val):
        if isinstance(train, str):
            train = [train]

        if isinstance(val, str):
            val = [val]

        folders = train + val
        nc, names = self.parse_classes(folders)

        train_text = ''.join(f'\n  - {t}/images' for t in train)
        val_text = ''.join(f'\n  - {v}/images' for v in val)

        data_yaml = f"""
train:{train_text}
val:{val_text}

nc: {nc}
names: {names}"""

        self.yaml = self.target_dir + '/data.yaml'
        with open(self.yaml, 'w') as file:
            file.write(data_yaml)

    def random_params(self):
        params = {
            'patience': 100,
            'optimizer': 'SGD',
            'lr0': round(0.001 * random.uniform(0.8, 1.2), 6),
            'lrf': round(0.001 * random.uniform(0.8, 1.2), 6),
            'weight_decay': round(0.0003 * random.uniform(0.7, 1.3), 6),
            'cos_lr': True,
            'save_period': max(1, int(10 + random.uniform(-2, 3))),
            'workers': 4,
            'close_mosaic': max(0, int(5 + random.uniform(-1, 2))),
            'hsv_h': round(random.uniform(0.0, 0.03), 4),
            'hsv_s': round(random.uniform(0.0, 0.03), 4),
            'hsv_v': round(random.uniform(0.0, 0.03), 4),
            'flipud': 0.0,
            'fliplr': round(random.uniform(0.4, 0.6), 2),
            'translate': round(random.uniform(0.005, 0.015), 4),
            'scale': round(random.uniform(0.2, 0.3), 3),
            'shear': round(random.uniform(0.03, 0.07), 3),
            'mixup': round(random.uniform(0.03, 0.07), 3),
            'cutmix': round(random.uniform(0.07, 0.13), 3),
            'warmup_epochs': max(0, int(3 + random.uniform(-1, 2))),
            'warmup_momentum': round(random.uniform(0.95, 1.0), 3),
            'augment': True,
            'conf': round(0.001 * random.uniform(0.5, 1.5), 6),
            'iou': round(random.uniform(0.25, 0.35), 2),
            'freeze': max(0, int(4 + random.uniform(-1, 2))),
        }

        return params

    def learn_MVP(self, model):
        model_config = (
            ('yolov8', 1280),
            ('yolo11', 960),
            ('yolo26', 1024)
        )

        for i, (model_name, imgsz) in enumerate(model_config):
            model_i = YOLO(model_name + model + '.pt')
            print(f"⌛⌛⌛   Start Learn model: {model_name + model}   ⌛⌛⌛")

            params = self.random_params()
            model_i.train(
                    data=self.yaml,
                    epochs=1,
                    imgsz=imgsz,
                    seed=self.seed,
                    **params
            )

            shutil.move('runs/detect/train/weights/best.pt', f'{self.target_dir}/best_{i}.pt')
            shutil.rmtree('runs')

            del model_i
            gc.collect()
            torch.cuda.empty_cache()

    def fit(self):
        self.learn_MVP(model=self.model)


# ad = AutoDetect(
#     train=train_data,
#     val=val_data,
#     model='n',
#     warmup=False,
#     inference_speed=-1,
# )

# ad.fit()