import os
import cv2
import torch
import random
import optuna
import numpy as np

from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion as wbf

optuna.logging.set_verbosity(optuna.logging.WARNING)

class ADPredict:

    def __init__(
        self,
        models_dir,
        image_paths,
        val_images_path,
        output_dir="predictions",
        device='auto',
        seed=42,
        deterministic=False,
        optuna_trials=30,
        val_samples=200,
        wbf_conf_type="avg", # or max
        conf_range=(0.005, 0.5),
        iou_range=(0.3, 0.7),
        skip_box_range=(0.001, 0.05),
        model_weights=None
    ):
        self.models_dir = Path(models_dir)
        self.test_img_dir = Path(image_paths)
        self.val_img_dir = Path(val_images_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        self.deterministic = deterministic
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self.optuna_trials = optuna_trials
        self.val_samples = val_samples
        self.wbf_conf_type = wbf_conf_type
        self.conf_range = conf_range
        self.iou_range = iou_range
        self.skip_box_range = skip_box_range
        self.model_weights = model_weights

        self.models = []
        self.model_paths = []
        self.val_images = []
        self.val_lbl_dir = Path(".")
        self.best_params = None

        self._set_seed()
        self._setup_models()
        self._setup_val_data()

    def _set_seed(self):

        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            if self.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True

    def _setup_models(self):
        self.model_paths = sorted(list(self.models_dir.glob("*.pt")))

        if not self.model_paths:
            raise FileNotFoundError(f"❌ No models find in {self.models_dir}")
        
        print(f"Find {len(self.model_paths)} models: {[p.name for p in self.model_paths]}")
        self.models = [YOLO(p).to(self.device) for p in self.model_paths]

    def _setup_val_data(self):
        self.val_images = sorted(list(self.val_img_dir.glob("*.jpg")) + list(self.val_img_dir.glob("*.png")))
        if not self.val_images:
            raise FileNotFoundError(f"❌ Images not found in {self.val_img_dir}")

        # Search val labels
        candidates = [self.val_img_dir.parent / "labels", self.val_img_dir / "labels", self.val_img_dir]
        for cand in candidates:
            if cand.is_dir() and any(cand.glob("*.txt")):
                self.val_lbl_dir = cand
                break
        
        if not any(self.val_lbl_dir.glob("*.txt")):
            raise FileNotFoundError(f"❌ no labels in {self.val_img_dir}")

    def _extract_raw_boxes(self, model, img_path, conf_thr):
        results = model.predict(img_path, conf=conf_thr, iou=1.0, max_det=1000, verbose=False, device=self.device)
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0), 1, 1

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy()
        h, w = res.orig_shape

        # Normalization [0, 1] for WBF
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        return boxes, confs, clss, h, w

    def _load_val_gt(self, img_path):
        lbl_path = self.val_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            lbl_path = self.val_lbl_dir / f"{img_path.name}.txt"
            if not lbl_path.exists():
                return np.empty((0, 4))
            
        if lbl_path.stat().st_size == 0:
            return np.empty((0, 4))

        gt = np.loadtxt(lbl_path)
        if gt.size == 0: return np.empty((0, 4))
        if gt.ndim == 1: gt = gt.reshape(1, -1)
        if gt.shape[1] < 5: return np.empty((0, 4))

        img = cv2.imread(str(img_path))
        if img is None: return np.empty((0, 4))
        h, w = img.shape[:2]

        x_c = gt[:, 1] * w
        y_c = gt[:, 2] * h
        bw = gt[:, 3] * w
        bh = gt[:, 4] * h

        return np.stack([x_c - bw / 2, y_c - bh / 2, x_c + bw / 2, y_c + bh / 2], axis=1)

    @staticmethod
    def _box_iou(box1, box2):
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def _compute_f1(self, confs, wbf_iou, skip_thr):
        sample_imgs = self.val_images[:self.val_samples]
        tp = fp = fn = 0
        n_models = len(self.models)
        weights = self.model_weights if self.model_weights else [1.0] * n_models

        for img_p in sample_imgs:
            boxes_list, scores_list, labels_list = [], [], []
            ref_h, ref_w = 1, 1
            for i, model in enumerate(self.models):
                b, s, c, h, w = self._extract_raw_boxes(model, img_p, confs[i])
                ref_h, ref_w = h, w
                boxes_list.append(b.tolist() if len(b) else [])
                scores_list.append(s.tolist() if len(s) else [])
                labels_list.append(c.tolist() if len(c) else [])

            if all(len(b) == 0 for b in boxes_list):
                fn += len(self._load_val_gt(img_p))
                continue

            pred_b, _, _ = wbf(boxes_list, scores_list, labels_list, weights=weights, iou_thr=wbf_iou, skip_box_thr=skip_thr, conf_type=self.wbf_conf_type)
            gt = self._load_val_gt(img_p)
            if len(pred_b) == 0:
                fn += len(gt)
                continue

            matched = np.zeros(len(gt), dtype=bool)
            pred_abs = pred_b.copy()
            pred_abs[:, [0, 2]] *= ref_w
            pred_abs[:, [1, 3]] *= ref_h

            for pb in pred_abs:
                best_iou, best_idx = 0.0, -1
                for i, gb in enumerate(gt):
                    if not matched[i]:
                        iou_val = self._box_iou(pb, gb)
                        if iou_val > best_iou: best_iou, best_idx = iou_val, i
                if best_iou >= 0.5:
                    tp += 1; matched[best_idx] = True
                else:
                    fp += 1
            fn += (~matched).sum()

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        # f1 = (2 * tp) / (2 * tp + fp + fn)
        return (2 * prec * rec) / (prec + rec + 1e-9)

    def _objective(self, trial):
        n = len(self.models)
        confs = [trial.suggest_float(f"conf_{i}", self.conf_range[0], self.conf_range[1]) for i in range(n)]
        wbf_iou = trial.suggest_float("wbf_iou", self.iou_range[0], self.iou_range[1])
        skip_thr = trial.suggest_float("skip_thr", self.skip_box_range[0], self.skip_box_range[1])
        
        return self._compute_f1(confs, wbf_iou, skip_thr)

    def optimize(self):
        """Runs Optuna to find optimal ensemble thresholds."""
        print(f"🚀 start Optuna ({self.optuna_trials} trials)")
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.optuna_trials, show_progress_bar=True)
        self.best_params = study.best_params
        print(f"🏆 Best Params: {self.best_params}")

        return self.best_params

    def predict(self):
        """Generates and stores predictions on the test set."""
        if self.best_params is None:
            print("⚠️ Parameters are not optimized. Starting optimization...")
            self.optimize()

        n = len(self.models)
        confs = [self.best_params[f"conf_{i}"] for i in range(n)]
        wbf_iou = self.best_params["wbf_iou"]
        skip_thr = self.best_params["skip_thr"]
        weights = self.model_weights if self.model_weights else [1.0] * n

        test_imgs = sorted(list(self.test_img_dir.glob("*.jpg")) + list(self.test_img_dir.glob("*.png")))
        if not test_imgs:
            raise FileNotFoundError(f"❌ Test images not found in {self.test_img_dir}")
        
        for img_path in tqdm(test_imgs):
            boxes_list, scores_list, labels_list = [], [], []
            ref_h, ref_w = 1, 1
            for i, model in enumerate(self.models):
                b, s, c, h, w = self._extract_raw_boxes(model, img_path, confs[i])
                ref_h, ref_w = h, w
                boxes_list.append(b.tolist() if len(b) else [])
                scores_list.append(s.tolist() if len(s) else [])
                labels_list.append(c.tolist() if len(c) else [])

            if all(len(b) == 0 for b in boxes_list):
                continue

            pred_b, pred_s, pred_c = wbf(boxes_list, scores_list, labels_list, weights=weights, iou_thr=wbf_iou, skip_box_thr=skip_thr, conf_type=self.wbf_conf_type)

            out_txt = self.output_dir / f"{img_path.stem}.txt"
            with open(out_txt, "w") as f:
                if len(pred_b) > 0:
                    x_c = (pred_b[:, 0] + pred_b[:, 2]) / 2
                    y_c = (pred_b[:, 1] + pred_b[:, 3]) / 2
                    bw  = pred_b[:, 2] - pred_b[:, 0]
                    bh  = pred_b[:, 3] - pred_b[:, 1]
                    for cls, conf, xc, yc, wb, hb in zip(pred_c, pred_s, x_c, y_c, bw, bh):
                        f.write(f"{int(cls)} {conf:.6f} {xc:.6f} {yc:.6f} {wb:.6f} {hb:.6f}\n")

        print(f"🎉 Success! Results in {self.output_dir}")
        return self.output_dir

    def run(self):
        self.optimize()
        return self.predict()

    def __call__(self):
        return self.run()
    
# predictor = ADPredict(
#     models_dir=result_dir,
#     image_paths=test_images,
#     val_images_path=val_data,
# )

# predictor()