# -*- coding: utf-8 -*-
"""
긴 이미지(128x256x3: 좌/우 128x128x3) 자동 분류 스크립트
- NORMAL_DIR의 정상 이미지로 임계값 τ 자동 캘리브레이션
- TEST_DIR의 이미지들을 좌/우로 분할해 임베딩 코사인 유사도로 양품/불량 분류
- 결과를 CSV로 저장

필요 패키지:
  pip install transformers pillow torch numpy
"""

import os
import csv
import json
import glob
from typing import Tuple, List, Dict, Optional

import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# =========================
# CONFIG: 여기만 수정하세요
# =========================
NORMAL_DIR = r"./Images/normal"   # 정상(캘리브레이션) 긴 이미지 폴더
TEST_DIR   = r"./Images/test"     # 테스트(분류) 긴 이미지 폴더
MODEL_ID   = "meta-llama/Llama-3.2-Vision-Instruct"  # 임베딩용 비전 모델 ID
OUT_CSV    = None  # None이면 TEST_DIR/result.csv 로 저장

# 이미지 크기/전처리
EXPECTED_HEIGHT = 128
EXPECTED_WIDTH  = 256
ALLOW_RESIZE    = True   # 128x256이 아니어도 강제 리사이즈

# 캘리브레이션 하위 분위수(q): 정상-정상 유사도 분포의 하위 q를 τ로 설정
CALIB_QUANTILE  = 0.10   # 10% 권장(오탐 줄이기)

# 확장자
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# =========================
# 이미지 유틸
# =========================
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def split_long_image(img: Image.Image,
                     expect_hw: Tuple[int, int]=(EXPECTED_HEIGHT, EXPECTED_WIDTH),
                     allow_resize: bool=ALLOW_RESIZE) -> Tuple[Image.Image, Image.Image]:
    """
    긴 이미지(128x256x3)를 좌/우 128x128로 분할.
    - allow_resize=True이면, 입력이 정확히 128x256이 아니더라도 강제로 리사이즈.
    """
    H_exp, W_exp = expect_hw
    if allow_resize and (img.size != (W_exp, H_exp)):
        img = img.resize((W_exp, H_exp))
    w, h = img.size
    assert w % 2 == 0, "이미지 가로는 2로 나누어 떨어져야 합니다."
    left = img.crop((0, 0, w // 2, h))
    right = img.crop((w // 2, 0, w, h))
    return left, right

# =========================
# 임베딩(LLM/VLM 비전 타워)
# =========================
class VisionEmbedder:
    def __init__(self, model_id: str, device: Optional[str]=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray:
        """
        비전 타워의 마지막 히든 평균 풀링 임베딩을 사용.
        """
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True
        )
        h = out.last_hidden_state.mean(dim=1)  # [B, D]
        h = torch.nn.functional.normalize(h, dim=-1)
        return h.squeeze(0).float().cpu().numpy()

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

# =========================
# 비교/캘리브레이션/분류
# =========================
def list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for ext in EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def compare_long_image(path: str, embedder: VisionEmbedder,
                       expect_hw=(EXPECTED_HEIGHT, EXPECTED_WIDTH),
                       allow_resize=ALLOW_RESIZE) -> Dict:
    img = load_image(path)
    left, right = split_long_image(img, expect_hw=expect_hw, allow_resize=allow_resize)
    e1 = embedder.embed(left)
    e2 = embedder.embed(right)
    s = cosine(e1, e2)
    return {"path": path, "similarity": s}

def calibrate_tau_from_folder(folder: str, embedder: VisionEmbedder,
                              quantile: float=CALIB_QUANTILE,
                              expect_hw=(EXPECTED_HEIGHT, EXPECTED_WIDTH),
                              allow_resize=ALLOW_RESIZE) -> Dict:
    """
    정상 긴 이미지 폴더에서 좌/우 쌍 유사도 분포의 하위 q-분위수를 τ로 설정.
    """
    files = list_images(folder)
    sims: List[float] = []
    for f in files:
        try:
            r = compare_long_image(f, embedder, expect_hw=expect_hw, allow_resize=allow_resize)
            sims.append(r["similarity"])
        except Exception as e:
            print(f"[WARN] 캘리브레이션 실패: {f} - {e}")
    if not sims:
        return {"error": "no_files_or_embeddings", "folder": folder}
    sims = np.array(sims, dtype=np.float32)
    tau = float(np.quantile(sims, quantile))
    return {
        "tau": tau,
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "q": quantile,
        "n": int(len(sims)),
        "folder": folder
    }

def classify_folder(folder: str, embedder: VisionEmbedder, tau: float,
                    out_csv: Optional[str]=None,
                    expect_hw=(EXPECTED_HEIGHT, EXPECTED_WIDTH),
                    allow_resize=ALLOW_RESIZE) -> Dict:
    """
    폴더 내 모든 긴 이미지를 좌/우 비교 후 CSV 저장.
    CSV 헤더: path,similarity,label
    """
    files = list_images(folder)
    if not files:
        return {"error": "no_files", "folder": folder}

    rows = []
    for f in files:
        try:
            r = compare_long_image(f, embedder, expect_hw=expect_hw, allow_resize=allow_resize)
            s = r["similarity"]
            label = "양품" if s >= tau else "불량"
            rows.append({"path": f, "similarity": s, "label": label})
        except Exception as e:
            print(f"[WARN] 분류 실패: {f} - {e}")

    if out_csv is None:
        out_csv = os.path.join(folder, "result.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=["path", "similarity", "label"])
        wr.writeheader()
        wr.writerows(rows)

    # 간단 통계
    y = [r["label"] for r in rows]
    cnt_good = y.count("양품")
    cnt_bad = y.count("불량")

    return {"saved": out_csv, "n": len(rows), "good": cnt_good, "bad": cnt_bad}

# =========================
# 메인: F5로 바로 실행
# =========================
def main():
    print("[INFO] 모델 로드 중...", MODEL_ID)
    embedder = VisionEmbedder(model_id=MODEL_ID)

    # 1) 캘리브레이션
    print("[INFO] 캘리브레이션 폴더:", NORMAL_DIR)
    cal = calibrate_tau_from_folder(NORMAL_DIR, embedder)
    if "error" in cal:
        raise RuntimeError(f"캘리브레이션 실패: {cal}")
    tau = cal["tau"]
    print("[INFO] 캘리브레이션 결과:")
    print(json.dumps(cal, ensure_ascii=False, indent=2))

    # 2) 테스트 분류
    print("[INFO] 테스트 폴더:", TEST_DIR)
    out_csv = OUT_CSV if OUT_CSV else os.path.join(TEST_DIR, "result.csv")
    res = classify_folder(TEST_DIR, embedder, tau, out_csv=out_csv)
    if "error" in res:
        raise RuntimeError(f"분류 실패: {res}")

    print("[INFO] 분류 완료:")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[INFO] 결과 CSV: {res['saved']}")

if __name__ == "__main__":
    main()