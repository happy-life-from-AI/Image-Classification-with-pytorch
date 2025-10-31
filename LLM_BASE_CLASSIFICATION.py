# -*- coding: utf-8 -*-
"""
긴 이미지(128x256x3: 좌/우 128x128x3) 자동 분류 스크립트 - Ollama VLM 기반
- NORMAL_DIR의 정상 이미지로 임계값 τ 자동 캘리브레이션(유사도 하위 q 분위수)
- TEST_DIR의 이미지들을 좌/우로 분할해 Ollama(멀티모달)에게 비교 요청
- Ollama가 반환한 similarity(0~1)를 사용해 양품/불량 분류
- 결과를 CSV로 저장

필요 패키지:
  pip install pillow numpy requests
사전 준비:
  - Ollama 설치 후 모델 다운로드 예:
      ollama pull llama3.2-vision
    (또는 llava)
"""

import os
import csv
import json
import glob
import base64
import re
from typing import Tuple, List, Dict, Optional

import numpy as np
import requests
from PIL import Image

# =========================
# CONFIG: 여기만 수정하세요
# =========================
NORMAL_DIR = r"./Images/normal"   # 정상(캘리브레이션) 긴 이미지 폴더
TEST_DIR   = r"./Images/test"     # 테스트(분류) 긴 이미지 폴더

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME  = "llama3.2-vision"   # 또는 "llava", 환경에 맞게 변경
TIMEOUT_SEC = 120                 # Ollama 요청 타임아웃

OUT_CSV     = None  # None이면 TEST_DIR/result.csv 로 저장

# 이미지 크기/전처리
EXPECTED_HEIGHT = 128
EXPECTED_WIDTH  = 256
ALLOW_RESIZE    = True   # 128x256이 아니어도 강제 리사이즈

# 캘리브레이션 하위 분위수(q): 정상 유사도 분포의 하위 q를 τ로 설정
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

def to_base64_jpeg(img: Image.Image, quality: int = 92) -> str:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# =========================
# Ollama 호출: 두 이미지 비교 프롬프트
# =========================
PROMPT_TEMPLATE = """
두 개의 이미지를 비교해 유사도 점수를 0.0~1.0 사이로 주고, '양품'(유사) 또는 '불량'(차이 존재)로 라벨링하라.
반드시 JSON 한 줄만 출력:
{ "similarity": <0~1>, "label": "양품|불량", "reason": "<간단한 근거>" }

채점 가이드(권장):
- 거의 동일: similarity 0.90~1.00
- 약간의 경미한 차이: 0.60~0.89
- 분명한 차이/결함: 0.00~0.59

이미지는 왼쪽(첫 번째), 오른쪽(두 번째) 순서로 제공된다.
JSON 외 텍스트는 출력하지 말 것.
""".strip()

def ollama_compare_two(left_img: Image.Image, right_img: Image.Image) -> Dict:
    """
    Ollama /api/generate 엔드포인트로 멀티모달 비교 요청.
    - 모델이 JSON 한 줄을 반환하도록 프롬프트 강제
    - 실패 시 보수적으로 similarity=0.0, label="불량"
    """
    url = f"{OLLAMA_HOST}/api/generate"
    b64_left = to_base64_jpeg(left_img)
    b64_right = to_base64_jpeg(right_img)

    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT_TEMPLATE,
        "images": [b64_left, b64_right],
        "stream": False,
        # 필요 시 temperature 등 추가 가능: "options": {"temperature": 0}
    }

    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()
        # JSON이 아닌 텍스트가 끼는 경우를 대비해 {} 블록만 추출
        json_str = extract_json_block(text)
        if json_str is None:
            # 마지막 시도로 따옴표 수정 등 간단 처리
            json_str = fallback_to_json_like(text)
        out = json.loads(json_str)
        sim = float(out.get("similarity", 0.0))
        lab = str(out.get("label", "불량")).strip()
        reason = str(out.get("reason", "")).strip()
        if lab not in ["양품", "불량"]:
            lab = "불량"
        sim = min(max(sim, 0.0), 1.0)
        return {"similarity": sim, "label": lab, "reason": reason, "raw": text}
    except Exception as e:
        return {"similarity": 0.0, "label": "불량", "reason": f"Ollama 실패: {e}", "raw": ""}

def extract_json_block(text: str) -> Optional[str]:
    """
    문자열에서 첫 번째 JSON 객체 블록 {...}만 추출
    """
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(0)

def fallback_to_json_like(text: str) -> str:
    """
    매우 단순한 폴백: 숫자/키워드만 추출해 JSON 구성 시도
    """
    # similarity 숫자 추출
    m = re.search(r"similarity[^0-9]*([01]?\.\d+|\d+)", text, flags=re.IGNORECASE)
    sim = 0.0
    if m:
        try:
            sim = float(m.group(1))
        except:
            sim = 0.0
    label = "양품" if sim >= 0.9 else "불량"
    return json.dumps({"similarity": float(sim), "label": label, "reason": "fallback"})

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

def compare_long_image(path: str,
                       expect_hw=(EXPECTED_HEIGHT, EXPECTED_WIDTH),
                       allow_resize=ALLOW_RESIZE) -> Dict:
    img = load_image(path)
    left, right = split_long_image(img, expect_hw=expect_hw, allow_resize=allow_resize)
    res = ollama_compare_two(left, right)
    # 이 스크립트에서는 최종 판정은 임계값 τ로 일관되게 결정
    return {"path": path, "similarity": float(res.get("similarity", 0.0)), "judge": res}

def calibrate_tau_from_folder(folder: str,
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
            r = compare_long_image(f, expect_hw=expect_hw, allow_resize=allow_resize)
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

def classify_folder(folder: str, tau: float,
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
            r = compare_long_image(f, expect_hw=expect_hw, allow_resize=allow_resize)
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
    print(f"[INFO] Ollama 모델: {MODEL_NAME} @ {OLLAMA_HOST}")
    # 1) 캘리브레이션
    print("[INFO] 캘리브레이션 폴더:", NORMAL_DIR)
    cal = calibrate_tau_from_folder(NORMAL_DIR)
    if "error" in cal:
        raise RuntimeError(f"캘리브레이션 실패: {cal}")
    tau = cal["tau"]
    print("[INFO] 캘리브레이션 결과:")
    print(json.dumps(cal, ensure_ascii=False, indent=2))

    # 2) 테스트 분류
    print("[INFO] 테스트 폴더:", TEST_DIR)
    out_csv = OUT_CSV if OUT_CSV else os.path.join(TEST_DIR, "result.csv")
    res = classify_folder(TEST_DIR, tau, out_csv=out_csv)
    if "error" in res:
        raise RuntimeError(f"분류 실패: {res}")

    print("[INFO] 분류 완료:")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[INFO] 결과 CSV: {res['saved']}")

if __name__ == "__main__":
    main()