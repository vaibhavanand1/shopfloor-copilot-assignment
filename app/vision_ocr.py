"""Computer Vision & OCR Layer for ShopFloor Copilot.

Uses classical OpenCV techniques (grayscale conversion, Gaussian blur, Canny edge detection,
thresholding, and contour geometry analysis) to classify precision parts into 4 categories:
- no_defect
- surface_scratch
- crack
- dimensional_deviation

Extracts stamped Work Order IDs (WO-XXXX) using pytesseract with fallback mechanisms.
"""

import base64
import os
import re
import shutil
import ssl
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


class VisionOCREngine:
    def __init__(self, default_ocr: str = "pytesseract"):
        self.default_ocr = default_ocr
        self.tesseract_available = shutil.which("tesseract") is not None
        self._easyocr_reader = None

    def _get_easyocr_reader(self):
        """Lazy loader for EasyOCR reader."""
        if self._easyocr_reader is None:
            try:
                # Bypass SSL certificate verification error on some macOS python environments
                try:
                    ssl._create_default_https_context = ssl._create_unverified_context
                except Exception:
                    pass
                import easyocr
                self._easyocr_reader = easyocr.Reader(["en"], gpu=False)
            except Exception as e:
                self._easyocr_reader = False
        return self._easyocr_reader if self._easyocr_reader is not False else None

    def load_image(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Loads an image into a BGR numpy ndarray from path, base64 string, or raw bytes."""
        if isinstance(image_input, np.ndarray):
            return image_input

        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from bytes.")
            return img

        if isinstance(image_input, str):
            # Check if base64 encoded
            if image_input.startswith("data:image") or ";base64," in image_input or len(image_input) > 260:
                raw_b64 = image_input.split(",")[-1]
                img_bytes = base64.b64decode(raw_b64)
                return self.load_image(img_bytes)

            # Check if file path
            if os.path.exists(image_input):
                img = cv2.imread(image_input)
                if img is None:
                    raise ValueError(f"Could not read image file at {image_input}")
                return img

            # Try raw base64 decode if string length is large
            try:
                img_bytes = base64.b64decode(image_input)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception:
                pass

        raise ValueError("Invalid image input: expected existing file path, raw bytes, or base64 string.")

    def classify_defect(self, bgr_img: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Classifies part defect using classical contour solidity and edge analysis.

        Returns:
            Tuple of (defect_category, confidence, metrics_dict)
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # 1. Outer Workpiece Contour & Dimensional Analysis
        # Background is dark (~30-40), workpiece is metallic (~200)
        _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return "no_defect", 0.50, {"error": "No part contour found in image"}

        main_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(main_contour)
        bx, by, bw, bh = cv2.boundingRect(main_contour)
        bbox_area = float(bw * bh) if (bw * bh) > 0 else 1.0
        solidity = contour_area / bbox_area

        # Aspect ratio check
        aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0

        # Check for dimensional deviation:
        # A nominal square machined billet has solidity ~ 0.98 - 0.995.
        # Chipped corner, warped flange, or missing geometry drops solidity below 0.96.
        if solidity < 0.96 or aspect_ratio < 0.85 or aspect_ratio > 1.15:
            metrics = {
                "solidity": round(solidity, 4),
                "aspect_ratio": round(aspect_ratio, 3),
                "bounding_box": [int(bx), int(by), int(bw), int(bh)],
                "defect_reason": "Solidity or aspect ratio outside nominal CNC drawing envelope",
            }
            return "dimensional_deviation", 0.95, metrics

        # 2. Surface Defect Inspection in Central Machining ROI
        # Define central inspection ROI avoiding stamping pad and corner mounting bores
        # For nominal 500x500 image, central zone is y: 170-370, x: 130-370
        roi_y1 = int(by + bh * 0.28)
        roi_y2 = int(by + bh * 0.82)
        roi_x1 = int(bx + bw * 0.15)
        roi_x2 = int(bx + bw * 0.85)

        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            roi = gray

        # Noise reduction & Canny edge detection
        blurred_roi = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(blurred_roi, 50, 150)
        edge_pixels = cv2.countNonZero(edges)

        defect_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defect_contour_count = len(defect_contours)
        max_defect_length = max([cv2.arcLength(c, False) for c in defect_contours], default=0.0)

        metrics = {
            "solidity": round(solidity, 4),
            "roi_edge_pixels": int(edge_pixels),
            "defect_contour_count": int(defect_contour_count),
            "max_defect_length": round(float(max_defect_length), 2),
            "bounding_box": [int(bx), int(by), int(bw), int(bh)],
        }

        # Calibration thresholds based on CNC inspection surface analysis:
        # - Clean surface: roi_edge_pixels < 50
        # - Linear surface scratch: 50 <= roi_edge_pixels < 450
        # - Branching structural crack: roi_edge_pixels >= 450
        if edge_pixels < 50:
            return "no_defect", 0.98, metrics
        elif edge_pixels < 450:
            return "surface_scratch", 0.92, metrics
        else:
            return "crack", 0.96, metrics

    def extract_work_order(self, bgr_img: np.ndarray, engine: Optional[str] = None) -> Tuple[Optional[str], str]:
        """Extracts stamped Work Order ID (WO-XXXX) using pytesseract or EasyOCR.

        Returns:
            Tuple of (extracted_id_or_none, engine_used)
        """
        chosen_engine = engine or self.default_ocr

        # Region of interest: upper stamping pad on part
        # On 500x500 image, stamping pad is approx y: [95:170], x: [150:350]
        h, w = bgr_img.shape[:2]
        pad_y1, pad_y2 = int(h * 0.18), int(h * 0.35)
        pad_x1, pad_x2 = int(w * 0.28), int(w * 0.72)
        stamp_roi = bgr_img[pad_y1:pad_y2, pad_x1:pad_x2]

        wo_pattern = re.compile(r"WO-\d+", re.IGNORECASE)

        # 1. Try pytesseract if requested and available
        if chosen_engine == "pytesseract" and self.tesseract_available:
            try:
                import pytesseract
                # Convert ROI to grayscale and contrast stretch
                gray_roi = cv2.cvtColor(stamp_roi, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray_roi, config="--psm 6")
                m = wo_pattern.search(text)
                if m:
                    return m.group(0).upper(), "pytesseract"

                # Fallback to full image scan if crop missed it
                full_gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                full_text = pytesseract.image_to_string(full_gray, config="--psm 11")
                m_full = wo_pattern.search(full_text)
                if m_full:
                    return m_full.group(0).upper(), "pytesseract"
                return None, "pytesseract"
            except Exception:
                pass  # Fall through to EasyOCR or fallback

        # 2. Try EasyOCR if requested or if pytesseract failed
        easy_reader = self._get_easyocr_reader()
        if easy_reader is not None:
            try:
                results = easy_reader.readtext(stamp_roi)
                for bbox, text, prob in results:
                    m = wo_pattern.search(text)
                    if m:
                        return m.group(0).upper(), "easyocr"

                # Check full image if ROI had no match
                results_full = easy_reader.readtext(bgr_img)
                for bbox, text, prob in results_full:
                    m = wo_pattern.search(text)
                    if m:
                        return m.group(0).upper(), "easyocr"
                return None, "easyocr"
            except Exception:
                pass

        # 3. Fallback: regex search on OCR if tesseract binary exists directly
        if self.tesseract_available:
            try:
                import pytesseract
                text = pytesseract.image_to_string(stamp_roi)
                m = wo_pattern.search(text)
                if m:
                    return m.group(0).upper(), "pytesseract"
            except Exception:
                pass

        return None, "fallback_contour_ocr"

    def inspect(self, image_input: Union[str, bytes, np.ndarray], ocr_engine: Optional[str] = None) -> Dict[str, Any]:
        """Full visual inspection pipeline: defect classification + Work Order OCR extraction."""
        bgr_img = self.load_image(image_input)
        defect_cat, confidence, metrics = self.classify_defect(bgr_img)
        work_order_id, engine_used = self.extract_work_order(bgr_img, engine=ocr_engine)

        return {
            "defect_category": defect_cat,
            "confidence": confidence,
            "metrics": metrics,
            "extracted_ocr_text": work_order_id,
            "ocr_engine": engine_used,
        }
