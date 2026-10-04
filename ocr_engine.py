"""
ocr_engine.py
=============
Offline OCR pipeline (Tesseract LSTM neural network) tuned for maximum accuracy.

Accuracy Enhancements:
1. Automatic 2x Lanczos upscaling with explicit 300 DPI metadata.
2. Multi-pass evaluation testing both PSM 3 (auto page segmentation) and
   PSM 6 (uniform block of text), selecting the highest-confidence extraction.
3. Multiple image preprocessing variants: Plain grayscale, Contrast + Unsharp Mask,
   Otsu Binarization, and Deskew.
4. High-fidelity layout-aware text reconstruction: de-hyphenates broken words,
   stitches soft line wraps in prose, and preserves structured lists and receipts.
"""

import re
import shutil
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pytesseract
import scipy.ndimage as ndi
from PIL import Image, ImageFilter, ImageOps

from text_formatter import ProperTextFormatter

MIN_WORD_CONF = 22          # tokens below this are treated as noise
GOOD_ENOUGH_CONF = 94       # stop trying variants when we reach this threshold


class MLImageReader:
    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            found = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
            pytesseract.pytesseract.tesseract_cmd = found
        self.formatter = ProperTextFormatter(enable_spell_check=False)

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[-1])
            return bg
        return image.convert("RGB")

    @staticmethod
    def _scale(image: Image.Image) -> Image.Image:
        """Upscale image with 300 DPI metadata so glyphs are crisp for LSTM."""
        w, h = image.size
        # Upscale 2x for standard text images to sharpen character glyphs
        factor = 2.0 if max(w, h) < 3000 else 1.0
        if factor > 1.0:
            new_w, new_h = int(w * factor), int(h * factor)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Set 300 DPI in metadata
        image.info['dpi'] = (300, 300)
        return image

    def _variants(self, image: Image.Image):
        """Yield (name, image) variants lazily."""
        base = self._scale(self._to_rgb(image))
        gray = ImageOps.grayscale(base)
        yield "plain", gray

        enhanced = ImageOps.autocontrast(gray, cutoff=1)
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
        yield "contrast", enhanced

        arr = np.array(enhanced)
        thresh = self._otsu_threshold(arr)
        binary = Image.fromarray(np.where(arr > thresh, 255, 0).astype(np.uint8))
        yield "binary", binary

        angle = self._detect_skew_angle(arr)
        if 1.5 <= abs(angle) <= 20:
            yield "deskew", enhanced.rotate(-angle, expand=True, fillcolor=255,
                                            resample=Image.Resampling.BICUBIC)

    def _run_psm(self, img: Image.Image, psm: int) -> Tuple[float, List[Dict[str, Any]], str]:
        """Runs Tesseract with a specific PSM mode."""
        config = f"--oem 1 --psm {psm} --dpi 300"
        try:
            data = pytesseract.image_to_data(
                img, config=config, output_type=pytesseract.Output.DICT
            )
        except Exception:
            return 0.0, [], ""

        words = []
        for i, txt in enumerate(data.get("text", [])):
            txt = txt.strip()
            conf = float(data["conf"][i])
            if not txt or conf < 0:
                continue
            words.append({
                "text": txt, "conf": conf,
                "left": data["left"][i], "top": data["top"][i],
                "width": data["width"][i], "height": data["height"][i],
                "block": data["block_num"][i], "par": data["par_num"][i],
                "line": data["line_num"][i],
            })

        kept = [w for w in words if w["conf"] >= MIN_WORD_CONF]
        if not kept:
            return 0.0, [], ""

        total_chars = sum(len(w["text"]) for w in kept)
        score = sum(w["conf"] * len(w["text"]) for w in kept) / max(total_chars, 1)

        raw = pytesseract.image_to_string(img, config=config).strip()
        return score, kept, raw

    def read_image(
        self,
        image_input: Union[str, Image.Image],
        enable_preprocessing: bool = True,
        psm: int = 3,
        spell_check: bool = False,
    ) -> Dict[str, Any]:
        image = Image.open(image_input) if isinstance(image_input, str) else image_input

        best_score = -1.0
        best_words: List[Dict[str, Any]] = []
        best_img = None
        best_name = "plain"
        best_raw = ""
        best_psm = psm

        # Test both requested PSM and PSM 6 (single uniform block)
        psm_candidates = [psm] if psm == 6 else [psm, 6]

        if enable_preprocessing:
            for name, variant in self._variants(image):
                for p in psm_candidates:
                    score, words, raw = self._run_psm(variant, p)
                    if score > best_score:
                        best_score = score
                        best_words = words
                        best_img = variant
                        best_name = f"{name} (psm {p})"
                        best_raw = raw
                        best_psm = p
                    if best_score >= GOOD_ENOUGH_CONF:
                        break
                if best_score >= GOOD_ENOUGH_CONF:
                    break
        else:
            plain = self._scale(self._to_rgb(image))
            best_score, best_words, best_raw = self._run_psm(plain, psm)
            best_img = plain

        text = self._assemble(best_words)
        if not text and best_raw:
            text = best_raw

        if spell_check:
            text = self.formatter.format_text(text)
        else:
            text = self.formatter.clean_light(text)

        confs = [w["conf"] for w in best_words]
        return {
            "proper_text": text,
            "raw_text": best_raw,
            "confidence": round(float(np.mean(confs)), 1) if confs else 0.0,
            "word_count": len(text.split()),
            "character_count": len(text),
            "skew_angle": 0.0,
            "variant": best_name,
            "words": best_words,
            "preprocessed_image": best_img,
        }

    @staticmethod
    def _assemble(words: List[Dict[str, Any]]) -> str:
        """Rebuild text from Tesseract's block/paragraph/line structure."""
        if not words:
            return ""

        lines: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
        for w in words:
            lines.setdefault((w["block"], w["par"], w["line"]), []).append(w)

        paragraphs: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for (b, p, l), ws in sorted(lines.items()):
            ws.sort(key=lambda x: x["left"])
            paragraphs.setdefault((b, p), []).append({
                "text": " ".join(x["text"] for x in ws),
                "left": min(x["left"] for x in ws),
                "right": max(x["left"] + x["width"] for x in ws),
            })

        out_pars = []
        for _, ls in sorted(paragraphs.items()):
            max_right = max(l["right"] for l in ls)
            min_left = min(l["left"] for l in ls)
            span = max(max_right - min_left, 1)
            result = ""
            for i, line in enumerate(ls):
                txt = line["text"]
                if i == len(ls) - 1:
                    result += txt
                    continue
                nxt = ls[i + 1]["text"]
                is_full = (line["right"] - min_left) >= 0.75 * span
                continues = nxt[:1].islower()
                if txt.endswith("-") and continues:
                    result += txt[:-1]            # de-hyphenate "re-\nquired"
                elif is_full and continues and not re.search(r"[.!?:]$", txt):
                    result += txt + " "           # prose wrap
                else:
                    result += txt + "\n"
            out_pars.append(result)
        return "\n\n".join(out_pars)

    @staticmethod
    def _otsu_threshold(image_arr: np.ndarray) -> int:
        hist, _ = np.histogram(image_arr.ravel(), bins=256, range=(0, 256))
        total = image_arr.size
        sum_total = np.dot(np.arange(256), hist)
        w_bg = sum_bg = 0.0
        best, thr = 0.0, 128
        for i in range(256):
            w_bg += hist[i]
            if w_bg == 0:
                continue
            w_fg = total - w_bg
            if w_fg == 0:
                break
            sum_bg += i * hist[i]
            var = w_bg * w_fg * (sum_bg / w_bg - (sum_total - sum_bg) / w_fg) ** 2
            if var > best:
                best, thr = var, i
        return thr

    def _detect_skew_angle(self, gray_arr: np.ndarray) -> float:
        h, w = gray_arr.shape
        step = int(max(1, max(h, w) / 600))
        sub = gray_arr[::step, ::step]
        binary = (sub < self._otsu_threshold(sub)).astype(np.float32)
        best_score, best_angle = -1.0, 0.0
        for angle in np.arange(-10.0, 10.5, 0.5):
            rot = binary if angle == 0 else ndi.rotate(binary, angle, reshape=False, order=0)
            score = np.var(rot.sum(axis=1))
            if score > best_score:
                best_score, best_angle = score, angle
        return float(best_angle)
