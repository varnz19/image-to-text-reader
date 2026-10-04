"""
reader.py
=========
Simple Image-to-Text Reader using an offline Machine Learning model.
Transforms noisy image text into clean, structured "Proper Text"
without making any external cloud AI API calls.

Usage (Python):
    from reader import ImageTextReader

    reader = ImageTextReader()
    result = reader.read("path/to/image.png")
    print(result["proper_text"])

Usage (Command Line):
    python3 reader.py sample.png
    python3 reader.py sample.png --save output.txt
"""

import sys
import os
import argparse
from typing import Dict, Any, Union
from PIL import Image

from ocr_engine import MLImageReader


class ImageTextReader:
    """
    High-level, user-friendly wrapper for the offline ML Image-to-Text Pipeline.
    """

    def __init__(self, spell_check: bool = True):
        self.engine = MLImageReader()
        self.spell_check = spell_check

    def read(
        self,
        image_input: Union[str, Image.Image],
        enable_preprocessing: bool = True
    ) -> Dict[str, Any]:
        """
        Reads text from an image and formats it into proper text.

        Returns:
            dict with:
                - 'proper_text': Clean, formatted, structured text.
                - 'raw_text': Raw unformatted OCR output.
                - 'confidence': Recognition confidence percentage.
                - 'word_count': Total number of words.
                - 'character_count': Length of proper text.
        """
        return self.engine.read_image(
            image_input=image_input,
            enable_preprocessing=enable_preprocessing,
            spell_check=self.spell_check
        )

    def read_to_text(self, image_input: Union[str, Image.Image]) -> str:
        """
        Convenience method that directly returns just the clean 'Proper Text' string.
        """
        return self.read(image_input)["proper_text"]


def main():
    parser = argparse.ArgumentParser(
        description="Simple Offline ML Image-to-Text Reader (Zero AI API calls needed)"
    )
    parser.add_argument("image", help="Path to image file (PNG, JPG, TIFF, WebP, etc.)")
    parser.add_argument("--raw", action="store_true", help="Print raw OCR output alongside proper text")
    parser.add_argument("--no-prep", action="store_true", help="Disable computer vision image preprocessing")
    parser.add_argument("--no-spell", action="store_true", help="Disable statistical spell correction")
    parser.add_argument("--save", type=str, default=None, help="Save formatted text to a file")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing '{args.image}' with offline ML model...")
    reader = ImageTextReader(spell_check=not args.no_spell)
    result = reader.read(args.image, enable_preprocessing=not args.no_prep)

    print("\n" + "=" * 55)
    print("                 PROPER TEXT                 ")
    print("=" * 55)
    print(result["proper_text"] if result["proper_text"] else "[No text detected]")
    print("=" * 55)
    print(f"Stats: {result['word_count']} words | {result['confidence']}% ML confidence\n")

    if args.raw:
        print("\n--- RAW UNFORMATTED OCR ---")
        print(result["raw_text"])

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(result["proper_text"])
        print(f"Formatted text saved to: {args.save}")


if __name__ == "__main__":
    main()
