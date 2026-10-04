"""
generate_samples.py
===================
Generates clean sample images with text to test the offline ML Image-to-Text Reader.
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

os.makedirs("samples", exist_ok=True)

# 1. Book page excerpt
img_book = Image.new("RGB", (700, 360), color=(252, 250, 242))
draw = ImageDraw.Draw(img_book)
book_text = """CHAPTER 4: THE ADVENTURE

The journey through the enchanted mountain re-
quired courage and relentless perseverance. Every single
traveler who stepped upon the ancient cobblestone
pathway was reminded of an eternal truth:

"Knowledge is not merely the accumulation of facts,
but the realization of how deeply everything in this
universe is interconnected."

With fresh determination, the explorers packed their
gear and set off toward the horizon."""
draw.text((40, 40), book_text, fill=(30, 30, 35))
img_book.save("samples/book_page.png")
print("Saved samples/book_page.png")

# 2. Store receipt
img_receipt = Image.new("RGB", (500, 450), color=(255, 255, 255))
draw_rcpt = ImageDraw.Draw(img_receipt)
receipt_text = """==================================
        FRESH MARKET MART        
     124 Central Avenue, NY       
==================================
Date: 2026-10-02       Time: 14:32

ITEM                 QTY    PRICE
----------------------------------
Organic Almond Milk   1     $4.50
Avocado Bag           2     $6.00
Whole Wheat Bread     1     $3.25
Dark Chocolate 85%    2     $5.50
----------------------------------
Subtotal:                  $19.25
Tax (8%):                   $1.54
TOTAL:                     $20.79

Payment Method: Visa Card ****4821
Thank you for shopping with us!
=================================="""
draw_rcpt.text((30, 25), receipt_text, fill=(15, 15, 20))
img_receipt.save("samples/receipt.png")
print("Saved samples/receipt.png")

# 3. Quote Card
img_quote = Image.new("RGB", (650, 260), color=(240, 244, 250))
draw_quote = ImageDraw.Draw(img_quote)
quote_text = """MACHINE LEARNING PHILOSOPHY

"Simplicity is the prerequisite for reliability.
When you build local models that run offline,
you gain privacy, speed, and complete independence
from cloud services."

— Modern Engineering Principles"""
draw_quote.text((40, 40), quote_text, fill=(25, 35, 60))
img_quote.save("samples/quote_card.png")
print("Saved samples/quote_card.png")
