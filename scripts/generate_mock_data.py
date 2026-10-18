"""Generate realistic synthetic CNC machined part images for ShopFloor Copilot.

Produces 4 part images with stamped Work Order IDs in data/mock_images/:
- part_clean.png: no_defect, stamped WO-1001
- part_scratch.png: surface_scratch, stamped WO-1002
- part_crack.png: crack, stamped WO-1003
- part_deviation.png: dimensional_deviation, stamped WO-1004
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mock_images"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (500, 500)
BG_COLOR = (32, 36, 42)
PART_COLOR = (198, 204, 212)
BORDER_COLOR = (120, 126, 134)
BEVEL_COLOR = (225, 230, 236)
TEXT_COLOR = (24, 28, 34)

def get_font(size: int = 28):
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

def draw_base_part(draw: ImageDraw.ImageDraw, shape_type: str = "nominal"):
    """Draw the aluminum machined part body."""
    if shape_type == "nominal":
        # Nominal 340x340 precision billet with rounded corners
        box = [80, 80, 420, 420]
        draw.rectangle(box, fill=PART_COLOR, outline=BORDER_COLOR, width=3)
        # Inner milling pocket/frame
        draw.rectangle([95, 95, 405, 405], outline=BEVEL_COLOR, width=2)
        # 4 mounting bore holes
        for cx, cy in [(120, 120), (380, 120), (120, 380), (380, 380)]:
            draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(70, 75, 82), outline=(130, 135, 142), width=2)
    elif shape_type == "deviation":
        # Missing corner / warped flange deviation
        points = [
            (80, 80),
            (290, 80),   # Starts deviating here - truncated top right
            (420, 210),  # Severe diagonal shear / missing corner
            (420, 420),
            (80, 420),
        ]
        draw.polygon(points, fill=PART_COLOR, outline=BORDER_COLOR, width=3)
        # Bore holes (one missing due to deviation)
        for cx, cy in [(120, 120), (120, 380), (380, 380)]:
            draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(70, 75, 82), outline=(130, 135, 142), width=2)

def stamp_work_order(draw: ImageDraw.ImageDraw, wo_id: str):
    """Stamp the Work Order text onto the machined stamping pad."""
    font = get_font(size=26)
    # Stamping pad at the top center
    pad_box = [170, 110, 330, 155]
    draw.rectangle(pad_box, fill=(235, 238, 242), outline=(150, 155, 162), width=1)
    draw.text((185, 118), wo_id, fill=TEXT_COLOR, font=font)

def create_clean_part():
    img = Image.new("RGB", IMG_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_base_part(draw, shape_type="nominal")
    stamp_work_order(draw, "WO-1001")
    filepath = DATA_DIR / "part_clean.png"
    img.save(filepath, "PNG")
    print(f"Generated {filepath}")

def create_scratch_part():
    img = Image.new("RGB", IMG_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_base_part(draw, shape_type="nominal")
    stamp_work_order(draw, "WO-1002")
    # Thin high-contrast linear scratch across the center face
    scratch_start = (140, 275)
    scratch_end = (340, 235)
    draw.line([scratch_start, scratch_end], fill=(45, 48, 52), width=2)
    # Bright specular highlight line right beside it
    draw.line([(scratch_start[0], scratch_start[1] + 1), (scratch_end[0], scratch_end[1] + 1)], fill=(255, 255, 255), width=1)
    filepath = DATA_DIR / "part_scratch.png"
    img.save(filepath, "PNG")
    print(f"Generated {filepath}")

def create_crack_part():
    img = Image.new("RGB", IMG_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_base_part(draw, shape_type="nominal")
    stamp_work_order(draw, "WO-1003")
    # Jagged branching fissure/crack
    main_crack = [
        (170, 310),
        (195, 285),
        (220, 295),
        (255, 260),
        (280, 275),
        (315, 240),
        (335, 250),
    ]
    draw.line(main_crack, fill=(20, 22, 25), width=4)
    # Branch 1
    branch_1 = [(220, 295), (235, 330), (250, 345)]
    draw.line(branch_1, fill=(25, 28, 30), width=3)
    # Branch 2
    branch_2 = [(280, 275), (300, 305)]
    draw.line(branch_2, fill=(25, 28, 30), width=3)
    filepath = DATA_DIR / "part_crack.png"
    img.save(filepath, "PNG")
    print(f"Generated {filepath}")

def create_deviation_part():
    img = Image.new("RGB", IMG_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_base_part(draw, shape_type="deviation")
    stamp_work_order(draw, "WO-1004")
    filepath = DATA_DIR / "part_deviation.png"
    img.save(filepath, "PNG")
    print(f"Generated {filepath}")

def main():
    print("Generating synthetic mock images for ShopFloor Copilot...")
    create_clean_part()
    create_scratch_part()
    create_crack_part()
    create_deviation_part()
    print("Done generating mock images.")

if __name__ == "__main__":
    main()
