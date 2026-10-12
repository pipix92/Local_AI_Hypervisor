"""
Ultra-Cyberpunk Neon Lightning Icon Generator for AI TUI.
Produces high-resolution icon.png (1024x1024) and multi-size icon.ico.
"""
import math
from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
OUT_DIR = r"C:\Users\User\Documents\ai_tui"

def create_cyberpunk_lightning():
    # Base transparent canvas
    base = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    
    # 1. Background disc with subtle dark cyber gradient and glowing border
    cx, cy = SIZE // 2, SIZE // 2
    r_outer = 480
    r_inner = 460
    
    bg_layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_bg = ImageDraw.Draw(bg_layer)
    
    # Radial dark violet-black background
    for r in range(r_inner, 0, -2):
        t = r / r_inner
        # Deep obsidian & dark indigo/black
        cr = int(6 + 18 * (1 - t * 0.8))
        cg = int(4 + 10 * (1 - t * 0.8))
        cb = int(18 + 36 * (1 - t * 0.8))
        alpha = int(240 * (1 - 0.2 * (t ** 2)))
        draw_bg.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(cr, cg, cb, alpha))
        
    # Cyber ring outer glow (magenta & cyan dual ring)
    ring_layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_ring = ImageDraw.Draw(ring_layer)
    
    # Outer ring segments / circuit accents
    draw_ring.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline=(0, 240, 255, 180), width=6)
    draw_ring.ellipse([cx - (r_outer - 16), cy - (r_outer - 16), cx + (r_outer - 16), cy + (r_outer - 16)], outline=(255, 0, 150, 140), width=4)
    
    # Add neon corner ticks / cyber indicators
    for angle in [30, 60, 120, 150, 210, 240, 300, 330]:
        rad = math.radians(angle)
        x1 = cx + math.cos(rad) * (r_outer - 25)
        y1 = cy + math.sin(rad) * (r_outer - 25)
        x2 = cx + math.cos(rad) * (r_outer + 8)
        y2 = cy + math.sin(rad) * (r_outer + 8)
        draw_ring.line([(x1, y1), (x2, y2)], fill=(0, 255, 255, 220), width=4)
        
    # Ring bloom
    ring_glow1 = ring_layer.filter(ImageFilter.GaussianBlur(16))
    ring_glow2 = ring_layer.filter(ImageFilter.GaussianBlur(6))
    
    base = Image.alpha_composite(base, bg_layer)
    base = Image.alpha_composite(base, ring_glow1)
    base = Image.alpha_composite(base, ring_glow2)
    base = Image.alpha_composite(base, ring_layer)
    
    # 2. Lightning Bolt geometry
    # Primary bolt coordinates (high detail, dynamic angled facets)
    # Normalized coords (0..1) centered at (0.5, 0.5)
    main_bolt_pts = [
        (0.56, 0.08),  # Top point
        (0.24, 0.48),  # Upper left knee
        (0.44, 0.48),  # Upper inner notch
        (0.28, 0.88),  # Lower left knee
        (0.82, 0.38),  # Lower right spike / elbow
        (0.54, 0.38),  # Right inner notch
        (0.70, 0.08),  # Top right facet
    ]
    
    # Secondary splinter lightning bolt (branching off)
    branch_pts = [
        (0.42, 0.50),
        (0.68, 0.72),
        (0.56, 0.74),
        (0.76, 0.94),
        (0.52, 0.84),
        (0.46, 0.60),
    ]
    
    scale_w = SIZE * 0.75
    scale_h = SIZE * 0.78
    
    def transform_poly(pts, ox=0, oy=0, scale=1.0):
        res = []
        for x, y in pts:
            px = cx + (x - 0.5) * scale_w * scale + ox
            py = cy + (y - 0.5) * scale_h * scale + oy
            res.append((px, py))
        return res

    bolt_poly = transform_poly(main_bolt_pts, ox=0, oy=-10)
    branch_poly = transform_poly(branch_pts, ox=-10, oy=-10)
    
    # 3. Multi-stage glowing passes (Neon Cyan + Neon Magenta)
    # Deep ambient neon pink/purple wash behind bolt
    amb_layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_amb = ImageDraw.Draw(amb_layer)
    draw_amb.polygon(bolt_poly, fill=(255, 0, 180, 200))
    draw_amb.polygon(branch_poly, fill=(0, 255, 255, 180))
    amb_glow = amb_layer.filter(ImageFilter.GaussianBlur(70))
    base = Image.alpha_composite(base, amb_glow)
    
    # Cyan wide glow
    cyan_wide = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_cw = ImageDraw.Draw(cyan_wide)
    draw_cw.polygon(bolt_poly, fill=(0, 255, 255, 220))
    draw_cw.polygon(branch_poly, fill=(0, 230, 255, 190))
    cyan_glow_wide = cyan_wide.filter(ImageFilter.GaussianBlur(38))
    base = Image.alpha_composite(base, cyan_glow_wide)

    # Magenta mid glow
    mag_mid = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_mm = ImageDraw.Draw(mag_mid)
    draw_mm.polygon(bolt_poly, outline=(255, 20, 180, 240), fill=(255, 0, 140, 180))
    for p in bolt_poly:
        draw_mm.line([bolt_poly[0], p], fill=(255, 0, 200, 80), width=4)
    mag_glow_mid = mag_mid.filter(ImageFilter.GaussianBlur(18))
    base = Image.alpha_composite(base, mag_glow_mid)

    # Cyan tight sharp glow
    cyan_tight = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_ct = ImageDraw.Draw(cyan_tight)
    draw_ct.polygon(bolt_poly, fill=(60, 255, 255, 240), outline=(0, 255, 255, 255))
    draw_ct.polygon(branch_poly, fill=(0, 255, 255, 220), outline=(255, 255, 255, 255))
    cyan_glow_tight = cyan_tight.filter(ImageFilter.GaussianBlur(8))
    base = Image.alpha_composite(base, cyan_glow_tight)

    # 4. Core pure energy layer (White / Light Cyan laser beam)
    core_layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_core = ImageDraw.Draw(core_layer)
    # Inset bolt for the intense hot white core
    bolt_core_poly = transform_poly(main_bolt_pts, ox=0, oy=-10, scale=0.88)
    draw_core.polygon(bolt_core_poly, fill=(240, 255, 255, 255))
    draw_core.polygon(bolt_poly, outline=(255, 255, 255, 255), width=3)
    
    # Internal cyber facet lines
    draw_core.line([bolt_poly[0], bolt_poly[3]], fill=(255, 255, 255, 220), width=4)
    draw_core.line([bolt_poly[1], bolt_poly[4]], fill=(200, 255, 255, 190), width=3)
    draw_core.line([bolt_poly[2], bolt_poly[5]], fill=(255, 120, 240, 220), width=3)
    
    core_blur = core_layer.filter(ImageFilter.GaussianBlur(2))
    base = Image.alpha_composite(base, core_blur)
    base = Image.alpha_composite(base, core_layer)

    # 5. Electric sparks / particles / cyber grid accents
    spark_layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_spk = ImageDraw.Draw(spark_layer)
    sparks = [
        (320, 240, 8, (0, 255, 255, 240)),
        (260, 360, 6, (255, 0, 200, 230)),
        (720, 310, 9, (0, 255, 255, 240)),
        (780, 420, 7, (255, 50, 220, 220)),
        (380, 680, 8, (0, 255, 255, 240)),
        (660, 620, 10, (0, 255, 255, 255)),
        (480, 820, 7, (255, 0, 220, 240)),
        (340, 800, 5, (0, 255, 255, 200)),
        (600, 160, 6, (255, 100, 255, 220)),
    ]
    for sx, sy, rad, col in sparks:
        draw_spk.ellipse([sx - rad, sy - rad, sx + rad, sy + rad], fill=col)
        # 4-point star flare
        draw_spk.line([(sx - rad * 2.5, sy), (sx + rad * 2.5, sy)], fill=(255, 255, 255, 220), width=2)
        draw_spk.line([(sx, sy - rad * 2.5), (sx, sy + rad * 2.5)], fill=(255, 255, 255, 220), width=2)
        
    spark_glow = spark_layer.filter(ImageFilter.GaussianBlur(4))
    base = Image.alpha_composite(base, spark_glow)
    base = Image.alpha_composite(base, spark_layer)
    
    # Save PNG
    png_path = f"{OUT_DIR}\\icon.png"
    base.save(png_path, "PNG")
    print(f"Generated PNG: {png_path} ({SIZE}x{SIZE})")
    
    # Save Multi-resolution ICO
    ico_path = f"{OUT_DIR}\\icon.ico"
    ico_sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    base.save(ico_path, format="ICO", sizes=ico_sizes)
    print(f"Generated ICO: {ico_path} with sizes: {ico_sizes}")

if __name__ == "__main__":
    create_cyberpunk_lightning()
