import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import albumentations as A


class ScratchDamage:
    """
    Simulates dashed, weathered scratch lines with subtle grayscale variation.
    Each scratch is made of short segments with gaps in between.
    """

    def __init__(
        self,
        num_scratches_range=(100, 200),
        segment_range=(4, 8),
        segment_length_range=(5, 15),
        gap_length_range=(3, 6),
        angle_range=(-0.2, 0.2),
        thickness=1,
        color_cycle=None
    ):
        self.num_scratches_range = num_scratches_range
        self.segment_range = segment_range
        self.segment_length_range = segment_length_range
        self.gap_length_range = gap_length_range
        self.angle_range = angle_range
        self.thickness = thickness
        self.color_cycle = color_cycle or [
            (30, 30, 30),
            (100, 100, 100),
            (180, 180, 180),
            (240, 240, 240)
        ]

    def apply(self, image_pil):
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')

        width, height = image_pil.size
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        mask_pil = Image.new('L', (width, height), 0)
        draw_mask = ImageDraw.Draw(mask_pil)

        num_scratches = random.randint(*self.num_scratches_range)

        for _ in range(num_scratches):
            x = random.randint(0, width)
            y = random.randint(0, height)
            angle = random.uniform(*self.angle_range)
            num_segments = random.randint(*self.segment_range)

            for seg in range(num_segments):
                seg_len = random.randint(*self.segment_length_range)
                gap_len = random.randint(*self.gap_length_range)

                dx = int(seg_len * np.sin(angle))
                dy = int(seg_len * np.cos(angle))

                x_end = np.clip(x + dx, 0, width - 1)
                y_end = np.clip(y + dy, 0, height - 1)

                color = self.color_cycle[seg % len(self.color_cycle)] + (
                    random.randint(100, 180),)

                draw.line([(x, y), (x_end, y_end)],
                          fill=color, width=self.thickness)
                draw_mask.line([(x, y), (x_end, y_end)],
                               fill=255, width=self.thickness)

                x = np.clip(x_end + int(gap_len * np.sin(angle)), 0, width - 1)
                y = np.clip(y_end + int(gap_len * np.cos(angle)), 0, height - 1)

                if y >= height:
                    break

        result = Image.alpha_composite(image_pil, overlay).convert('RGB')
        if mask_pil.mode != 'RGBA':
            mask_pil = mask_pil.convert('RGBA')
            
        return result, mask_pil


class WaterDiscolouration:
    """
    Simulates water stain effects by overlaying irregular semi-transparent
    brownish blobs on the image. Blur is applied for realism.
    """

    def __init__(
        self,
        num_stains_range=(1, 5),
        stain_size_factor_range=(0.1, 0.4),
        alpha_range=(50, 150),
        color_tint=(200, 180, 150)
    ):
        self.num_stains_range = num_stains_range
        self.stain_size_factor_range = stain_size_factor_range
        self.alpha_range = alpha_range
        self.color_tint = color_tint

    def apply(self, image_pil):
        # Transparent overlay for stains
        overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        width, height = image_pil.size
        num_stains = random.randint(*self.num_stains_range)

        mask_pil = Image.new('L', (width, height), 0)
        draw_mask = ImageDraw.Draw(mask_pil)

        for _ in range(num_stains):
            rw = random.uniform(*self.stain_size_factor_range) * width / 2
            rh = random.uniform(*self.stain_size_factor_range) * height / 2
            cx = random.randint(0, width)
            cy = random.randint(0, height)
            bbox = [cx - rw, cy - rh, cx + rw, cy + rh]

            alpha = random.randint(*self.alpha_range)
            stain_color = (*self.color_tint, alpha)

            draw.ellipse(bbox, fill=stain_color)
            draw_mask.ellipse(bbox, fill=255)

        overlay = overlay.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(5, 15)))

        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')

        result = Image.alpha_composite(image_pil, overlay).convert('RGB')
        if mask_pil.mode != 'RGBA':
            mask_pil = mask_pil.convert('RGBA')
            
        return result, mask_pil


class CraquelureDamage:
    """
    Simulates craquelure (fine cracking) by blending augmented crack masks
    with random earthy tint and variable opacity over the painting.
    """

    def __init__(
        self,
        crack_mask_dir="../data/crack-masks",
        alpha_range=(0.5, 0.9),
        color_options=None
    ):
        self.crack_mask_dir = crack_mask_dir
        self.alpha_range = alpha_range
        self.color_options = color_options or [
            [0.2, 0.18, 0.15],
            [0.25, 0.23, 0.2],
            [0.3, 0.3, 0.3],
            [0.28, 0.27, 0.2],
            [0.35, 0.3, 0.25]
        ]

        self.crack_mask_paths = [
            os.path.join(self.crack_mask_dir, fname)
            for fname in os.listdir(self.crack_mask_dir)
        ]

    def apply(self, image_pil):
        crack_path = random.choice(self.crack_mask_paths)
        crack_mask = Image.open(crack_path).convert("L").resize(image_pil.size)

        img_np = np.array(image_pil).astype(np.float32) / 255.0
        mask_np = np.array(crack_mask).astype(np.uint8)

        mask_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])
        augmented = mask_aug(image=mask_np)
        aug_mask = augmented["image"]

        _, binary_aug_mask = cv2.threshold(
            aug_mask, 127, 255, cv2.THRESH_BINARY)

        aug_mask_3c = np.stack([binary_aug_mask / 255] * 3, axis=-1)

        base_tint = np.array(random.choice(self.color_options))
        dark_crack = np.ones_like(img_np) * base_tint
        alpha = random.uniform(*self.alpha_range)

        craquelured_np = (
            img_np * (1 - alpha * aug_mask_3c) +
            dark_crack * (alpha * aug_mask_3c)
        )
        result = Image.fromarray((craquelured_np * 255).astype(np.uint8))
        mask_pil = Image.fromarray(binary_aug_mask)

        if mask_pil.mode != 'RGBA':
            mask_pil = mask_pil.convert('RGBA')
            
        return result, mask_pil