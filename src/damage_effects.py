import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import albumentations as A


class TearDamage:
    """
    Simulates physical tears along the edges of a painting.
    Tears are drawn as jagged polygons filled with a background color,
    resembling rips that expose canvas or backing material.
    """

    def __init__(
        self,
        num_tears_range=(1, 3),
        max_tear_length_factor=0.5,
        tear_width_range=(5, 20),
        background_color=(200, 200, 200)
    ):
        self.num_tears_range = num_tears_range
        self.max_tear_length_factor = max_tear_length_factor
        self.tear_width_range = tear_width_range
        self.background_color = background_color

    def apply(self, image_pil):
        """
        Apply random edge tears to the given image.
        Returns a new image with jagged tear effects added.
        """
        # Convert to RGBA to support transparency
        original_mode = image_pil.mode
        if original_mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')

        draw = ImageDraw.Draw(image_pil)
        width, height = image_pil.size
        num_tears = random.randint(*self.num_tears_range)

        for _ in range(num_tears):
            # Randomly pick an edge to begin the tear
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                start_x, start_y = random.randint(0, width), 0
            elif edge == 'bottom':
                start_x, start_y = random.randint(0, width), height - 1
            elif edge == 'left':
                start_x, start_y = 0, random.randint(0, height)
            else:
                start_x, start_y = width - 1, random.randint(0, height)

            tear_points = [(start_x, start_y)]
            current_x, current_y = start_x, start_y

            # Length of the tear as a proportion of the image dimension
            tear_length = (
                random.uniform(0.1, self.max_tear_length_factor)
                * min(width, height)
            )
            num_segments = random.randint(3, 7)

            # Build a jagged path inward from the edge
            for _ in range(num_segments):
                # Move in the main direction + some noise
                if edge in ['top', 'bottom']:
                    current_y += (
                        1 if edge == 'top' else -1
                    ) * (tear_length / num_segments)
                    current_x += random.uniform(
                        -tear_length / 3, tear_length / 3
                    )
                else:
                    current_x += (
                        1 if edge == 'left' else -1
                    ) * (tear_length / num_segments)
                    current_y += random.uniform(
                        -tear_length / 3, tear_length / 3
                    )

                # Add jaggedness perpendicular to tear direction
                offset_x = random.randint(
                    -self.tear_width_range[1], self.tear_width_range[1]
                )
                offset_y = random.randint(
                    -self.tear_width_range[1], self.tear_width_range[1]
                )
                p1_x = max(0, min(width - 1, int(current_x + offset_x)))
                p1_y = max(0, min(height - 1, int(current_y + offset_y)))
                tear_points.append((p1_x, p1_y))

            if len(tear_points) > 2:
                # Build polygon representing the tear width
                poly_points = []

                # First side (clockwise)
                for px, py in tear_points:
                    angle = np.arctan2(py - start_y, px - start_x) + np.pi / 2
                    w = random.randint(*self.tear_width_range) / 2
                    poly_points.append((
                        int(px + w * np.cos(angle)),
                        int(py + w * np.sin(angle))
                    ))

                # Second side (counter-clockwise)
                for px, py in reversed(tear_points):
                    angle = np.arctan2(py - start_y, px - start_x) - np.pi / 2
                    w = random.randint(*self.tear_width_range) / 2
                    poly_points.append((
                        int(px + w * np.cos(angle)),
                        int(py + w * np.sin(angle))
                    ))

                # Fill the polygon with the simulated backing color
                draw.polygon(
                    poly_points,
                    fill=self.background_color + (255,)
                )

        return image_pil.convert(original_mode)


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

                color = self.color_cycle[seg % len(self.color_cycle)] + (random.randint(100, 180),)
                draw.line([(x, y), (x_end, y_end)], fill=color, width=self.thickness)

                # Move to end of gap
                x = np.clip(x_end + int(gap_len * np.sin(angle)), 0, width - 1)
                y = np.clip(y_end + int(gap_len * np.cos(angle)), 0, height - 1)

                if y >= height:
                    break

        return Image.alpha_composite(image_pil, overlay).convert('RGB')


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
        """
        Apply semi-transparent brownish stains, blurred for realism.
        Returns a stained version of the original image.
        """
        # Create a transparent overlay for stains
        overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image_pil.size
        num_stains = random.randint(*self.num_stains_range)

        for _ in range(num_stains):
            # Random stain radius
            rw = (
                random.uniform(*self.stain_size_factor_range) * width / 2
            )
            rh = (
                random.uniform(*self.stain_size_factor_range) * height / 2
            )
            cx, cy = random.randint(0, width), random.randint(0, height)
            bbox = [cx - rw, cy - rh, cx + rw, cy + rh]

            # Semi-transparent brownish color
            alpha = random.randint(*self.alpha_range)
            stain_color = (*self.color_tint, alpha)

            draw.ellipse(bbox, fill=stain_color)

        # Blur to blend stains smoothly
        overlay = overlay.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(5, 15))
        )

        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')

        return Image.alpha_composite(image_pil, overlay).convert('RGB')


class FlakingDamage:
    def __init__(
        self,
        num_flakes_range=(80, 120),
        flake_size_range=(12, 28),
        num_clusters=2,
        cluster_radius=100,
        flake_colors=None
    ):
        self.num_flakes_range = num_flakes_range
        self.flake_size_range = flake_size_range
        self.num_clusters = num_clusters
        self.cluster_radius = cluster_radius
        self.flake_colors = flake_colors or [
            (198, 183, 165),  # raw linen
            (220, 205, 190),  # primed canvas
            (181, 166, 142),  # aged gesso
            (214, 200, 185),  # neutral warm canvas
            (201, 188, 168)   # dusty beige
        ]

    def _random_polygon(self, cx, cy, width, height):
        num_points = random.randint(5, 8)
        angle_step = 2 * np.pi / num_points
        points = []

        for i in range(num_points):
            angle = i * angle_step + random.uniform(-0.2, 0.2)
            r_x = random.uniform(0.4, 1.0) * width / 2
            r_y = random.uniform(0.4, 1.0) * height / 2
            x = int(cx + r_x * np.cos(angle))
            y = int(cy + r_y * np.sin(angle))
            points.append((x, y))

        return points

    def apply(self, image_pil):
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')

        width, height = image_pil.size
        image_np = np.array(image_pil)

        # Choose one background color per trial
        flake_color = random.choice(self.flake_colors)

        # Create flake cluster centers
        cluster_centers = [
            (random.randint(0, width), random.randint(0, height))
            for _ in range(self.num_clusters)
        ]

        num_flakes = random.randint(*self.num_flakes_range)

        # Create RGBA layer
        flake_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(flake_layer)

        for _ in range(num_flakes):
            # Flake center near a cluster center
            cluster_cx, cluster_cy = random.choice(cluster_centers)
            cx = int(np.clip(np.random.normal(cluster_cx, self.cluster_radius), 0, width - 1))
            cy = int(np.clip(np.random.normal(cluster_cy, self.cluster_radius), 0, height - 1))

            flake_w = random.randint(*self.flake_size_range)
            flake_h = int(flake_w * random.uniform(0.4, 0.7))

            polygon = self._random_polygon(cx, cy, flake_w, flake_h)
            draw.polygon(polygon, fill=flake_color + (255,))

        result = Image.alpha_composite(image_pil, flake_layer)
        return result.convert('RGB')
        

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
            [0.2, 0.18, 0.15],  # dark brown
            [0.25, 0.23, 0.2],  # medium brown
            [0.3, 0.3, 0.3],    # grayish
            [0.28, 0.27, 0.2],  # olive
            [0.35, 0.3, 0.25]   # warm ochre
        ]

        self.crack_mask_paths = [
            os.path.join(self.crack_mask_dir, fname)
            for fname in os.listdir(self.crack_mask_dir)
        ]

    def apply(self, image_pil):
        """
        Apply craquelure damage using a randomly selected and augmented crack mask.
        Returns a PIL image with crack effects blended in.
        """
        crack_path = random.choice(self.crack_mask_paths)
        crack_mask = Image.open(crack_path).convert("L").resize(image_pil.size)

        img_np = np.array(image_pil).astype(np.float32) / 255.0
        mask_np = np.array(crack_mask).astype(np.uint8)

        mask_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])
        augmented = mask_aug(image=mask_np)
        aug_mask = augmented["image"].astype(np.float32) / 255.0
        aug_mask_3c = np.stack([aug_mask] * 3, axis=-1)

        base_tint = np.array(random.choice(self.color_options))
        dark_crack = np.ones_like(img_np) * base_tint
        alpha = random.uniform(*self.alpha_range)

        craquelured_np = (
            img_np * (1 - alpha * aug_mask_3c) +
            dark_crack * (alpha * aug_mask_3c)
        )
        craquelured_img = Image.fromarray((craquelured_np * 255).astype(np.uint8))

        return craquelured_img