import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


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
    Simulates physical scratches on the surface of a painting.
    These are straight lines of varying brightness and thickness
    to reflect scuffs or abrasions on the canvas.
    """

    def __init__(
        self,
        num_scratches_range=(1, 10),
        max_length_factor=0.8,
        thickness_range=(1, 3),
        color_range=((0, 50), (200, 255))
    ):
        self.num_scratches_range = num_scratches_range
        self.max_length_factor = max_length_factor
        self.thickness_range = thickness_range
        self.color_range = color_range

    def apply(self, image_pil):
        """
        Add random scratches as lines across the image.
        Scratches may be light or dark depending on chosen color range.
        """
        draw = ImageDraw.Draw(image_pil)
        width, height = image_pil.size
        num_scratches = random.randint(*self.num_scratches_range)

        for _ in range(num_scratches):
            # Random starting point
            x1, y1 = random.randint(0, width), random.randint(0, height)

            # Random angle and length
            length = (
                random.uniform(0.1, self.max_length_factor)
                * min(width, height)
            )
            angle = random.uniform(0, 2 * np.pi)
            x2 = max(0, min(width - 1, int(x1 + length * np.cos(angle))))
            y2 = max(0, min(height - 1, int(y1 + length * np.sin(angle))))

            thickness = random.randint(*self.thickness_range)

            # Light or dark scratch depending on random draw
            if random.random() > 0.5:
                color_val = random.randint(*self.color_range[0])
            else:
                color_val = random.randint(*self.color_range[1])

            color = (color_val, color_val, color_val)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)

        return image_pil


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
