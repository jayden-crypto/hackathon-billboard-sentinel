#!/usr/bin/env python3
"""
Create a simple test billboard image for testing the API
"""
from PIL import Image, ImageDraw, ImageFont
import os

# Create a simple billboard image
width, height = 800, 300
image = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(image)

# Draw a simple billboard
draw.rectangle([50, 50, width-50, height-50], fill='lightblue', outline='black', width=3)
draw.text((width//2-100, height//2-20), "TEST BILLBOARD", fill='black')
draw.text((width//2-80, height//2+10), "Unauthorized", fill='red')

# Save the image
output_path = '/Users/jayden/Desktop/hackathon copy/test_billboard.jpg'
image.save(output_path, 'JPEG')
print(f"Test billboard image created: {output_path}")
