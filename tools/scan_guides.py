import PIL.Image
import json

img = PIL.Image.open("d:/projet/Desert_Strike_Return_to_the_Gulf/assets/MS-DOS - Desert Strike_ Return to the Gulf - Vehicles - Apache Helicopter.png")
pixels = img.load()
width, height = img.size

# Pink lines (Row guides) usually #FF00FF
# Cyan lines (Col guides) usually #00FFFF

row_y = []
for y in range(height):
    # Check if a significant portion of the row is pink
    pink_count = 0
    for x in range(0, width, 10):
        r, g, b, a = pixels[x, y]
        if r > 250 and g < 10 and b > 250: # Pink
            pink_count += 1
    if pink_count > width // 20: 
        row_y.append(y)

col_x = []
for x in range(width):
    # Check if a significant portion of the col is cyan
    cyan_count = 0
    for y in range(0, height, 10):
        r, g, b, a = pixels[x, y]
        if r < 10 and g > 250 and b > 250: # Cyan
            cyan_count += 1
    if cyan_count > height // 20:
        col_x.append(x)

print(json.dumps({"rows": row_y, "cols": col_x}))
