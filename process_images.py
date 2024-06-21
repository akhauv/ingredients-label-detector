import os
from PIL import Image

input_dir = './raw_images'
output_dir = './resized_images'
new_size = (600, 800)

for filename in os.listdir(input_dir):
    if not filename.endswith('.jpeg'):
        continue
    print(filename)
    img = Image.open(os.path.join(input_dir, filename))
    new_img = img.resize(new_size)
    new_img.save(os.path.join(output_dir, filename))