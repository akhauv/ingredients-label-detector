import os
from PIL import Image, ExifTags

input_dir = '../data/raw_images'
output_dir = '../data/resized_images'
new_size = (600, 800)

count = 1
for filename in os.listdir(input_dir):
    if not filename.endswith('.jpeg'):
        continue
    img = Image.open(os.path.join(input_dir, filename))

     # Check if image has orientation metadata
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        # No EXIF orientation metadata found
        pass

    new_img = img.resize(new_size)
    new_img.save(os.path.join(output_dir, str(count) + ".jpeg"))
    count = count + 1