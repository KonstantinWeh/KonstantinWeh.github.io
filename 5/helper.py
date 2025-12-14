from PIL import Image

try:
    img = Image.open('campanile_high_res.jpg')
    img = img.resize((64, 64), Image.LANCZOS)
    img.save('campanile.jpg')
    print('Image resized and saved as campanile.jpg')
except Exception as e:
    print('Error resizing image:', e)
