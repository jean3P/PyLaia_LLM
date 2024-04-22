from PIL import Image
import os
import sys


def resize_image(input_folder, output_folder, base_height=128):
    # Resize images
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):  # Adjust if your images have a different format
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)
            h_percent = (base_height / float(img.size[1]))
            w_size = int((float(img.size[0]) * float(h_percent)))
            img = img.resize((w_size, base_height), Image.Resampling.LANCZOS)

            output_path = os.path.join(output_folder, filename)
            img.save(output_path)
            print(f"Resized and saved: {output_path}")


def main():
    abs_path = sys.path[0]
    print("Current Working Directory:", os.getcwd())
    # base_path = os.path.join(abs_path)
    # source_folder = os.path.join(base_path, 'datasets', 'washingtondb-v1.0', 'data', 'line_images_normalized')  # Update this path to where your images are stored
    # destination_folder = os.path.join(base_path, 'datasets', 'washingtondb-v1.0', 'data', 'normalized_size')  # Update this path to where you want to save the resized images
    # resize_image(source_folder, destination_folder)


if __name__ == "__main__":
    main()
