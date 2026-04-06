import os
import random
import cv2
import numpy as np

#SETTINGS
IMAGE_SIZE = 64
SAMPLES_PER_CLASS = 1000
OUTPUT_DIR = "factory_data"

def create_dataset():
    #Create folders
    categories = ['good', 'defective']
    for category in categories:
        path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(path, exist_ok=True)
        
    print(f"Generating {SAMPLES_PER_CLASS * 2} synthetic images...")

    for i in range(SAMPLES_PER_CLASS):
        #Create a blank "metal" surface (gray noise)
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        noise = np.random.randint(100, 200, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        img = cv2.add(img, noise) # Base metal texture

        # Save the "Good" version
        cv2.imwrite(f"{OUTPUT_DIR}/good/part_{i}.png", img)

        #Create the "Defective" version (Add a scratch)
        start_pt = (random.randint(0, IMAGE_SIZE), random.randint(0, IMAGE_SIZE))
        end_pt = (random.randint(0, IMAGE_SIZE), random.randint(0, IMAGE_SIZE))
        
        #Draw the crack (color=50 is dark gray, thickness=2)
        cv2.line(img, start_pt, end_pt, 50, 2)
        
        #Save the "Defective" version
        cv2.imwrite(f"{OUTPUT_DIR}/defective/part_{i}.png", img)

    print("Virtual factory run complete. Data stored in 'factory_data' folder.")

if __name__ == "__main__":
    create_dataset()