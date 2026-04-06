import tensorflow as tf
import cv2
import numpy as np

#Load the trained brain
model = tf.keras.models.load_model('defect_inspector.h5')

def scan_image(image_path):
    #Load the image as grayscale
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        print("Error: Image not found.")
        return

    #Preprocess it for the brain (Resize to 64x64, normalize)
    img = cv2.resize(img_original, (64, 64))
    img = img.reshape(1, 64, 64, 1) # Add batch dimension
    img = img / 255.0

    #Predict
    prediction = model.predict(img)
    
    is_defective = prediction[0][0] < 0.5 
    
    result_text = "DEFECT DETECTED!" if is_defective else "PASSED QC"
    color = (0, 0, 255) if is_defective else (0, 255, 0) # Red or Green

    #Show Result
    display_img = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    cv2.putText(display_img, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('Quality Control Scanner', display_img)
    print(f"Scan Result: {result_text} (Score: {prediction[0][0]:.4f})")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    scan_image("factory_data/defective/part_5.png")