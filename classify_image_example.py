from PIL import Image
import os

def classify(image_path):
    # Display the image.
    display(Image.open(image_path))
    
    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)
   
    # Print the scores and names for the top-10 predictions.
    model.print_scores(pred=pred, k=10, only_first_name=True)

classify('inception/cropped_panda.jpg')
classify('mcf10a-fluorescent.jpg')
