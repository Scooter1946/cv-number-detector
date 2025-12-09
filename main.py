"""
Contributors:
    Justin Chen, chen5731@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from train_model import Net
import utils

def main():

    #START OF IMAGE PROCESSING STUFF
    
    # was originally gonna use gpu on my windows but cpu seems to work just fine
    device = "cpu" 

    # loading model
    model = Net().to(device)

    # tries to load model from the training file
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
    except FileNotFoundError:
        print("Error: model.pth not found. Run train_model.py first.")
        return
    model.eval() # puts model in "evaluation mode", disables dropout

    # camera setup
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) # width
    cap.set(4, 480) # height

    print("Starting camera feed.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            continue

        imgOriginal = img.copy()
        
        # converts recieved image to only show edges
        imgThres = utils.preProcess(img)
        # finds the paper in the image
        biggest = utils.findPaper(imgThres)
        
        if biggest.size != 0:
            # gets processed image of the paper "facing" the camera
            imgWarped = utils.getWarp(imgOriginal, biggest)

            
            #REST OF THIS IS NEURAL NETWORK STUFF

            # same as in train_model file, basically transforms image into data the model can use 
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            # unsqueeze adds a fake dim at the beginning of tensor (batch dim specifying # of batches, as in training)
            # dims: batches, channels, height, width
            imgTensor = transform(imgWarped).unsqueeze(0).to(device)
            
            # stops gradient calculation (since not training, no need to backpropogate)
            with torch.no_grad():
                output = model(imgTensor)
                # takes the value with the greatest probability
                prediction = output.argmax(dim=1, keepdim=True).item()
                confidence = torch.exp(output).max().item()
            
            if confidence > 0.9:
                print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
                
                # draw on screen
                cv2.putText(imgOriginal, f"Number: {prediction}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            elif confidence > 0.7: 
                print("Error: Prediction is uncertain. Try changing the lighting or moving the image.")
            # trace out the "paper" on the original image
            cv2.drawContours(imgOriginal, [biggest], -1, (0, 255, 0), 20)
            
            # debugging
            cv2.imshow("Warped", imgWarped)

        cv2.imshow("Result", imgOriginal)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
