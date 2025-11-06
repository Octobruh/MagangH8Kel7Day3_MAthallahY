import cv2
import numpy as np
import math as mt

# Perspective transform
# Selecting coords with regards to angle
# The following function is to calculate the trapezoid consequent of the perspective shift.
# Look at the pdf to see where this formula came to be
def calculate_coordinates(X, Y, L, H, theta):
    # Calculate sin(theta) and cos(theta)
    sin_theta = mt.sin(theta)
    cos_theta = mt.cos(theta)
    
    # Calculate the common fractional part to avoid repetition
    # (2L - H * sin(theta) * cos(theta)) / (2L + H * sin(theta) * cos(theta))
    try:
        numerator = 2 * L - H * sin_theta * cos_theta
        denominator = 2 * L + H * sin_theta * cos_theta
        
        if denominator == 0:
            print("Error: Division by zero. Denominator (2L + H*sin(theta)*cos(theta)) is zero.")
            return None
            
        common_factor = numerator / denominator
    except Exception as e:
        print(f"Error during calculation of common_factor: {e}")
        return None

    # Calculate Xtl
    # Xtl = 1/2 * [X - X * common_factor]
    Xtl = 0.5 * (X - X * common_factor)
    
    # Calculate Ytl
    # Ytl = Y - (H * sin(theta))
    Ytl = Y - (H * sin_theta)
    
    # Calculate Xtr
    # Xtr = 1/2 * [X + X * common_factor]
    Xtr = 0.5 * (X + X * common_factor)
    
    # Calculate Ytr
    # Ytr = Y - (H * sin(theta))
    Ytr = Y - (H * sin_theta)
    # Return the results in a dictionary for clarity
    return {
        'Xtl': Xtl,
        'Ytl': Ytl,
        'Xtr': Xtr,
        'Ytr': Ytr
    }


# IMPORTANT! These values are in pixels (px).
X_input = 640.0      #image resolution X
Y_input = 480.0      #image resolution Y
L_input = 150*4.267  #camera height from the ground ALSO IN PIXELS
H_input = 480.0      #ground area 'tallness'

# Need to actually measure later in Day 4

# Angle in degrees
theta_degrees = 76
# Convert degrees to radians for math functions
theta_radians = mt.radians(theta_degrees)

img = cv2.imread("Picsss.jpg")
frame = cv2.resize(img, (int(X_input), int(Y_input)))

coordinates = calculate_coordinates(X_input, Y_input, L_input, H_input, theta_radians)

tl = (int(coordinates['Xtl']), int(coordinates['Ytl']))
bl = (int(0), int(Y_input))
tr = (int(coordinates['Xtr']), int(coordinates['Ytr']))
br = (int(X_input), int(Y_input))

# Draw the trapezoid for visualisaton
cv2.circle(frame, tl, 5, (0,0,255), -1)
cv2.circle(frame, bl, 5, (0,0,255), -1)
cv2.circle(frame, tr, 5, (0,0,255), -1)
cv2.circle(frame, br, 5, (0,0,255), -1)

cv2.line(frame, tl, tr, (0,0,255), 2)
cv2.line(frame, tr, br, (0,0,255), 2)
cv2.line(frame, br, bl, (0,0,255), 2)
cv2.line(frame, bl, tl, (0,0,255), 2)

# Apply transformation
pts1 = np.float32([tl, bl, tr, br])
pts2 = np.float32([[0,0], [0,Y_input], [X_input,0], [X_input,Y_input]]) #original image corners

matrix = cv2.getPerspectiveTransform(pts1, pts2)
trans_frame = cv2.warpPerspective(frame, matrix, (int(X_input),int(Y_input))) #original image res

cv2.imshow('Ori', frame)
cv2.imshow('Transed', trans_frame)
cv2.imwrite('Transed.jpg', trans_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

#next, pass the transformed image to YOLO
#todo, test with YOLO, get real camera to calibrate