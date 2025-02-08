import cv2
import numpy as np
import matplotlib.pyplot as plt


#loading and showing an image
image = cv2.imread("ball.jpg")
# cv2.imshow("ball image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# converting image to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grayscale image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows


# saving the processed image
cv2.imwrite("grayscale_ball.jpg",gray)

#resizing an image
resized = cv2.resize(image, (30,30))
# cv2.imshow("resized image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#to rotate image
(h,w) = image.shape[:2]
center = (w//2, h//2)
m = cv2.getRotationMatrix2D(center,45,0.5)
rotated = cv2.warpAffine(image,m,(w,h))
# cv2.imshow("rotated image",rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()


#canny edge detection
edges = cv2.Canny(gray,50,150)
# cv2.imshow("edge detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#blurring an image
blurred = cv2.GaussianBlur(image, (15,15), 10)
# cv2.imshow("blurred image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()





#transforming football field
#perspective warping or birds eye view
import cv2
import numpy as np

image = cv2.imread("offside_detection2.png")

# Define 4 source points (corners of the field in the image)
pts1 = np.float32([[320, 50], [700, 50], [50, 500], [950, 500]])

# Define 4 destination points (top-down view)
pts2 = np.float32([[0, 0], [500, 0], [0, 300], [500, 300]])

# Compute transformation matrix
matrix = cv2.getPerspectiveTransform(pts1, pts2)

# Apply the warp perspective
warped = cv2.warpPerspective(image, matrix, (500, 300))
# cv2.imshow("Warped Image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

















# with automatic edge detection
#!!!!!!!!!!!!!!!!!!!!!not working currently!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def order_points(pts):
    """ Orders points in the correct sequence: top-left, top-right, bottom-left, bottom-right. """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference of points to determine top-left, top-right, etc.
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left has the smallest sum
    rect[1] = pts[np.argmin(diff)]  # Top-right has the smallest difference
    rect[2] = pts[np.argmax(diff)]  # Bottom-left has the largest difference
    rect[3] = pts[np.argmax(s)]  # Bottom-right has the largest sum

    return rect

def find_corners(image):
    """ Automatically detects the 4 main corners of the largest quadrilateral in the image. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur to remove noise
    edges = cv2.Canny(blurred, 50, 150)  # Apply Canny edge detection

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Get largest contours

    for c in contours:
        peri = cv2.arcLength(c, True)  # Perimeter of contour
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # Approximate polygon shape

        if len(approx) == 4:  # If the shape has 4 corners, we assume it's the field
            return order_points(approx.reshape(4, 2))

    return None  # If no quadrilateral is found

def warp_perspective(image, pts):
    """ Applies perspective transformation to warp the image into a top-down view. """
    if pts is None:
        print("⚠️ No quadrilateral detected. Please check the image.")
        return None

    # Define output size (change as needed)
    width, height = 500, 300

    # Define the destination points for the top-down view
    dst_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(pts, dst_pts)

    # Apply the transformation
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped

# Load image
image = cv2.imread("offside_detection2.png")  # Change this to your image file

# Find corners automatically
corners = find_corners(image)

# Warp perspective based on detected corners
warped_image = warp_perspective(image, corners)

# Show results
if warped_image is not None:
    cv2.imshow("Original Image", image)
    cv2.imshow("Warped Image", warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
