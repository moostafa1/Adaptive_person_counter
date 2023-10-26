import cv2
import numpy as np

# List to store points
points = []

def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the (x, y) coordinates to the list of points
        points.append((x, y))
        # Draw a small circle to indicate the point
        cv2.circle(temp_img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", temp_img)

    # Draw the polygon if there are at least 2 points
    if len(points) > 1:
        cv2.polylines(temp_img, [np.array(points)], False, (0, 255, 0), 2)
        cv2.imshow("Image", temp_img)

image = cv2.imread(r'counted images\to_count.jpg')
temp_img = image.copy()

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_polygon)

while True:
    cv2.imshow("Image", temp_img)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop when the ESC key is pressed
    if key == 27:
        break

# If you want to draw a closed polygon
if len(points) > 2:
    cv2.polylines(image, [np.array(points)], True, (0, 255, 0), 2)

# Save the modified image
cv2.imwrite('path_to_save_modified_image.jpg', image)
cv2.destroyAllWindows()
