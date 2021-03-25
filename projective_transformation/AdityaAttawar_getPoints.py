import cv2

keypoints_left = []
keypoints_right = []
IMAGE_LEFT_OR_RIGHT = "left_or_right_image"


# callback function to register coordinates clicked on left and right images
###################################
def click_event(event, x, y, flags, params):
    global left_image, right_image
    if event == cv2.EVENT_LBUTTONDOWN:
        print(params[IMAGE_LEFT_OR_RIGHT])
        print('x-coordinate = ', x)
        print('y-coordinate = ', y)
        print('----------------------')
        if params[IMAGE_LEFT_OR_RIGHT] == "left":
            keypoints_left.append((x, y))
            cv2.circle(left_image, (x, y), 3, (255, 0, 0))
            cv2.imshow("left", left_image)
        elif params[IMAGE_LEFT_OR_RIGHT] == "right":
            keypoints_right.append((x, y))
            cv2.circle(right_image, (x, y), 3, (255, 0, 0))
            cv2.imshow("right", right_image)


# register click event on either left or right image
###################################
left_image = cv2.imread('left.jpg', 1)
cv2.imshow("left", left_image)
cv2.setMouseCallback('left', click_event,
                     param={IMAGE_LEFT_OR_RIGHT: "left", "image_object": left_image})
right_image = cv2.imread('right.jpg', 1)
cv2.imshow("right", right_image)
cv2.setMouseCallback('right', click_event,
                     param={IMAGE_LEFT_OR_RIGHT: "right", "image_object": right_image})
#images_stacked = np.concatenate((left_image, right_image), axis=1)
#cv2.imshow("images_stacked", images_stacked)
cv2.waitKey(0)
cv2.destroyWindow('left_image')


# Write left and right image keypoints to files
###################################
f_left = open("keypoints_leftImage.txt", "w")
for i in range(len(keypoints_left)):
    f_left.write("{},{}\n".format(keypoints_left[i][0], keypoints_left[i][1]))
f_left.close()

f_right = open("keypoints_rightImage.txt", "w")
for i in range(len(keypoints_right)):
    f_right.write("{},{}\n".format(keypoints_right[i][0], keypoints_right[i][1]))
f_right.close()

cv2.destroyAllWindows()