import cv2
import glob
import numpy as np


class TwitterFollowingButtonCount:
    """
    Class to find Following Button in Twitter Images
    """

    def __init__(self, directory):
        """
        Function to read following button image from given directory
        :return:
        """

        self.following_button_image = cv2.imread(directory, 0)

        # Capture width and height of following button image
        self.width, self.height = self.following_button_image.shape[::-1]

    # Following Button Image
    following_button_image = any
    width = any
    height = any

    def read_twitter_images(self, directory):
        """
        Function to read all images from directory
        :param directory: path to the directory for test images
        """

        for test_image_name in glob.glob(directory + '*.png'):
            model_result = self.model(directory, test_image_name.split('\\')[1])

            # Final output the Number of following buttons in test image
            print("Number of boxes found: {0}".format(model_result['box_count']))
            print(model_result['corners'])

    def model(self, directory, test_image_name):
        raw_corners = []
        corners = []
        box_count = 0

        # Read test image
        test_image = cv2.imread(directory + test_image_name)

        # Grey Scaling to pick easily
        test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        # find following_button
        results = cv2.matchTemplate(test_image_gray, self.following_button_image, cv2.TM_CCOEFF_NORMED)

        # Most precise value chosing threshold
        threshold = 0.94

        # Location for bright corners
        location = np.where(results >= threshold)

        # X, Y corners but un-ordered
        raw_corners.append([set(location[1]), set(location[0])])

        # Arranging in order X, Y coordinates to find following button
        for raw_corner in raw_corners:
            xmin = np.mean(np.array(list(raw_corner[0])), dtype='int32')
            xmax = xmin + self.width

            y_values = sorted(list(raw_corner[1]))
            for index in range(len(y_values)):
                box_count = box_count + 1
                ymin = y_values[index]
                ymax = y_values[index] + self.height

                # Corner ordering
                corners.append([ymin, xmin, ymax, xmax])

        return {
            'box_count': box_count,
            'corners': corners
        }


if __name__ == "__main__":
    twitterFollowingButtonCount = TwitterFollowingButtonCount(r'following_button/following_button.png')
    twitterFollowingButtonCount.read_twitter_images('test_images/')
