import os
import cv2
import numpy as np
from tkinter import filedialog

from matplotlib import pyplot as plt

if __name__ == '__main__':

    seen = False

    paths = filedialog.askopenfilenames(initialdir=os.getcwd(), title="Select Files",
                                        filetypes=(("jpg files", "*.jpg*"), ("png files", "*.png*")))
    save_path = 'C:/Users/Giulia Pezzutti/Documents/Datasets/Sophie Dataset Processed/'

    for path in paths:

        # path = 'C:/Users/Giulia Pezzutti/Documents/Datasets/Sophie Dataset/Faces_134_h.jpg'
        name = os.path.splitext(os.path.basename(path.rstrip(os.sep)))[0]

        # ORIGINAL

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if seen:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(save_path+name+'.jpg', img)

        # BLACK AND WHITE

        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if seen:
            cv2.imshow('image black and white', img_bw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(save_path+name+'_bw.jpg', img_bw)

        # SMALLER PIC

        h, w = img.shape[:2]

        scale_percent = 10  # percent of original size
        width = int(w * scale_percent / 100)
        height = int(h * scale_percent / 100)
        dim = (width, height)

        img_res = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        if seen:
            cv2.imshow('image resized', img_res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(save_path+name+'_res.jpg', img_res)

        # EDGE DETECTOR

        img_ed = cv2.Canny(img, 100, 200)

        if seen:
            cv2.imshow('image edges', img_ed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(save_path+name+'_edges.jpg', img_ed)

        # BLURRING

        img_blur = cv2.GaussianBlur(img, (99, 99), 50)

        if seen:
            cv2.imshow('image blurred', img_blur)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(save_path+name+'_blur.jpg', img_blur)

        # # CENTRAL BLURRED CIRCLE
        #
        # if "_h" in name:
        #     radius = h // 3
        # else:
        #     radius = w // 3
        # circle_mask = np.zeros_like(img)
        # circle_mask = cv2.circle(circle_mask, (w // 2, h // 2), radius, (255, 255, 255), -1)
        #
        # img_circ_in = np.where(circle_mask > 0, img_blur, img)
        #
        # if seen:
        #     cv2.imshow('masked image', img_circ_in)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #
        # cv2.imwrite(save_path+name+'_circ_in.jpg', img_circ_in)
        #
        # # PERIPHERAL BLURRED CIRCLE
        #
        # circle_mask = cv2.bitwise_not(circle_mask)
        #
        # img_circ_out = np.where(circle_mask > 0, img_blur, img)
        #
        # if seen:
        #     cv2.imshow('masked image', img_circ_out)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #
        # cv2.imwrite(save_path+name+'_circ_out.jpg', img_circ_out)

        # GRADUAL CIRCULAR BLURRING

        def blend_with_mask_matrix(src1, src2, mask):
            res_channels = []
            for c in range(0, src1.shape[2]):
                a = src1[:, :, c]
                b = src2[:, :, c]
                m = mask[:, :, c]
                result = cv2.add(
                    cv2.multiply(b, cv2.divide(np.full_like(m, 255) - m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
                    cv2.multiply(a, cv2.divide(m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
                    dtype=cv2.CV_8U)
                res_channels += [result]
            result = cv2.merge(res_channels)
            return result


        def blur_image(cv_image, rad, cent, gaussian_core, sigma_x):
            blurred = cv2.GaussianBlur(cv_image, gaussian_core, sigma_x)
            circle_not_mask = np.zeros_like(cv_image)
            cv2.circle(circle_not_mask, cent, rad, (255, 255, 255), -1)
            cv2.GaussianBlur(circle_not_mask, (101, 101), 111, dst=circle_not_mask)
            res = blend_with_mask_matrix(cv_image, blurred, circle_not_mask)
            return res


        center = (w // 2, h // 2)
        if "_h" in name:
            radius = h // 3
        else:
            radius = w // 3

        res = blur_image(img, radius, center, (101, 101), 0)
        cv2.imwrite(save_path+name+'_circ_blur.jpg', res)
        plt.imshow(res)
        plt.title('Gradual Blur \n With Circle Mask')
        plt.show()
