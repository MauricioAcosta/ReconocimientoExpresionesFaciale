import cv2
import time
from Predict import predict

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        time.sleep(1)
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        print('img.shape', img.shape)
        predict(img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()