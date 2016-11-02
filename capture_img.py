import time
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print "Couldn't open cap"
else:
    for idx in range(10):
        ret, img = cap.read()
        if ret:
            cv2.imshow("Test {}, ret={!r}".format(idx, ret), img)
            time.sleep(2)
            cv2.destroyAllWindows()
        else:
            print "Couldn't read image {}".format(idx)

# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
