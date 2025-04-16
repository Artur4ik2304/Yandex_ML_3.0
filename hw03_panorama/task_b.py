import cv2

def extract_key_points(img1, img2):
    sift = cv2.SIFT_create()
    kpts1, desc1 = sift.detectAndCompute(img1, None)
    kpts2, desc2 = sift.detectAndCompute(img2, None)
    return kpts1, desc1, kpts2, desc2
