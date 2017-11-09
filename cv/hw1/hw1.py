import numpy as np
import cv2
import glob
from typing import List, Tuple
from itertools import combinations
X = 0
Y = 1

def readimages(paths: List[str], size: (int, int)):
    imgs = [cv2.imread(path) for path in paths]
    imgs = [cv2.resize(img, size) for img in imgs]
    return imgs

display_window = cv2.namedWindow('display', cv2.WINDOW_NORMAL)
def display(img):
    cv2.imshow('display', img)
    cv2.waitKey(0) & 0xff


def calibrate(img, bord_size=(5, 7)):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_MAX_ITER, 30, 0.001)
    obj_point = np.array([[x, y, 0] for x in range(bord_size[X]) for y in range(bord_size[Y])], np.float32)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(grey, bord_size, None)
    if success:
        corner2 = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, bord_size, corner2, success)
        display(img)
        return (True, obj_point, corners)
    else:
        return (False, None, None)


def checkboard_calibration(filenames):
    imgs = readimages(filenames)
    objpoints = []
    imgpoints = []
    h, w, _ = imgs[0].shape
    for (success, objpoint, corners) in map(calibrate, imgs):
        if success:
            objpoints.append(objpoint)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoint, imgpoints, (h, w), None, None)
    return mtx, rvec


def find_corners(image):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display(grey)
    corners = cv2.cornerHarris(grey, 5, 21, 0.1)
    corners = cv2.dilate(corners, None)
    ret, corners = cv2.threshold(corners, 0.01*corners.max(), 225, 0)
    corners = np.uint8(corners)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(grey, np.float32(centroids), (5, 5), (-1, -1), criteria)
    image = np.copy(image)
    corners = [(x, y) for x,y in corners]
    for corner in corners:
        cv2.circle(image, corner, 2, 255, -1)
    display(image)
    return corners


def order_points(points):
    """
    if the segment composed by first two segement
        intersect the segement composed by the second two
        segement, we recorder them
    """
    from operator import itemgetter
    p1 = max(points, key=itemgetter(X))
    points.remove(p1)
    p2 = max(points, key=itemgetter(Y))
    points.remove(p2)
    p3 = min(points, key=itemgetter(X))
    points.remove(p3)
    p4 = min(points, key=itemgetter(Y))
    return p1, p2, p3, p4


def find_rectangle(points):
    """
    To find the points, we assume that the four points should
    obey the relationship such that the distance between two points
    should equal to the distance between other two points
    """
    def d(p1, p2, p3, p4):
        x0 = p1[X]-p2[X]
        y0 = p1[Y]-p2[Y]
        x1 = p3[X]-p4[X]
        y1 = p3[Y]-p4[Y]
        d0 = x0*x0+y0*y0
        d1 = x1*x1+y1*y1
        return abs(d0 - d1) / (d0 + d1)
    z = points
    dist = float('inf') # a large number as infinity
    chosen = None
    for (p1, p2, p3, p4) in combinations(points, 4):
        p1, p2, p3, p4 = order_points([p1, p2, p3, p4])
        new_dist = d(p1, p2, p3, p4)+d(p1, p3, p2, p4)+d(p1, p4, p2, p3)+d(p1, p2, p2, p3)
        if dist > new_dist:
            dist = new_dist
            chosen = [p1, p2, p3, p4]
    return chosen


def wrapPerspective(image, points, size):
    dst = np.array([
        [0,0], [size-1, 0],
        [size-1, size-1], [0, size-1]
        ], dtype='float32')
    M = cv2.getPerspectiveTransform(points, dst)
    wraped = cv2.wrapPerspective(image, M, (size, size))
    return wraped


def drawRectangle(image, rect):
    img = np.copy(img)
    p1, p2, p3, p4 = rect
    cv2.line(img, p1, p2, 255, 2)
    cv2.line(img, p2, p3, 255, 2)
    cv2.line(img, p3, p4, 255, 2)
    cv2.line(img, p4, p1, 255, 2)
    display(img)


def calibration(obj_points, img_points, size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print(mtx)
    print(dist)
    print(rvecs)
    print(tvecs)
    for r,t in zip(rvecs, tvecs):
        print('%s, %s'%(r.T[0], t.T[0]))
    return cv2.getOptimalNewCameraMatrix(mtx, dist, size, 1, size) + (dist, mtx)


def undistort(img, cammtx, roi, dst, mtx):
    dst = cv2.undistort(img, mtx, dist, None, cammtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    display(dst)
    return dst


def contour_detection(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey, 127,255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if len(c) > 4]
    print(contours)
    img = np.copy(img)
    cv2.drawContours(img, contours, -1, (0, 255,0), 3)
    display(img)


def find_homography(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=500)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_features = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_features.append(m)
    img3 = cv2.drawKeypoints(img1, kp1, None, (0, 0, 255))
    display(img3)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_features]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_features]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.2)
    matchesMask = mask.ravel().tolist()
    img4 = cv2.drawMatches(img1, kp1, img2, kp2, good_features, None,
            matchColor=(255,0,0), singlePointColor=(255,0,0), flags=2)
    display(img4)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_features, None,
            matchColor=(0,255,0), singlePointColor=(255,0,0), matchesMask=matchesMask, flags=2)
    display(img3)
    return M


def run_calibration():
    size = (504, 378)
    checkboards = readimages(glob.glob('./calibrations/*.jpg'), size)
    obj_points, img_points = [], []
    for success, objpt, imgpt in map(calibrate, checkboards):
        if success:
            obj_points.append(objpt)
            img_points.append(imgpt)
    newcameramtx, roi, dist, mtx = calibration(obj_points, img_points, size)
    return newcameramtx, roi, dist, mtx

def run_perspective_wrap():
    size = (504, 378)
    stamp_size = (100, 100)
    stamp_imgs = readimages(glob.glob('./stamps/*.jpg'), size)
    stamp = readimages(['stamp-pos.jpg'], stamp_size)[0]
    for stamp_img in stamp_imgs:
        M = find_homography(stamp_img, stamp)
        if M is not None:
            new_img = cv2.warpPerspective(stamp_img, M, stamp_size)
            display(new_img)

if __name__=='__main__':
    run_calibration()
    run_perspective_wrap()

