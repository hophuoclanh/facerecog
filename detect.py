import numpy as np
import cv2
import pickle

frontal_detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
profile_detector = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')

cap = cv2.VideoCapture("phuoclanh.mp4")

def get_area(box, coord="left-top-right-bottom"):
    if coord == "xywh":
        w, y, w, h = box
        area = w * h
        return area

    if coord == "left-top-right-bottom":
        left, top, right, bottom = box
        area = (right - left) * (bottom - top)
        return area
    return 0.0


def get_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # convert x, y, w, h to left, top, right, bottom
    # box 1
    left1, top1 = x1, y1
    right1 = left1 + w1
    bottom1 = top1 + h1
    # box 2
    left2, top2 = x2, y2
    right2 = left2 + w2
    bottom2 = top2 + h2

    # check whether one box contains the other, if yes, return iou score as 1
    if left2 > left1 and right2 < right1 and top2 > top1 and bottom2 < bottom1:
        return 1.0
    if left1 > left2 and right1 < right2 and top1 > top2 and bottom1 < bottom2:
        return 1.0

    # determine the coordinates of the intersection rectangle
    left = max([left1, left2])
    top = max([top1, top2])
    right = min([right1, right2])
    bottom = min([bottom1, bottom2])

    # check whether there is an intersection area
    if right < left or bottom < top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = get_area((left, top, right, bottom))

    # compute the area of box1 and box2
    box1_area = get_area(box1, coord = "xywh")
    box2_area = get_area(box2, coord = "xywh")

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def write_rect(image, box, color=((255, 0, 0)), stroke=2):
    x, y, w, h = box
    end_cord_x = x + w
    end_cord_y = y + h
    cv2.rectangle(image, (x, y), (end_cord_x, end_cord_y), color, stroke)

def detect_face( gray_image, detectors):
    all_faces = []
    for detector in detectors:
        detected = detector.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=5)
        all_faces.extend(detected)

    num_faces = len(all_faces)
    overlapped_indices = []
    print("the number of detected faces before applying iou")

    for i in range(num_faces):
        for j in range(i + 1, num_faces):
            face1 = all_faces[i]
            face2 = all_faces[j]
            iou_score = get_iou(face1, face2)
            if iou_score > 0.4:
                print("There is overlapped box, with the iou score ", iou_score)
                area1 = get_area(face1, coord= "xywh")
                area2 = get_area(face2, coord= "xywh")
                if area1 >= area2:
                    overlapped_indices.append(j)
                else:
                    overlapped_indices.append(i)

    overlapped_indices = set(overlapped_indices)

    print("This is the overlapped indices: ", overlapped_indices)
    print("The face list before removing overlapped area: ", all_faces)

    valid_faces = []
    for i, face in enumerate(all_faces):
        if i not in overlapped_indices:
            valid_faces.append(face)
    print("the face list after removing overlapped area: ", valid_faces)

    return valid_faces
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_faces = detect_face(gray, [frontal_detector, profile_detector])

    for face in detected_faces:
        color = (0, 250, 200)  # BGR 0-255 --> blue
        stroke = 2

        write_rect(frame, face, color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()