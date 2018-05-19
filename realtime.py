
from collections import OrderedDict
import numpy as np
import dlib
import cv2

EMO_DICT = {
    -1: 'Not file',
    0: 'Нейтральное',
    1: 'Злость',
    2: 'Презрение',
    3: 'Отвращение',
    4: 'Страх',
    5: 'Счастье',
    6: 'Грусть',
    7: 'Удивление'
}

EMO_DICT_EN = {
    -1: 'Not file',
    0: 'Neutral',
    1: 'Anger',
    2: 'Contempt',
    3: 'Disgust',
    4: 'Fear',
    5: 'Happy',
    6: 'Sad',
    7: 'Surprise'
}

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def shape_rotate(shape, dtype="float"):
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    leftEyeCenter = leftEyePts.mean(axis=0)
    rightEyeCenter = rightEyePts.mean(axis=0)

    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180.0
    coords = np.zeros((len(shape), 2), dtype=dtype)
    angle = angle * np.pi / 180.0
    for i in range(0, len(shape)):
        coords[i][0] = shape[i][0] * np.cos(angle) + shape[i][1] * np.sin(angle)
        coords[i][1] = -shape[i][0] * np.sin(angle) + shape[i][1] * np.cos(angle)

    return coords


def shape_normalize(shape):
    coords = np.zeros((len(shape), 2), dtype='float')

    maxX = np.amax(shape[:, 0])
    minX = np.amin(shape[:, 0])
    maxY = np.amax(shape[:, 1])
    minY = np.amin(shape[:, 1])

    scaleX = 1 / (maxX - minX)
    scaleY = 1 / (maxY - minY)

    scaleMin = scaleY

    if scaleX < scaleY:
        scaleMin = scaleX

    for i in range(0, len(shape)):
        coords[i][0] = (shape[i][0] - minX) * scaleMin
        coords[i][1] = (shape[i][1] - minY) * scaleMin

    return coords


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/kirillovchinnikov/.virtualenvs/chat_env/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')

EMO_DICT = {
    -1: 'Not file',
    0: 'Neutral',
    1: 'Anger',
    2: 'Contempt',
    3: 'Disgust',
    4: 'Fear',
    5: 'Happy',
    6: 'Sad',
    7: 'Surprise'
}


def shape_rotate(shape, dtype="float"):
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    leftEyeCenter = leftEyePts.mean(axis=0)
    rightEyeCenter = rightEyePts.mean(axis=0)

    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180.0
    coords = np.zeros((len(shape), 2), dtype=dtype)
    angle = angle * np.pi / 180.0
    for i in range(0, len(shape)):
        coords[i][0] = shape[i][0] * np.cos(angle) + shape[i][1] * np.sin(angle)
        coords[i][1] = -shape[i][0] * np.sin(angle) + shape[i][1] * np.cos(angle)
    return coords


def shape_normalize(shape):
    coords = np.zeros((len(shape), 2), dtype='float')

    maxX = np.amax(shape[:, 0])
    minX = np.amin(shape[:, 0])
    maxY = np.amax(shape[:, 1])
    minY = np.amin(shape[:, 1])

    scaleX = 1 / (maxX - minX)
    scaleY = 1 / (maxY - minY)

    scaleMin = scaleY

    if scaleX < scaleY:
        scaleMin = scaleX

    for i in range(0, len(shape)):
        coords[i][0] = (shape[i][0] - minX) * scaleMin
        coords[i][1] = (shape[i][1] - minY) * scaleMin

    return coords

cap = cv2.VideoCapture(0)

from catboost import CatBoostClassifier
model = CatBoostClassifier(loss_function='MultiClass')
model.load_model(fname='model_yan.cbm')

while True:
    ret, frame = cap.read()

    image = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    overlay = image.copy()
    output = image.copy()
    alpha = 0.8

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        shape_r = shape_rotate(shape)

        shape_n = shape_normalize(shape_r)

        X_input = []

        X_input.append(np.hstack(shape_n))

        X_input = np.array(X_input)
        X_input.shape
        proba_predict = model.predict_proba(X_input)[0]

        proba_dict = {i: p for (i, p) in enumerate(proba_predict)}

        emotions = ''
        for emotion_i in proba_dict.keys():
            emotions = emotions + '{}: {:.2f} \n'.format(EMO_DICT_EN[emotion_i], float(proba_dict[emotion_i]))


        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)

        x0, y0, dy = x + w, y+15, 20
        for i, line in enumerate(emotions.split('\n')):
            y_i = y0 + i * dy
            cv2.putText(overlay, line, (x0, y_i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # for (x, y) in shape:
        #     cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()