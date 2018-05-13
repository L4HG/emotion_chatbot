# Python libraries that we need to import for our bot
import random
from flask import Flask, request
from pymessenger.bot import Bot

app = Flask(__name__)
ACCESS_TOKEN = 'ACCESS_TOKEN'
VERIFY_TOKEN = 'VERIFY_TOKEN'
from config import *
bot = Bot(ACCESS_TOKEN)

import io
from PIL import Image # $ pip install pillow
import face_recognition_models
import dlib
import numpy as np
import cv2
import urllib.request
from collections import OrderedDict

face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

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
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

from catboost import CatBoostClassifier
model = CatBoostClassifier(loss_function='MultiClass')
model.load_model(fname='model_yan.cbm')

# We will receive messages that Facebook sends our bot at this endpoint
@app.route("/", methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        """Before allowing people to message your bot, Facebook has implemented a verify token
        that confirms all requests that your bot receives came from Facebook."""
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    # if the request was not get, it must be POST and we can just proceed with sending a message back to user
    else:
        # get whatever message a user sent the bot
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for x in messaging:
                if x.get('message'):
                    recipient_id = x['sender']['id']
                    if x['message'].get('text'):
                        message = x['message']['text']
                        bot.send_text_message(recipient_id, message)
                    if x['message'].get('attachments'):
                        for att in x['message'].get('attachments'):
                            if att['type'] == "image" or True:
                                temp = io.BytesIO()
                                out_file = io.BytesIO()
                                im = Image.open(urllib.request.urlopen(att['payload']['url']))
                                image = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                                # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                                # detect faces in the grayscale image
                                rects = face_detector(gray, 1)

                                # loop over the face detections
                                for (i, rect) in enumerate(rects):
                                    # determine the facial landmarks for the face region, then
                                    # convert the facial landmark (x, y)-coordinates to a NumPy
                                    # array
                                    shape = pose_predictor_68_point(gray, rect)
                                    shape = shape_to_np(shape)

                                    shape_r = shape_rotate(shape)

                                    shape_n = shape_normalize(shape_r)

                                    X_input = []

                                    X_input.append(np.hstack(shape_n))

                                    X_input = np.array(X_input)
                                    X_input.shape
                                    proba_predict = model.predict_proba(X_input)

                                    proba_dict = {i: p for (i, p) in enumerate(proba_predict)}

                                    predict = model.predict(X_input)
                                    num_predict = proba_predict.argmax(axis=1)[0]
                                    # print(proba_predict,predict, proba_predict.argmax(axis=1)[0])
                                    emotion = EMO_DICT[num_predict] + ' probab = {0:.2f}%'.format(
                                        proba_predict[0][num_predict])

                                    # convert dlib's rectangle to a OpenCV-style bounding box
                                    # [i.e., (x, y, w, h)], then draw the face bounding box
                                    (x, y, w, h) = rect_to_bb(rect)
                                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                    # show the face number
                                    cv2.putText(image, "Emotion {}".format(emotion), (x - 10, y - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                    # loop over the (x, y)-coordinates for the facial landmarks
                                    # and draw them on the image
                                    for (x, y) in shape:
                                        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                                img_str = cv2.imencode('.jpg', image)[1].tostring()
                                bot.send_attachment(recipient_id, att['type'], io.StringIO(img_str))
                else:
                    pass
    return "Message Processed"


def verify_fb_token(token_sent):
    # take token sent by facebook and verify it matches the verify token you sent
    # if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


# chooses a random message to send to the user
def get_message():
    sample_responses = ["You are stunning!", "We're proud of you.", "Keep on being you!",
                        "We're greatful to know you :)"]
    # return selected item to the user
    return random.choice(sample_responses)


# uses PyMessenger to send response to user
def send_message(recipient_id, response):
    # sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"


if __name__ == "__main__":
    app.run(host="0.0.0.0")