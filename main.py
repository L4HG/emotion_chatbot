# Python libraries that we need to import for our bot
import random
from config import *
import io
from PIL import Image  # $ pip install pillow
import face_recognition_models
import dlib
import numpy as np
import cv2
import urllib.request
from collections import OrderedDict
import time

import telegram
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler)

face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

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


from catboost import CatBoostClassifier

model = CatBoostClassifier(loss_function='MultiClass')
model.load_model(fname='model_yan.cbm')

file_path = '/var/opt/emotion_chatbot_db/'


# file_path = ''

class Handlers:
    @staticmethod
    def start_command(bot, update, user_data):
        user = update.message.from_user
        chat_id = update.message.chat.id
        if 'current_user' in user_data:
            current_user = user_data['current_user']
        else:
            current_user = user.username
            user_data['current_user'] = current_user

        message = 'Добро пожаловать! Вы можете оставить свой отзыв в виде фото, всем улыбкам будет подарок.'
        bot.sendMessage(chat_id, message)
        return

    @staticmethod
    def text_command(bot, update, user_data):
        user = update.message.from_user
        chat_id = update.message.chat.id
        if 'current_user' in user_data:
            current_user = user_data['current_user']
        else:
            current_user = user.username
            user_data['current_user'] = current_user

        message = 'Спасибо за отзыв! Вы можете оставить отзыв и в виде фотографии.'
        bot.sendMessage(chat_id, message)
        input_message_id = update.message.message_id
        bot.forward_message(chat_id='-100' + '1324810869',
                            from_chat_id=chat_id,
                            message_id=input_message_id)
        return

    @staticmethod
    def photo_handler(bot, update, user_data):
        user = update.message.from_user
        chat_id = update.message.chat.id
        if 'current_user' in user_data:
            current_user = user_data['current_user']
        else:
            current_user = user.username
            user_data['current_user'] = current_user
        input_message_id = update.message.message_id
        bot.forward_message(chat_id='-100' + '1324810869',
                            from_chat_id=chat_id,
                            message_id=input_message_id)

        photo_file = bot.getFile(update.message.photo[-1].file_id)
        photo_file_name = file_path + '{}.jpg'.format(str(photo_file.file_id).replace('-', ''))
        photo_file.download(photo_file_name)

        try:
            im = Image.open(photo_file_name)
            print(im)
            image_np = np.array(im)
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            rects = face_detector(gray, 1)

            for (i, rect) in enumerate(rects):
                shape = pose_predictor_68_point(gray, rect)
                shape = shape_to_np(shape)

                shape_r = shape_rotate(shape)

                shape_n = shape_normalize(shape_r)
                shape_nn = shape_normalize(shape)

                X_input = []

                X_input.append(np.hstack(shape_n))

                X_input = np.array(X_input)
                X_input.shape
                proba_predict = model.predict_proba(X_input)[0]

                proba_dict = {i: p for (i, p) in enumerate(proba_predict)}

                image_zeros = np.zeros(255 * 255 * 3, dtype='uint8').reshape(255, 255, 3)
                image_zeros[:, :] = (255, 255, 255)
                for (x, y) in shape_nn:
                    cv2.circle(image_zeros, (int(x * 255), int(y * 255)), 1, (0, 0, 255), -1)
                short_file_name = '{}.jpg'.format(time.time())
                file_name = file_path + '{}'.format(short_file_name)
                cv2.imwrite(file_name, image_zeros, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # bot.send_text_message(recipient_id, image_url)
                bot.send_photo(chat_id=chat_id, photo=open(file_name, 'rb'))
                emotions = ''
                for emotion_i in proba_dict.keys():
                    emotions = emotions + '{}: {} \r\n'.format(EMO_DICT[emotion_i], proba_dict[emotion_i])
                message = 'Ваши эмоции:\r\n{}'.format(emotions)
                bot.sendMessage(chat_id, message)
                bot.sendMessage(chat_id='-100' + '1324810869', text=message)

                if proba_dict[5] > 0.1 and user_data.get('code_sent', 0) != 1:
                    happy_message = ('Спасибо за улыбку!\r\n' +
                                     'Специально для вас мы подготовили скидочный код ' +
                                     'на подписку курса о машинном обучении в бизнесе \r\n' +
                                     'Код: BinaryCV5\r\n' +
                                     'Для получения информации о данном курсе, пришлите данный код на Email: course@arboook.com'
                                     )
                    bot.sendMessage(chat_id, happy_message)
                    bot.sendMessage(chat_id='-100' + '1324810869', text=happy_message)
                    user_data['code_sent'] = 1
                elif user_data.get('code_sent', 0) == 1:
                    message = 'Так красиво улыбаетесь!'
                    bot.sendMessage(chat_id, message)
                elif proba_dict[5] < 0.1:
                    message = 'Не грустите, машинное обучение развеселит вас!'
                    bot.sendMessage(chat_id, message)
        except Exception as e:
            # bot.sendMessage(chat_id, str(e))
            bot.sendMessage(chat_id='-100' + '1324810869', text=str(e))

        return


def main():
    updater = Updater(token=bot_token)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler('start', Handlers.start_command, pass_user_data=True))
    dispatcher.add_handler(MessageHandler(Filters.text, Handlers.text_command, pass_user_data=True))

    dispatcher.add_handler(MessageHandler(Filters.photo, Handlers.photo_handler, pass_user_data=True))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
