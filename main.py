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
import json
from requests_toolbelt import MultipartEncoder
import requests


def send_image_bytes(self, recipient_id, image_bytes):
    '''Send an image to the specified recipient.
    Image must be PNG or JPEG or GIF (more might be supported).
    https://developers.facebook.com/docs/messenger-platform/send-api-reference/image-attachment
    Input:
        recipient_id: recipient id to send to
        image_path: path to image to be sent
    Output:
        Response from API as <dict>
    '''
    payload = {
        'recipient': json.dumps(
            {
                'id': recipient_id
            }
        ),
        'message': json.dumps(
            {
                'attachment': {
                    'type': 'image',
                    'payload': {}
                }
            }
        ),
        'filedata': ('image_file', image_bytes)
    }
    multipart_data = MultipartEncoder(payload)
    multipart_header = {
        'Content-Type': multipart_data.content_type
    }
    return requests.post(self.base_url, data=multipart_data, headers=multipart_header).json()


bot.send_image_bytes = send_image_bytes

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
            for x_m in messaging:
                if x_m.get('message'):
                    recipient_id = x_m['sender']['id']
                    if x_m['message'].get('text'):
                        message = x_m['message']['text']
                        bot.send_text_message(recipient_id, message)
                    if x_m['message'].get('attachments'):
                        for att in x_m['message'].get('attachments'):
                            try:
                                im = Image.open(urllib.request.urlopen(att['payload']['url']))
                                image_np = np.array(im)
                                img_bytes = cv2.imencode('.jpg', image_np)[1].tostring()
                                bot.send_text_message(recipient_id, len(img_bytes))
                                bot.send_image(recipient_id, img_bytes)
                            except Exception as e:
                                bot.send_text_message(recipient_id, str(e))
                else:
                    pass
    return "Message Processed"


def verify_fb_token(token_sent):
    # take token sent by facebook and verify it matches the verify token you sent
    # if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'

if __name__ == "__main__":
    app.run(host="0.0.0.0")