# Python libraries that we need to import for our bot
import random
from flask import Flask, request
from pymessenger.bot import Bot

app = Flask(__name__)
ACCESS_TOKEN = 'ACCESS_TOKEN'
VERIFY_TOKEN = 'VERIFY_TOKEN'
from config import *
bot = Bot(ACCESS_TOKEN)

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
                        bot.send_message(recipient_id, message)
                    if x_m['message'].get('attachments'):
                        for att in x_m['message'].get('attachments'):
                            bot.send_message(recipient_id, att['type'])
                            bot.send_image_url(recipient_id, att['payload']['url'])
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