from flask import Flask, request
from PIL import Image

app = Flask(__name__)


import mood.emotion_service


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image']  # get the image
        threshold = request.form.get('threshold')
        threshold = 0.5 if threshold is None else float(threshold)
        container_size = request.form.get("containerSize")
        container_size = (640, 480) if container_size is None else tuple(map(int, container_size.split(",")))

        image_object = Image.open(image_file)
        objects = mood.emotion_service.get_emotions(image_object, threshold, container_size)
        return objects

    except Exception as e:
        print('POST /image error: %e', e)
        return e


# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')  # Put any other methods you need here
    return response


if __name__ == '__main__':
    app.run()
