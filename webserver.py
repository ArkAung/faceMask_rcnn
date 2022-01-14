"""
    Simple Flask webserver which uses Mask RCNN trained to detect and segment face masks in images.

    Usage: python webserver.py --weights </path/to/downloaded/weight/file> --device [cpu|gpu]
    Example on running on CPU and weights file called face_mask.h5 reside in the same directory as code.
        * python webserver.py --weights face_mask.h5 --device cpu
"""
import argparse
import os

from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename

import constants
from infer import get_model, process_image

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def create_folders():
    if not os.path.isdir(constants.UPLOAD_DIR):
        os.makedirs(constants.UPLOAD_DIR)
    if not os.path.isdir(constants.RESULT_DIR):
        os.makedirs(constants.RESULT_DIR)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('upload.html')
    elif request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(success=False,
                           description="No file part")
        file = request.files['file']
        if file.filename == '':
            return jsonify(success=False,
                           description="No file selected for uploading")
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            _, coverage = process_image(model=mdl, img_path=upload_path,
                                        result_dir=constants.RESULT_DIR,
                                        output_filename=constants.OUTPUT_IMAGE_FILENAME)
            result_url = "{}{}/{}".format(request.url_root, constants.RESULT_DIR, constants.OUTPUT_IMAGE_FILENAME)
            return jsonify(success=True,
                           percent_masked=round(coverage, 3),
                           output=result_url)


@app.route('/results/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    results = os.path.join('.', constants.RESULT_DIR)
    return send_from_directory(directory=results, filename=filename)


if __name__ == "__main__":
    create_folders()

    parser = argparse.ArgumentParser(
        description='Flask webserver to detect face masks')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--device', required=False,
                        default='cpu',
                        help="Device to run evaluation on [cpu(default)|gpu]")
    parser.add_argument('--ngrok', required=False, dest="ngrok", action='store_true')
    args = parser.parse_args()

    assert args.device == 'cpu' or args.device == 'gpu', "Device should either be 'cpu' or 'gpu'"

    if args.ngrok:
        from flask_ngrok import run_with_ngrok
        run_with_ngrok(app)

    mdl = get_model(weight_path=args.weights, device=args.device)

    app.config['UPLOAD_FOLDER'] = constants.UPLOAD_DIR
    app.secret_key = 'supersecret'
    app.run()
