
import flask
from flask import request, jsonify, render_template

import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from char_rnn import generate_fake_words, generate_real_words

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/api/v1/fake_words/', methods=['GET'])
def api_fake_words():
    # Get parameters
    save_folder = request.args.get('save_folder', default=None, type=str)
    iteration_number = request.args.get('iteration_number', default=None, type=int)
    num_words = request.args.get('num_words', default=None, type=int)
    if (save_folder is None) or (iteration_number is None) or (num_words is None):
        return jsonify({
            'status': 'Error',
            'error_message': 'Invalid parameters'
        })

    # Run neural net
    words = generate_fake_words(save_folder=save_folder, iteration_number=iteration_number, num_words=num_words)

    # Return results
    return jsonify({
        'status': 'Success',
        'fake_words': words,
    })


@app.route('/api/v1/real_words/', methods=['GET'])
def api_real_words():
    # Get parameters
    save_folder = request.args.get('save_folder', default=None, type=str)
    num_words = request.args.get('num_words', default=None, type=int)
    if (save_folder is None) or (num_words is None):
        return jsonify({
            'status': 'Error',
            'error_message': 'Invalid parameters'
        })

    # Get real words
    words = generate_real_words(save_folder=save_folder, num_words=num_words)

    # Return results
    return jsonify({
        'status': 'Success',
        'real_words': words,
    })


# Run debug server
app.run(host='0.0.0.0')
