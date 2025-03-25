from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("ai_sign_lstm_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json['keypoints']).reshape(1, -1)
    prediction = model.predict(data)
    gesture = np.argmax(prediction)
    return jsonify({"gesture": gesture})

if __name__ == '__main__':
    app.run(debug=True)
