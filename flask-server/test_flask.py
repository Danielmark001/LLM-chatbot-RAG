
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/test", methods=['GET'])
def test_endpoint():
    return jsonify({"message": "Flask server is working!"}), 200

if __name__ == "__main__":
    print("Starting test Flask server...")
    app.run(debug=True, port=5001)
