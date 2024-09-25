from flask import Flask, request, jsonify

app = Flask(__name__)

captured_html = ""


@app.route('/get_html', methods=['POST'])
def get_html():
    global captured_html
    data = request.get_json()
    captured_html = data.get('html', '')
    print(captured_html)
    return jsonify(success=True)


@app.route('/show_html', methods=['GET'])
def show_html():
    return captured_html


if __name__ == '__main__':
    app.run(debug=True)
