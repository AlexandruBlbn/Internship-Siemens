from flask import Flask, request

app = Flask(__name__)

@app.route('/speech', methods=['POST'])
def receive_speech():
    data = request.json
    print(f"Am primit text: {data.get('text')}")
    # Aici poți face ce vrei cu textul (ex: pornești o funcție, etc.)
    return {'status': 'ok'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
