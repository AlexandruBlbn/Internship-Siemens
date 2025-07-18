import threading
from flask import Flask, request
from PyQt5.QtCore import QObject, pyqtSignal

class NetworkReceiver(QObject):
    text_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.app = Flask("CurabotNet")
        self.app.add_url_rule('/speech','recv',self._recv,methods=['POST'])

    def _recv(self):
        data = request.json or {}
        text = data.get('text','')
        if text:
            self.text_received.emit(text)
            return {'status':'ok'}
        return {'status':'error'}

    def start(self, host='0.0.0.0', port=5000):
        t = threading.Thread(
            target=lambda: self.app.run(host=host,port=port,debug=False,use_reloader=False),
            daemon=True
        )
        t.start()