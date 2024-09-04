import sys
import time

sys.path.append("./")
import argparse
from datetime import timedelta
from flask import Flask, render_template, request, session
import uuid

app = Flask(__name__)
app.secret_key = "12312321"  # 应该使用更加安全的密钥
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=10)
global_chat_id_to_chat_obj = {}


@app.route("/")
def index():
    session["chat_id"] = str(uuid.uuid4())
    from qabot import QaBot

    global_chat_id_to_chat_obj[session["chat_id"]] = (QaBot(args.llm_url), time.time())

    # 删除过期的qabot 对象
    dels_keys = []
    for key, (_, time_mark) in global_chat_id_to_chat_obj.items():
        if time.time() - time_mark >= 10 * 60:
            dels_keys.append(key)

    for key in dels_keys:
        del global_chat_id_to_chat_obj[key]

    return render_template("chat.html")


@app.route("/chat")
def chat():
    user_input = request.args.get("message", "")
    print("get", user_input)
    print("type", type(user_input))
    qabot, _ = global_chat_id_to_chat_obj[session["chat_id"]]
    global_chat_id_to_chat_obj[session["chat_id"]] = (qabot, time.time())
    return qabot.answer(user_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument("--llm_url", type=str, default="http://localhost:8017/generate", help="llm url")
    parser.add_argument("--port", type=int, default=8088, help="port")
    args = parser.parse_args()
    app.run(debug=True, port=args.port)
