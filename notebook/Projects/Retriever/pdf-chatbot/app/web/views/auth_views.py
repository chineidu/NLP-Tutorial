from typing import Any

from flask import Blueprint, g, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from app.web.db.models import User

bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@bp.route("/user", methods=["GET"])
def get_user() -> Any:
    if g.user is not None:
        return g.user.as_dict()

    return jsonify(None)


@bp.route("/signup", methods=["POST"])
def signup() -> Any:
    email = request.json.get("email")
    password = request.json.get("password")

    user = User.create(email=email, password=generate_password_hash(password))
    session["user_id"] = user.id

    return user.as_dict()


@bp.route("/signin", methods=["POST"])
def signin() -> Any:
    email = request.json.get("email")
    password = request.json.get("password")

    user = User.find_by(email=email)

    if not check_password_hash(user.password, password):
        return {"message": "Incorrect password."}, 400

    session.permanent = True
    session["user_id"] = user.id

    return user.as_dict()


@bp.route("/signout", methods=["POST"])
def signout() -> Any:
    session.clear()
    return {"message": "Successfully logged out."}
