import os
from typing import Any

from flask import Blueprint, current_app, send_from_directory

bp = Blueprint(
    "client",
    __name__,
)


@bp.route("/", defaults={"path": ""})
@bp.route("/<path:path>")
def catch_all(path) -> Any:
    if path != "" and os.path.exists(os.path.join(current_app.static_folder, path)):
        return send_from_directory(current_app.static_folder, path)
    return send_from_directory(current_app.static_folder, "index.html")
