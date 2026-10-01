"""
app.py
------
Flask web application that lets a user upload an image, runs it through the
trained Xception deepfake-detection model, and displays the REAL/FAKE
prediction with a confidence score.
"""

import os
from pathlib import Path
from flask import Flask, render_template, request

from src.predict import predict_image

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    uploaded_image_url = None

    if request.method == "POST":
        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please choose an image file to upload."
        elif not allowed_file(file.filename):
            error = "Unsupported file type. Please upload a PNG, JPG, JPEG, or BMP image."
        else:
            save_path = UPLOAD_DIR / file.filename
            file.save(save_path)
            uploaded_image_url = f"/static/uploads/{file.filename}"

            try:
                result = predict_image(save_path)
            except FileNotFoundError as e:
                error = str(e)
            except Exception as e:
                error = f"Prediction failed: {e}"

    return render_template(
        "index.html", result=result, error=error, image_url=uploaded_image_url
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
