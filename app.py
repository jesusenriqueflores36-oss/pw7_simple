import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request

from pipeline import run_full_pipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    context = {}
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            context["error_msg"] = "No se recibió ninguna imagen."
            return render_template("index.html", **context)

        try:
            # Leer imagen en OpenCV (BGR)
            file_bytes = np.frombuffer(file.read(), np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imdecode devolvió None")
        except Exception as e:
            context["error_msg"] = f"No se pudo leer la imagen: {e}"
            return render_template("index.html", **context)

        # Carpeta donde se guardan resultados estáticos
        save_root = os.path.join("static", "results")
        os.makedirs(save_root, exist_ok=True)

        resultado = run_full_pipeline(bgr, save_root=save_root, prefix=str(uuid.uuid4())[:8])

        if not resultado.get("ok", False):
            context["error_msg"] = resultado.get("msg", "Error en el pipeline de modelos.")
            return render_template("index.html", **context)

        context.update(resultado)

    return render_template("index.html", **context)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
