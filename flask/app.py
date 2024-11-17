import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, redirect, url_for
import csv
from werkzeug.utils import secure_filename

# Ajouter le chemin YOLOv5 au sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "yolov5"))

from models.common import DetectMultiBackend  # Utilisation de YOLOv5
from segment_anything import SamPredictor, sam_model_registry  # Import SAM

# Initialiser l'application Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Initialiser YOLOv5
weights_path = Path('../yolov5/runs/train/exp5/weights/best.pt')
device = 'cpu'
model = DetectMultiBackend(weights_path, device=device)

# Charger et initialiser SAM
sam_checkpoint = "C:/Users/mon pc/OneDrive/Bureau/projet_model_deployment/segment-anything/vit_h.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam_predictor = SamPredictor(sam)

# Préparation du fichier CSV pour les résultats
output_csv = Path('./detection_results.csv')
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'Component', 'Area', 'VoidPercentage', 'MaxVoidPercentage'])

# Fonction pour traiter l'image avec SAM et YOLOv5
def process_image(image_path):
    # Charger l'image
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Annotation avec SAM (exemple avec des points interactifs)
    sam_predictor.set_image(img_rgb)
    input_point = np.array([[100, 100], [200, 200]])  # Exemple de points
    input_label = np.array([1, 0])  # 1 = composant, 0 = non-composant
    masks, scores, logits = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # Calculs
    component_count = np.sum(masks[0] == 1)
    void_area_total = np.sum(masks[0] == 0)
    img_area = h * w
    void_percentage = (void_area_total / component_count) * 100 if component_count else 0
    max_void_percentage = np.max(masks[0]) / component_count * 100 if component_count else 0

    return img, component_count, void_area_total, void_percentage, max_void_percentage

# Route principale pour le téléchargement de l'image
import os
from werkzeug.utils import secure_filename

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Rendre le nom de fichier sûr
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Traitement de l'image
            img, component_count, area, void_percent, max_void_percent = process_image(image_path)

            # Écriture des résultats dans le CSV
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, component_count, area, f"{void_percent:.2f}%", f"{max_void_percent:.2f}%"])

            # Redirection vers la page des résultats
            return redirect(url_for("results", image=filename))

    return render_template("index.html")


# nouvelle rroute du fichier uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))



# Route pour afficher les résultats
@app.route("/results")
def results():
    image = request.args.get("image")
    data = []
    # Charger les données du fichier CSV
    with open(output_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Construire l'URL pour accéder à l'image dans 'uploads'
    image_url = url_for('uploaded_file', filename=image)

    return render_template("results.html", data=data, image=image, image_url=image_url)


# Lancer l'application
if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
