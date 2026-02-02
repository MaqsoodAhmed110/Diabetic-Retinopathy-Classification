from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import torchvision.models as models

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "static/uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Class Labels
class_labels = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative_DR"
}

# Load the trained model
model = models.resnet50(pretrained=False)  # Ensure same architecture
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 5 classes
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()  # Set to evaluation mode

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to Classify Image
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return class_labels[predicted_class]  # Convert number to class name

# Route for Home Page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No selected file")
        
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            prediction = classify_image(file_path)
            return render_template("index.html", prediction=prediction, image=file.filename)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
