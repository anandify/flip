import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import os

# Define class names and transformations
class_names = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']
device = torch.device("cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
manual_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize
])

SHELF_LIFE = {
    "Apple": 7,    # Shelf life of 7 days
    "Banana": 6,   # Shelf life of 6 days
    "Orange": 10   # Shelf life of 10 days
}

# Model builder function
def create_model_baseline_effnetb0(out_feats: int, device: torch.device = None) -> torch.nn.Module:
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Modify the classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=out_feats, bias=True)
    ).to(device)

    model.name = "effnetb0"
    print(f"[INFO] Created a model: {model.name}")
    return model

# Load the model
def load_model():
    model = create_model_baseline_effnetb0(out_feats=len(class_names), device=device)
    model.load_state_dict(torch.load("effnetb0_freshness_10_epochs.pt", map_location=device))
    model.eval()
    return model

model = load_model()

# Prediction function
def predict_image_with_shelf_life(image: Image.Image):
    transformed_image = manual_transform(image).to(device)

    with torch.no_grad():
        logits = model(transformed_image.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=-1)
        predicted_index = probabilities.argmax(dim=-1).item()
        predicted_class = class_names[predicted_index]
        percentage = probabilities[0, predicted_index].item() * 100

    fruit_type = predicted_class.split()[-1]
    total_life_cycle = SHELF_LIFE.get(fruit_type, 0)

    if "Fresh" in predicted_class:
        remaining_days = round(total_life_cycle * (percentage / 100))
        age_in_days = total_life_cycle - remaining_days
        result = {
            "Prediction": predicted_class,
            "Confidence": f"{percentage:.1f}%",
            "TotalLifeCycle": f"{total_life_cycle} days",
            "EstimatedAge": f"{age_in_days} days",
            "RemainingDays": f"{remaining_days} days"
        }
    else:
        age_in_days = round(total_life_cycle * (percentage / 100))
        remaining_days = total_life_cycle - age_in_days
        result = {
            "Prediction": predicted_class,
            "Confidence": f"{percentage:.1f}%",
            "TotalLifeCycle": f"{total_life_cycle} days",
            "EstimatedAge": f"{age_in_days} days",
            "RemainingDays": f"{remaining_days} days"
        }

    return result

# Streamlit app
st.title("Fruit Freshness Prediction")

# Upload image
uploaded_file = st.file_uploader("Choose an image of fruit", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and process the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    result = predict_image_with_shelf_life(image)
    
    # Display result
    st.subheader("Prediction Results")
    st.write(f"**Prediction:** {result['Prediction']}")
    st.write(f"**Confidence:** {result['Confidence']}")
    st.write(f"**Total Life Cycle:** {result['TotalLifeCycle']}")
    st.write(f"**Estimated Age:** {result['EstimatedAge']}")
    st.write(f"**Days Remaining:** {result['RemainingDays']}")
