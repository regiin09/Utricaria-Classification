from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

# Define current screenshot as source
source = 0

# Run inference on the source
results = model(source)  