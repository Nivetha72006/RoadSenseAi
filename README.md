# RoadSenseAi 🚗

An intelligent deep learning system for real-time vehicle type classification with confidence-based decision logic.
Built as part of a hackathon project, RoadSenseAI classifies vehicle images into 5 categories — including priority detection of ambulances for emergency response scenarios.

**Vehicle Classes**

Car · Bike · Bus · Truck · Ambulance (Priority Class)

**Features**

MobileNetV2 backbone with transfer learning and partial layer unfreezing
5-class classification with real-time image prediction
Confidence-based decision layer with 3 tiers
Ambulance priority detection with dedicated emergency logic
Streamlit web UI — clean multi-page interface
Fully offline — no external AI APIs used

**Confidence Decision System**

Confidence                   Decision

Ambulance ≥ 75%             🚨 High Priority Emergency     
≥ 85%                       ✅ High Confidence
65% – 84%                   ❓ Needs Review
< 65%                       ⚠️ Uncertain

**Tech Stack**

Python · TensorFlow · Keras · MobileNetV2 · Streamlit · NumPy · Pillow

**Project Structure**

RoadSenseAI/

├── app.py               # Streamlit UI


├── src/

    ├── train.py         # Model training script

    ├── decision.py      # Confidence decision logic

├── model/

     │   └── vehicle_model.keras

└── dataset/

    └── Bus/

    └── Car/

    └── Truck/

    └── Ambulance/

    └── Bike/
    
**How to Run**

pip install -r requirements.txt
streamlit run app.py

**Dataset**

Publicly available vehicle image dataset with ~100 images per class across 5 categories. 
Preprocessing includes resize to 224×224, normalization, augmentation (rotation, zoom, flip), and class weight balancing.
