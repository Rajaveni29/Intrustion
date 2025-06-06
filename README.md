Project Title: Intrusion Detection Using Deep Learning Classifiers

📘 Project Overview:
This project focuses on detecting network intrusions using deep learning models to enhance cybersecurity. By leveraging a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) model, the system identifies suspicious activities in network traffic and classifies them as normal or attack instances. The model is trained on labeled datasets and deployed via a user-friendly web application for real-time testing.

🎯 Objectives:
•	Detect different types of network intrusions with high accuracy.
•	Combine the power of CNN (for spatial features) and LSTM (for sequential patterns).
•	Provide a simple web interface for users to test the model with input data.

🗂 Project Structure:
intrusion-detector/
├── data/              # Dataset files (preprocessed & raw)
├── models/            # Trained CNN-LSTM models
├── notebooks/         # Jupyter notebooks for training and testing
├── app/               # Web interface (Flask/Django)
│   ├── static/        # CSS, JS, images
│   ├── templates/     # HTML templates
│   └── app.py         # Backend logic
├── utils/             # Preprocessing and helper functions
├── requirements.txt   # Python libraries
├── README.md          # Project documentation

💻 Technologies Used:
Languages: Python, HTML, CSS, JavaScript
Frameworks/Libraries:
•	TensorFlow / Keras
•	Pandas, NumPy
•	scikit-learn
•	Flask (for the web app)
Tools: Jupyter Notebook, VS Code, Git

📊 Dataset:
Public intrusion detection datasets like NSL-KDD Cup
Dataset Link  [https://www.kaggle.com/code/abdallahmohamedamin/sentiment-analysis-using-cnn-lstm-and-cnn-lstm]
Contains labeled records:
•	normal
•	DoS, Probe, R2L, U2R (various attack types)

🔁 Model Pipeline:
1. Data Cleaning & Normalization
2. Feature Selection
3. Model Building (CNN + LSTM)
4. Evaluation (Accuracy, Precision, Recall, F1-score)
5. Web App Integration

✅  Results:
Accuracy: 96% on the test set
Best Model: CNN-LSTM hybrid, outperforming standalone classifiers

🚀 Future Scope:
•	Deploy as a cloud-based security API.
•	Extend detection to real-time streaming data.
•	Add support for other network protocols and IoT traffic.

Team Members

Angel Rosini Marry G
Atchaya G
Hasini M
Rajaveni R

Guidance
Project Guide: Mrs. M.Priyadharshini,M.E.
Head of Department: Dr. K. Krishnakumari, Ph.D.
A.V.C. College of Engineering, Mayiladuthurai
Affiliated to Anna University, Chennai


License

This project is intended for academic and research use only.



