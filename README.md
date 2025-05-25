Project Title: Intrusion Detection Using Deep Learning Classifiers


Project Overview

Cyberattacks pose a significant threat to digital infrastructure, with intrusion detection being a critical component of cybersecurity. Traditional detection systems often rely on predefined rules or shallow learning methods, which are limited in adapting to evolving attack patterns and generate high false alarm rates. Moreover, manual inspection of network traffic is time-consuming and prone to oversight.

This project introduces an intelligent deep learning-based system for multi-class network intrusion detection, focusing on identifying various cyber threats such as Denial of Service (DoS), reconnaissance, backdoors, and malware attacks from normal traffic. The proposed solution utilizes a hybrid model combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. CNNs extract spatial features from network traffic data, while LSTMs learn temporal dependencies across sequences. Together, they enhance the detection capabilities of the system.

The system is trained and evaluated on the UNSW-NB15 dataset, using advanced preprocessing and performance tuning to ensure accurate, scalable, and real-time intrusion detection.


Theoretical Concepts



1. Deep Learning

A subfield of machine learning that uses layered neural networks to learn complex patterns in data. It enables systems to automatically extract high-level features from raw input.

2. Convolutional Neural Networks (CNN)
CNNs are powerful for spatial data processing. In this project, they are used to extract spatial features from the structured network data.

3. Long Short-Term Memory (LSTM)
LSTMs are a type of Recurrent Neural Network (RNN) designed to remember long-term dependencies. They analyze temporal behaviors in network traffic over time.

4. Hybrid Deep Learning Model
Combining CNN and LSTM leverages both spatial and sequential information, significantly enhancing detection performance.

5. Data Preprocessing Techniques

•	Label Encoding
•	Normalization
•	Feature Selection
•	Balancing the dataset




Workflow

1. Dataset Collection: Using the UNSW-NB15 dataset.

2. Data Preprocessing: Feature scaling, label encoding, and balancing.

3. Model Design: Creating a CNN-LSTM hybrid architecture.

4. Model Training: Training the model using TensorFlow/Keras.

5. Evaluation: Assessing model performance with accuracy, precision, recall, and F1-score.



Advantages

•	Detects a wide range of network intrusions.
•	Learns patterns automatically without manual rules.
•	Scalable and deployable in real-time systems.
•	Reduces false positives significantly.
•	Applicable in enterprise and cloud security.



Results Summary

Attack Type			Accuracy	Comments
DoS	97%	High precision and recall
Malware	95%		Effective detection of malware patterns
Backdoor	92%	Good results, improve with more data
Reconnaissance	93%	Strong results, can improve with tuning
Normal	98%	Excellent classification

		
	
		
		
		


Future Enhancements

•	Integration with real-time packet capture tools.
•	Deployment on cloud or edge infrastructure.
•	Visual interpretation using Explainable AI (XAI).
•	Inclusion of user behaviour and threat intelligence.
•	Testing with more modern datasets like CIC-IDS2018.


Technologies Used

Programming Language: Python
Deep Learning Frameworks: TensorFlow, Keras
Model Architecture: CNN + LSTM Hybrid
Visualization Tools: Matplotlib, Seaborn
Deployment Framework (Optional): Flask
Database: MySQL



Dataset Used

UNSW-NB15 Dataset
Link[https://www.kaggle.com/code/abdallahmohamedamin/sentiment-analysis-using-cnn-lstm-and-cnn-lstm]


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



