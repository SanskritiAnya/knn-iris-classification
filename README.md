# **KNN Iris Classification**
This project implements the K-Nearest Neighbors algorithm to classify iris flower species using the classic Iris dataset. It covers data loading, feature normalization, training with multiple K values, model evaluation, and visualization of decision boundaries.

📌 **Project Overview**  
The Iris dataset consists of 150 samples with 4 numerical features representing sepal and petal dimensions. The goal is to classify each sample into one of three species. The project demonstrates the full KNN pipeline from data preprocessing to detailed model evaluation.

✅ **Key Steps Performed**  
📥 Data Loading & Exploration  
Loaded the Iris dataset, explored feature distributions, and examined class balance.

⚙️ Feature Normalization  
Applied StandardScaler to normalize features for improved KNN performance.

🤖 KNN Training & K Variation  
Trained KNN classifiers for K values from 1 to 10 and compared accuracies to select the optimal K.

📊 Model Evaluation  
Used confusion matrix and classification report to analyze model performance on test data.

🖼️ Decision Boundary Visualization  
Visualized decision boundaries using petal length and petal width features to interpret model behavior.

🛠️ **Tools & Libraries Used**  
- Python 3  
- Pandas – Data manipulation and exploration  
- Scikit-learn – Dataset, preprocessing, KNN model, evaluation metrics  
- Matplotlib – Plotting decision boundaries and results  
- Seaborn – Confusion matrix visualization  

📁 **Project Structure**  
knn-iris-classification/  
├── 01_load_explore_dataset.py # Load and explore Iris dataset  
├── 02_normalise_features.py # Normalize dataset features  
├── 03_knn_training_kvariation.py # Train KNN models with different K values  
├── 04_evaluation_confusion_matrix.py # Evaluate model with confusion matrix & report  
├── 05_visualize_decision_boundaries.py # Visualize decision boundaries  
└── README.md # Project documentation  

🎯 **Goal**  
To build an effective KNN classifier on the Iris dataset, understanding the impact of feature scaling and parameter tuning, and interpreting model decisions visually.

🙌 **Acknowledgements**  
Dataset source: Iris dataset from Scikit-learn

📬 **Contact**  
Sanskriti Anya  
📧 sanskritianya17@gmail.com  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/sanskriti-anya-6bb2b4332)
