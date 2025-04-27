# Fake News Detection

This project focuses on building a **Fake News Detection** system using **Machine Learning** techniques.

It uses:
- **Natural Language Processing (NLP)** for text processing
- **Logistic Regression** for classification
- **Principal Component Analysis (PCA)** for visualization

---

## Project Structure

- **Dataset**: News articles labeled as real or fake.
- **Text Preprocessing**: Using `CountVectorizer` to convert text into numerical features.
- **Model Training**: Logistic Regression model is trained on the extracted features.
- **Dimensionality Reduction**: PCA is applied to reduce the 3000-dimensional feature space into 2 principal components for plotting decision boundaries.
- **Visualization**:
  - Decision boundary for Training set
  - Decision boundary for Test set

---

## Libraries Used

- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`
  - `CountVectorizer`
  - `PCA`
  - `LogisticRegression`
  - `train_test_split`

---

## How It Works

1. **Load and preprocess** the dataset.
2. **Vectorize** the news articles using `CountVectorizer` (max 3000 features).
3. **Split** the data into training and test sets.
4. **Apply PCA** to reduce feature dimensions from 3000 to 2.
5. **Train Logistic Regression** on the reduced features.
6. **Visualize** the decision boundary on both training and test sets.

---

## Results

- **Training Set**:  
  Decision boundary plotted using the first two principal components.

- **Test Set**:  
  Model performance is visualized by plotting the test points along with the decision boundary.

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/soruprohan/Fake-News-Detection.git
   cd fake-news-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Fake-News-Detection.ipynb
   ```

---

## Future Improvements

- Try advanced models like Random Forest, XGBoost. Though these don't yeild better accuracy in this case.
- Use more sophisticated NLP methods (TF-IDF, Word Embeddings).
- Deploy the model as a web application.

---

## Acknowledgements

- Dataset taken from kaggle. Link: 
https://www.kaggle.com/datasets/jainpooja/fake-news-detection
- Scikit-learn documentation.

---

> **Note**:  
> PCA was used only for visualization. The original model uses full 3000-dimensional features for actual prediction.

---

## Contributors
- ***Hassin Arman Nihal*** Github profile: https://github.com/hassin070 
- ***Md. Sorup Rohan*** Github Profile: https://github.com/soruprohan
