# Emotion Classification in Social Media Using Multiple Text Representation and Deep Recurrent Neural Networks

## Table of Contents
- [Introduction](#introduction)
- [Paper Overview](#paper-overview)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Text Representation](#text-representation)
  - [Deep Recurrent Neural Networks](#deep-recurrent-neural-networks)
- [Experiments and Results](#experiments-and-results)
  - [Model Selection](#model-selection)
  - [Performance Metrics](#performance-metrics)
  - [Evaluation Results](#evaluation-results)
- [Results and Discussion](#results-and-discussion)
  - [Model Performance](#model-performance)
  - [Confusion Matrix Analysis](#confusion-matrix-analysis)
  - [Conclusion](#conclusion)
- [License](#license)

## Introduction
In today’s modern world, social media users often express their emotions, sharing their thoughts on personal experiences or societal issues, especially during critical times such as the COVID-19 pandemic. With more than 5.22 billion social media users worldwide, emotion analysis has become a vital area of research to understand public sentiment. This study investigates multiple text representation techniques (TF-IDF, Word2Vec, GloVe) combined with deep recurrent neural networks (GRU, LSTM, Bi-GRU, Bi-LSTM) to classify emotions in social media text.

## Paper Overview
This research evaluates emotion classification in social media by combining text representation techniques with deep learning models. We explore how well models perform on a dataset of 393,822 samples categorized into six emotions: sadness, joy, love, anger, fear, and surprise. The preprocessing steps, model configurations, and evaluation processes are outlined in the paper. Our findings suggest that Word2Vec with Bi-LSTM achieves the best F1-Score (0.9248), outperforming other configurations.

## Methodology
![Figure 1](https://github.com/user-attachments/assets/973d5393-54f5-44ed-a56b-8f0b007df11e)

### Data Preprocessing
The dataset used in this study consists of 416,809 unique text samples from Twitter, each labeled with one of six predominant emotions. The preprocessing steps include:
- **Noise Removal**: Removal of emojis, URLs, mentions, hashtags, numbers, and punctuation.
- **Spelling Correction**: Using SymSpell for typographical corrections.
- **Stopwords Removal**: Performed twice using the NLTK library and after lemmatization with spaCy.
- **Lemmatization**: Reducing words to their base forms while retaining meaning.

The data was split into training, validation, and testing sets using an 80:10:10 ratio. Stratified sampling was applied to maintain label balance, ensuring representative training.

### Text Representation
This study compares three text representation techniques for transforming text into numerical features:

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**: This technique assigns weights to terms based on their frequency within a document and rarity across the entire dataset.
2. **Word2Vec**: A neural network-based model that creates word embeddings by predicting the context surrounding a target word. The Skip-gram model is used with a context window of 5 words and a vector size of 75.
3. **GloVe (Global Vectors for Word Representation)**: Pre-trained embeddings based on co-occurrence frequencies across large text corpora. A 50-dimensional embedding is used for better representation.

### Deep Recurrent Neural Networks
For emotion classification, we experiment with four deep recurrent neural networks:
![Figure 4](https://github.com/user-attachments/assets/02d12541-26ec-4ba4-8ac6-1afb63506edf)

1. **GRU (Gated Recurrent Unit)**: Captures sequential dependencies with two gates: update and reset.
2. **LSTM (Long Short-Term Memory)**: Retains information over longer sequences using memory cells.
3. **Bi-GRU (Bidirectional GRU)**: Processes data in both forward and backward directions, capturing richer context.
4. **Bi-LSTM (Bidirectional LSTM)**: Enhances LSTM by processing data in both directions for better understanding.

## Experiments and Results

### Model Selection
The models were selected based on validation loss and F1-score. The highest-performing model for each text representation technique was chosen for final evaluation.

### Performance Metrics
Key metrics used to evaluate the models include:
- **Validation Loss**: Measures the model's ability to generalize on unseen data.
- **Test Loss**: Measures the model’s final performance after training.
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The proportion of positive results correctly identified.
- **Recall**: The proportion of actual positives correctly identified.
- **F1-Score**: The harmonic mean of precision and recall, prioritizing balanced performance.

### Evaluation Results
![Table](https://github.com/user-attachments/assets/bda84836-0a16-4077-8b54-49d85d318d1a)

Table above summarizes the performance of the best models across all text representation techniques. These models were evaluated based on the metrics mentioned above.
The best model for each text representation technique was then selected based on its highest F1-Score, which was prioritized due to the imbalanced nature of the dataset.
- **TF-IDF**: Bi-GRU achieved the best F1-Score of 0.8902.
- **Word2Vec**: Bi-LSTM achieved the highest F1-Score of 0.9201.
- **GloVe**: LSTM outperformed others with an F1-Score of 0.9176.

## Results and Discussion

### Model Performance
Our results indicate that **Word2Vec with Bi-LSTM** performed the best across all models, achieving the highest F1-Score of 0.9248. This configuration outperformed other combinations of text representation and deep recurrent models, highlighting the effectiveness of pre-trained embeddings like Word2Vec for capturing complex emotional nuances.

Comparing **TF-IDF** with the other representations, it is clear that while TF-IDF provides a strong baseline, it does not capture the semantic richness of words in the same way as **Word2Vec** or **GloVe**. Pre-trained embeddings demonstrate superior performance because they capture contextual relationships between words, which are crucial for emotion classification.

### Confusion Matrix Analysis
![Figure 8](https://github.com/user-attachments/assets/82cc6057-e4fb-46e6-9274-13df57b9370e)

In addition to performance metrics, we evaluated the confusion matrix for the best-performing models. The confusion matrix provides insights into how well the model is distinguishing between different emotion categories.

- **Word2Vec + Bi-LSTM**: The confusion matrix for this model showed high precision in recognizing joy, sadness, and anger, with lower misclassification between fear and surprise.
- **TF-IDF + Bi-GRU**: TF-IDF struggled with distinguishing between love and joy, resulting in more misclassifications.
- **GloVe + LSTM**: GloVe also exhibited strong performance with a high recall for sadness and joy but was prone to misclassifying surprise as fear.

This analysis confirms that pre-trained embeddings like Word2Vec offer richer semantic representations, which improve model accuracy across various emotional categories.

### Conclusion
The combination of pre-trained embeddings and deep recurrent neural networks such as **Bi-LSTM** yields the best performance for emotion classification in social media text. This study contributes valuable insights into the optimization of emotion classification tasks and is applicable in areas such as sentiment analysis, mental health monitoring, and trend detection on social media platforms.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
