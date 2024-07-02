
# Market Sentiment Analysis

This project performs sentiment analysis on financial news using the GPT-2 model to classify text into positive, negative, or neutral categories. The analysis helps in understanding market sentiment towards specific stocks, potentially predicting their price movements.




## Features
#### Sentiment Analysis: Classifies financial news into positive, negative, or neutral categories.

#### Model Training: Fine-tunes GPT-2 with added dropout to improve model performance.

#### Data Tokenization: Utilizes GPT-2 tokenizer for text preprocessing.

#### Evaluation: Provides training and validation accuracy to assess model performance.

#### Prediction: Offers functionality to predict sentiment for new text inputs.
## Tech Stack

#### Programming Language: Python
#### Machine Learning Framework: PyTorch
#### Transformers Library: Hugging Face Transformers
#### Pre-trained Model: GPT-2
#### Data Processing: Pandas, NumPy
#### Evaluation: Scikit-learn
#### Data Loading: PyTorch DataLoader
#### Tokenization: GPT-2 Tokenizer
## Installation and Reuse

Clone the repository using:
```git clone https://github.com/ayush-2405/Market-Sentiment-Analysis.git```

Use ```model.load_state_dict(torch.load('best_model.pt'))``` to load the pre-trained best model provided in the repository.

Run the following code:

```text = "Financial news about a stock here"```
```predicted_label = predict_single_text(model, tokenizer, text, MAX_LEN, device)```
```print(f"Predicted Label: {predicted_label}")```

Model would provide a predicted lable.

'negative': 0, 'neutral': 1, 'positive': 2
