# British Airways Data Science Project

This repository contains a comprehensive data science project for **British Airways**, focusing on customer feedback analysis and predictive modeling for flight bookings.

## Project Overview

The project is divided into two main tasks:
1. **Customer Sentiment Analysis**: Scraping and analyzing customer reviews to understand brand perception.
2. **Predictive Modeling**: Building a machine learning model to predict the likelihood of a customer completing a booking.

---

## Task 1: Customer Sentiment Analysis

### Goal
To scrape customer reviews from [Skytrax](https://www.airlinequality.com/airline-reviews/british-airways) and apply NLP techniques to extract insights into customer satisfaction.

### Files
- `getting_started.ipynb`: Initial notebook for web scraping reviews using `BeautifulSoup`.
- `analysis.ipynb`: The primary analysis notebook covering data cleaning, sentiment scoring (VADER), and visualization.
- `data/BA_reviews.csv`: The dataset containing scraped reviews.
- `visuals/Task-1/`: Visual outputs including word clouds, sentiment distributions, and common phrases.

### Key Techniques
- Web Scraping with `requests` and `BeautifulSoup`.
- Text Cleaning and Preprocessing (NLTK, Regex).
- Sentiment Analysis using `VADER`.
- Topic and Frequency Analysis.

---

## Task 2: Booking Completion Prediction

### Goal
To build a predictive model that identifies customers most likely to complete a flight booking based on their behavior and flight details.

### Files
- `Getting Started ML.ipynb`: Exploratory Data Analysis (EDA) and initial modeling steps.
- `Task_2.py`: The production-ready script for the final machine learning pipeline.
- `data/customer_booking.csv`: Dataset containing features like lead time, flight duration, and number of passengers.
- `visuals/Task-2/`: Model evaluation charts (ROC curves, confusion matrices, feature importance).

### Machine Learning Pipeline
- **Preprocessing**: Handling missing values, scaling numerical features, and one-hot encoding categorical variables.
- **Model**: `RandomForestClassifier`.
- **Evaluation**: 5-fold Stratified Cross-Validation, AUC-ROC, and Precision-Recall metrics.

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Resources**:
   Ensure you have the required NLTK data (punkt, stopwords, vader_lexicon) which are automatically downloaded in the notebooks.

---

## Tech Stack
- **Language**: Python 3
- **Libraries**:
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `wordcloud`
  - NLP: `nltk`, `beautifulsoup4`
  - Machine Learning: `scikit-learn`

---

## Presentations
The project includes presentation templates for both tasks (`Presentation Template - Task 1/2.pptx`) to summarize insights for stakeholders.
