## Capstone_Project - Intensity Analysis (Build your own model using NLP and Python)
An intelligent system using NLP to predict the intensity in the text reviews.

### Objective:
The objective of this project is to develop an intelligent system using NLP to predict the intensity
in the text reviews. By analyzing various parameters and process data, the system will predict the
intensity where its happiness, angriness or sadness. This predictive capability will enable to
proactively optimize their processes, and improve overall customer satisfaction.

### Purpose of this Project:
- **Prioritize Responses:**
Identify highly emotional reviews for faster response or resolution.
Ensure critical feedback is addressed promptly to maintain customer satisfaction.

- **Market and Sentiment Analysis:**
Monitor trends in customer sentiment over time.
Analyze competitor reviews for strategic insights.

- **Support Decision-Making:**
Assist businesses in prioritizing features or areas of service improvement by gauging emotional reactions.
Enable data-driven decision-making through emotional insights.

- **Automated Review Moderation:**
Flag harmful, fake, or inappropriate reviews for moderation.
Detect emotionally charged content to maintain platform quality.

- **Improve Interaction Platforms:**
Enhance conversational AI by tailoring responses to the emotional intensity of user input.
Detect and mitigate potentially harmful interactions by identifying extreme emotional expressions.

This project provides a foundation for actionable emotional insights, enabling businesses to adapt and respond effectively to customer needs and feedback.

### Methodology:
### 1. Collection of the data from any possible resources.:
The dataset for this project can be accessed by clicking the link provided below

[https://kh3-ls-storage.s3.us-east-1.amazonaws.com/Updated Project guide data set/Intensity_data.zip](https://kh3-ls-storage.s3.us-east-1.amazonaws.com/Updated%20Project%20guide%20data%20set/Intensity_data.zip)

**About the data:** This data set includes 3 csv files named angriness.csv, happiness.csv and sadness.csv, each conatining 2 columns namely content and intensity.

- content : Contains the customer reviews
- intensity : Describes the sentiment in the review which is angriness, sadness, happiness.

### 2. Exploratory Data Analysis (EDA):

**Purpose:** To understand the dataset structure, identify patterns, and detect outliers or anomalies.

**Techniques Used:**
- Checked the shape and information of the dataset, missing values, duplicated rows in the data set.
- Analyzed text length, word count, and frequency distributions.
- Generated visualizations such as histograms and word clouds to explore text data distribution.
- Examined class distributions and identified potential relationships between features.

**Tools and Libraries:**
Pandas, Matplotlib, Seaborn, and WordCloud.

### 3. Data Preprocessing

**Steps Followed:**

- **Cleaning Text Data:** Applied text cleaning by converting text to lowercase, removing special characters, numbers, and extra spaces using the re library.  
- **Stopword Removal:** Removed common words like "and", "but", "not" using NLTK's stopwords. Certain words like "not" were retained to preserve sentiment meaning.
- **Tokenization:** Split the text into words (optional for more control).
- **Lemmatization:** Applied WordNetLemmatizer from NLTK to reduce words to their base form.

**Tools and Libraries Used:** re, nltk

**Methods Followed:** Tokenization, stopword removal, and lemmatization to normalize text.

### 4. Train/Test Split
We should split the data before vectorization to avoid "data leakage." If we perform vectorization before splitting, the vectorizer may learn information from the entire dataset, including the test set, which can bias your results.

**Steps Followed:**
- Split the dataset into training and testing sets with an 80/20 ratio to ensure a balanced evaluation.
- Used train_test_split from sklearn.model_selection.

**Tools and Libraries Used:** sklearn.model_selection.train_test_split

**Methods Followed:**
Ensured the target variable distribution in both splits remained consistent using stratify for class labels. 

### 5. Feature Engineering and Feature Selection
**Steps Followed:**

- **Label Encoding:** Label encoded the target variable that contains, angriness, happiness, sadness.

- **TF-IDF Vectorization:** Converted the cleaned text into numerical form using the TfidfVectorizer from sklearn. Configured parameters such as max_features and ngram_range for optimal feature extraction.
 
    **About TF-IDF Vectorizer:**
        TF-IDF Vectorization to convert text data into numerical feature vectors, which can be used as input for machine learning models. The TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency) is a popular technique in Natural Language Processing (NLP) used to convert text into numerical features based on the importance of words in a document relative to a collection of documents (corpus).

    **When to Use TF-IDF:** TF-IDF is most effective when you're working on text classification tasks, such as sentiment analysis, spam detection, and topic classification. It is also commonly used in information retrieval systems, where the goal is to retrieve the most relevant documents based on query terms.

    **Significance of TF-IDF in NLP:**
  
     - **Capturing Importance:** TF-IDF focuses on identifying words that are meaningful in the context of a specific document while down-weighting words that appear frequently across many documents (e.g., "the", "and", etc.). This makes it ideal for tasks where identifying the most significant words is crucial, such as in text classification or information retrieval.
       
     - **Improving Model Performance:** By using TF-IDF, models can learn better from words that truly represent the content of a document, leading to improved performance in text-based tasks. Words with higher TF-IDF scores are considered more relevant to the meaning of the document, making them more impactful for machine learning models.
  
     - **Efficient Representation:** TF-IDF provides a sparse, efficient vector representation of text, making it easier to apply machine learning algorithms.
 
- **Feature Selection:** Experimented with different numbers of features (e.g., top 5000, 10000 words) and selected the set that provided the best model performance.

**Tools and Libraries Used:** sklearn.feature_extraction.text.TfidfVectorizer

**Methods Followed:** Analyzed feature importance based on TF-IDF scores.

### 6. Metrics for Model Evaluation
**Metrics Chosen:**
- **Accuracy:** Percentage of correctly predicted instances.
- **Precision:** Ability to correctly identify positive cases.
- **Recall:** Ability to capture all positive cases.
- **F1 Score:** Balances precision and recall.
- **Confusion Matrix:** Visualized true vs. predicted results.

**Tools and Libraries Used:**
sklearn.metrics

**Methods Followed:**
Used multiple metrics to evaluate both general and class-specific performance.

### 7. Model Selection, Training, Predicting, and Assessment
**Steps Followed:**
- **Initial Models:** For text classification projects, common options include:
      1. Logistic Regression
      2. Support Vector Machines (SVM)
      3. Naive Bayes
      4. K Neighbors Classifier
      5. Decision Tree Classifier
      6. Random Forest Classifier
      7. LSTM (Long Short Term Memory) - Deep Learning Model
      8. CNN (Convolutional Nueral Network) - Deep Learning Model

**Final Model:** The best performance was achieved using the Support Vector Classifier (SVC).

**Training:** Trained models using the training set.

**Prediction:** Generated predictions on the test set.

**Assessment:** Compared model metrics to finalize the best-performing model.

**Tools and Libraries Used:**
sklearn.svm, sklearn.ensemble, sklearn.linear_model

**Methods Followed:**
Iterative testing of models with varying hyperparameters.

### 8. Hyperparameter Tuning/Model Improvement
**Steps Followed:**
- Performed Grid Search and Random Search for hyperparameter tuning on all the models including deep learning.
- Performed data augmentation to increase dataset size and improve model generalization.

**Tools and Libraries Used:**
sklearn.model_selection.GridSearchCV

**Methods Followed:**
Grid search with cross-validation to determine the optimal model configuration.

### 9. Model Deployment Plan
**Steps Followed:**
- **Model Export:** Saved the trained model and vectorizer using joblib.
- **Streamlit Integration:** Built a Streamlit application for an interactive UI where users can input text and get predictions.
- **Hosting:** Hosted the Streamlit app online (e.g., Streamlit Cloud).

**Tools and Libraries Used:**
joblib, streamlit

**Methods Followed:**
Used modular code to integrate preprocessing, feature extraction, and prediction in the app.

**Model Deployment Steps:**
- Trained and saved the model locally.
- Developed a Streamlit app to load the model and vectorizer.
- Hosted the app and provided a shareable link for user access.

### Results:
![image](https://github.com/user-attachments/assets/64a3bbbb-714e-4305-a332-4dc5f7bcaba4)

#### Interpreting the Results

From the above scorecard, we have compared the performances of various models across key metrics: **Accuracy, Precision, Recall, and F1 Score**. The summary of the observations are:

1. **Support Vector Classifier (SVC) Models**:
   - **Best Tuned SVC** stands out as the top-performing model, achieving an **Accuracy, Precision, Recall, and F1 Score of 96.56%** across all metrics. This highlights that SVC, especially after hyperparameter tuning, is highly effective for this classification task.
   - The initial **SVC model** also performed strongly, with metrics at 93.29%, showcasing its baseline strength.

2. **K-Nearest Neighbors (KNN) Models**:
   - The **Best Tuned KNN** model performed impressively with an accuracy of **95.83%**, making it the second-best model overall.
   - The untuned **KNN model** also did well, achieving **94.15% accuracy**, but hyperparameter tuning significantly boosted its performance.

3. **Random Forest Classifier (RFC) Models**:
   - The **Best Tuned Random Forest Classifier** delivered an accuracy of **95.62%**, showing consistent and reliable performance across metrics.
   - The initial **Random Forest Classifier** performed equally, with an accuracy of **95.62%**, but still maintained competitive results.

4. **Deep Learning Models**:
   - The **LSTM** model achieved strong results, with:
     - Initial LSTM: **95.38% accuracy**
     - Tuned LSTM: Slightly lower performance at **94.76%**, possibly due to overfitting during tuning.
   - The **CNN models** had similar performance:
     - Initial CNN: **95.09% accuracy**
     - Tuned CNN: Equally as, **95.09%**, showcasing consistent performance.

5. **Logistic Regression (LR) Models**:
   - The **Best Tuned Logistic Regression** achieved an accuracy of **94.24%**, proving that LR is highly effective for simpler classification tasks.
   - The initial **Logistic Regression** model performed slightly lower, with an accuracy of **91.90%**.

6. **Decision Tree Classifier (DTC) Models**:
   - Both **tuned** and **initial Decision Tree Classifiers** achieved similar accuracy at **91.82%**, indicating limited improvement from tuning.

7. **Naive Bayes (NB) Models**:
   - Naive Bayes models performed the lowest among all, with:
     - Tuned NB: **87.16% accuracy**
     - Initial NB: Slightly lower at **86.35%**.

#### **Key Takeaways**:

1. **Top Performers**:
   - The **Best Tuned SVC** model outperformed all others, demonstrating its robustness and suitability for this classification problem.
   - **KNN (Best Tuned)** and **Random Forest (Best Tuned)** also showed excellent results, making them strong alternatives to SVC.

2. **Deep Learning Models**:
   - While **LSTM** and **CNN** models performed well (above 94% accuracy), they didn't outperform traditional machine learning models like SVC or tuned KNN. This suggests that simpler models are better suited for this specific dataset.

3. **Logistic Regression**:
   - Logistic Regression performed exceptionally well post-tuning, with competitive accuracy and F1 scores, proving its reliability for text classification tasks.

4. **Ensemble Models**:
   - Random Forest demonstrated strong performance, although it was slightly behind the top-performing models like SVC and tuned KNN.

5. **Naive Bayes**:
   - Naive Bayes showed relatively weaker performance, indicating it might not be the best fit for this dataset's complexity.

#### **Conclusion**:
1. **Best Model**: The **Best Tuned Support Vector Classifier** is the most effective model, achieving the highest performance across all metrics.
2. **Competitive Models**: **Tuned KNN** and **Random Forest Classifier** provide strong alternatives with excellent accuracy and reliability.
3. **Deep Learning**: While **LSTM** and **CNN** models performed decently, they didn't surpass traditional ML models for this specific dataset.
4. **Simple Models**: Logistic Regression, especially after tuning, proves to be a simple yet competitive solution.
5. **Underperformers**: Naive Bayes and Decision Tree models lagged behind, suggesting they are less suitable for this task.

Finally, **Support Vector Classifier** is the best choice for this task, with **tuned KNN** and **Random Forest** being excellent backups.

### Instructions to Run This Project:

**Jupyter Notebook Steps**

1. **Download the Project:**
- Download this project repository as a zip file.
- Unzip the folder to your desired location on your system.

2. **Set Up Anaconda:**
- Go to the Anaconda website and download the installer for your operating system (Windows, macOS, or Linux).
- Install Anaconda following the instructions provided during the installation.

3. **Launch Jupyter Notebook:**
- Open Anaconda Navigator from your installed applications.
- Launch the Jupyter Notebook interface from Anaconda Navigator.
- A browser window will open displaying the Jupyter Notebook dashboard.

4. **Navigate to the Project Folder:**
- Use the file explorer in Jupyter Notebook to navigate to the folder where you unzipped the project.
- Navigate to: intensityanalysis > notebooks > intensityanalysis.ipynb.

5. **Adjust File Paths:**
- Open the file intensityanalysis.ipynb.
- Update any file paths in the notebook to point to the directories inside your unzipped project folder (e.g., paths for data, models, and visuals).

6. **Run the Notebook:**
- Click on "Run All" or execute each cell sequentially to process the data, train models, and generate outputs.
- Outputs such as processed data, trained models, and generated visuals will be saved in the corresponding directories (data, models, visuals).

### Instructions to Use the Text Intensity Analyzer App

- **Access the Application:**
Click on the link to the hosted Streamlit app - https://capstoneprojectintensityanalysisapp-upgrad.streamlit.app/.
The app will open in your browser.

- **Enter the Text:**
On the homepage of the app, you’ll see a text box labeled "Enter your text below".
Type or paste the text snippet you want to analyze into this box.
Example: "I am extremely happy and excited about the future!"

- **Analyze Intensity:**
Click on the "Analyze Intensity" button below the text box.
Wait for a few seconds as the app processes your input.

- **View Results:**
The app will display the predicted emotional intensity of your input text.
The possible outputs are: Happiness, Angriness, Sadness

- **Analyze Another Text:**
Clear the text box, enter a new text snippet, and click "Analyze Intensity" again to repeat the process.

- **Additional Features (if implemented):**
If your app includes additional sections like visualizations, explanations, or download options, navigate and explore them as per the on-screen instructions.

**Troubleshooting Tips:**
- Ensure you have a stable internet connection when accessing the app.
- If the app doesn't load or shows an error, refresh the page or try again later.
