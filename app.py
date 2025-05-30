import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy

from numpy import sqrt, argmax
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import pickle

def show_explore_page():
    st.title("Explore Jobs Dataset")
    
    # Loading Dataset into DataFrame
    st.write("#### 1. About Dataset")
    df = pd.read_csv('fake_job_postings.csv')
    st.dataframe(df.head())

    rows = df.shape[0]
    cols = df.shape[1]

    st.write("This dataset has", rows, "rows and ",cols, "columns.")

    st.write("#### 2. Exploratory Data Analysis")

    st.write("##### 2.1 Missing Values")

    fig = sns.set(rc={'figure.figsize': (8, 5)})
    fig, ax = plt.subplots()
    plt.title("Heat Map for Missing Values")
    sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
    st.pyplot(fig)

    # Filling Na with Blank Spaces
    df.fillna('', inplace=True)

    st.write("##### 2.2 Comparing Number of Fraudlent and Non-Fraudlent Job Posting")

    fig = sns.set(rc={'figure.figsize': (10, 3)})
    fig, ax = plt.subplots()
    plt.title("Number of Fradulent Vs Non-Fraudlent Jobs")
    sns.countplot(y='fraudulent', data=df)
    st.pyplot(fig)

    not_fraudulent = df.groupby('fraudulent')['fraudulent'].count()[0]
    fraudulent = df.groupby('fraudulent')['fraudulent'].count()[1]

    st.write(not_fraudulent, "jobs are NOT Fraudulent and ", fraudulent, " jobs are Fraudulent.")

    st.write("##### 2.3 Experiencewise Count")

    exp = dict(df.required_experience.value_counts())
    del exp['']

    fig = sns.set(rc={'figure.figsize': (10, 5)})
    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")
    plt.bar(exp.keys(),exp.values())
    plt.title('No. of Jobs with Experience')
    plt.xlabel('Experience')
    plt.ylabel('No. of jobs')
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.write("##### 2.4 Countrywise Job Count")

    # First Spliting location Column to extract Country Code
    def split(location):
        l = location.split(',')
        return l[0]

    df['country'] = df.location.apply(split)

    countr = dict(df.country.value_counts()[:14])
    del countr['']

    fig = sns.set(rc={'figure.figsize': (10, 5)})
    fig, ax = plt.subplots()
    plt.title('Country-wise Job Posting')
    plt.bar(countr.keys(), countr.values())
    plt.ylabel('No. of jobs')
    plt.xlabel('Countries')
    st.pyplot(fig)

    st.write("##### 2.5 Education Job Count")

    edu = dict(df.required_education.value_counts()[:7])
    del edu['']

    fig = sns.set(rc={'figure.figsize': (10, 5)})
    fig, ax = plt.subplots()
    plt.title('Job Posting based on Education')
    plt.bar(edu.keys(), edu.values())
    plt.ylabel('No. of jobs')
    plt.xlabel('Education')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write("##### 2.6 Top 10 Titles of Jobs Posted which were NOT fraudulent")

    dic = dict(df[df.fraudulent==0].title.value_counts()[:10])
    dic_df = pd.DataFrame.from_dict(dic, orient ='index')
    dic_df.columns = ["Number of Jobs"]
    st.dataframe(dic_df)

    st.write("##### 2.7 Top 10 Titles of Jobs Posted which were fraudulent")

    dic = dict(df[df.fraudulent==1].title.value_counts()[:10])
    dic_df = pd.DataFrame.from_dict(dic, orient ='index')
    dic_df.columns = ["Number of Jobs"]
    st.dataframe(dic_df)

# Creating a Dataframe with word-vectors in TF-IDF form and Target values

def final_df(df, is_train, vectorizer, column):

    # TF-IDF form
    if is_train:
        x = vectorizer.fit_transform(df.loc[:,column])
    else:
        x = vectorizer.transform(df.loc[:,column])

    # TF-IDF form to Dataframe
    temp = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

    # Droping the text column
    df.drop(df.loc[:,column].name, axis = 1, inplace=True)

    # Returning TF-IDF form with target
    return pd.concat([temp, df], axis=1)


# Training the model with various combination and returns y_test and y_pred

def train_model(df, input, target, test_size, over_sample, vectorizer, model):

    X = df.drop(target, axis=1)
    y = df[target]
    print("Splitted Data into X and Y.")

    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print("Splitted Data into Train and Test.")
    
    # Training Preprocessing
    X_train = final_df(X_train, True, vectorizer, input)
    X_train.dropna(inplace=True)
    print("Vectorized Training Data.")

    if over_sample:
        sm = SMOTE(random_state = 2)
        X_train, Y_train = sm.fit_resample(X_train, Y_train.ravel())
        print("Oversampling Done for Training Data.")

    # Testing Preprocessing
    x_test = final_df(x_test, False, vectorizer, input)
    x_test.dropna(inplace=True)
    print("Vectorized Testing Data.")

    # fitting the model
    model = model.fit(X_train, Y_train)
    print("Model Fitted Successfully.")

    # calculating y_pred
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)

    return model, x_test, y_test, y_pred_prob

def evaluate(y_test, y_pred, y_pred_prob):
    roc_auc = round(roc_auc_score(y_test, y_pred_prob[:, 1]), 2)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1], pos_label=1)
    
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    
    # locate the index of the largest g-mean
    ix = argmax(gmeans)

    y_pred = (y_pred > thresholds[ix])

    accuracy = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**ROC-AUC Score** \t\t: {roc_auc*100} %")
        st.write('**Best Threshold** \t\t: %.3f' % (thresholds[ix]))
    with col2:
        st.write('**G-Mean** \t\t\t: %.3f' % (gmeans[ix]))
        st.write(f"**Model Accuracy** : {round(accuracy,2,)*100} %")

    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

def trainer(df, test_size, over_sample, vectorizer, model):
    model, x_test, y_test, y_pred_prob = train_model(
        df=df, 
        input='text', 
        target='fraudulent', 
        test_size=test_size,
        over_sample=over_sample, 
        vectorizer=vectorizer, 
        model=model)

    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)

    evaluate(y_test, y_pred, y_pred_prob)


nlp = spacy.load('en_core_web_sm')

# Text Preprocessing with varoius combination

def spacy_process(text):
  # Converts to lowercase
  text = text.strip().lower()

  # passing text to spacy's nlp object
  doc = nlp(text)
    
  # Lemmatization
  lemma_list = []
  for token in doc:
    lemma_list.append(token.lemma_)
  
  # Filter the stopword
  filtered_sentence =[] 
  for word in lemma_list:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
      filtered_sentence.append(word)
    
  # Remove punctuation
  punctuations="?:!.,;$\'-_"
  for word in filtered_sentence:
    if word in punctuations:
      filtered_sentence.remove(word)

  return " ".join(filtered_sentence)

# For Loading the Pickle File
def load_model():
    with open('pickle/notebook_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
#########################################################

def compare_model_page():

    button = False

    st.title("Model Page")

    df = pd.read_csv("clean_df.csv")

    st.write("#### 1. Vectorizer Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        gram = st.selectbox("**Select Grams**", ("Mono-Gram", "Bi-Gram", "Tri-Gram"))
        
        if gram == "Mono-Gram":
            gram = (1,1)
        elif gram == "Bi-Gram":
            gram = (2,2)
        elif gram == "Tri-Gram":
            gram = (3,3)

    with col2:
        no_features = st.slider('**Select Max-Features**', 1, 1000, 100)

    with col3:
        vec = st.selectbox("**Select Vectorizer**", ("Count", "TF-IDF"))

    if vec == "Count":
        vectorizer = CountVectorizer(ngram_range=gram, max_features = no_features)
    elif vec == "TF-IDF":
        vectorizer = TfidfVectorizer(ngram_range=gram, max_features = no_features)

    model = st.selectbox("**Select Model**", ("Logistic Regression","Random Forest","Support Vector Machine"))

    st.write("#### 2. Data Configuration")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider('**Select Test Size**', 10, 100, 30)
        test_size = test_size/100

    with col2:
        over_sample = st.selectbox('**Do Over-Sampling**', ['Yes', 'No'])
        if over_sample == 'Yes':
            over_sample = True
        elif over_sample == 'No':
            over_sample = False

    st.write("#### 3. Model Configuration")

    if model == "Logistic Regression":

        col1, col2 = st.columns(2)

        with col1:
            penalty = st.selectbox("**Select Penalty**", ("l1","l2","elasticnet"))
            random_state = st.slider('**Select Random State**', 1, 1000, 42)
        with col2:
            solver = st.selectbox("**Select Solver**", ("liblinear","newton-cg", "newton-cholesky", "sag", "saga"))
            n_jobs = st.slider('**Select N-Jobs**', 1, 1000, 42)

        model = LogisticRegression(
            penalty=penalty,
            solver=solver,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        train = st.button("Train")

        if train:
            st.write("#### 4. Model Evaluation")
            trainer(df, test_size, over_sample, vectorizer, model)
            button = st.button('Save Logistic Regression as Pickle')

    elif model == "Random Forest":
        col1, col2 = st.columns(2)

        with col1:
            criterion = st.selectbox("**Select Criterion**", ("gini","entropy","elasticnet"))
            n_estimators = st.slider('**Select N-Estimatorse**', 1, 1000, 100)
            n_jobs = st.slider('**Select N-Jobs**', 1, 1000, 10)
        with col2:
            max_features = st.selectbox("**Select Max-Features**", ("sqrt","log2"))
            max_depth = st.slider('**Select Max-Depth**', 1, 50, 10)
            random_state = st.slider('**Select Random-State**', 1, 1000, 42)

        model = RandomForestClassifier(
            criterion=criterion,
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )

        train = st.button("Train")

        if train:
            st.write("#### 4. Model Evaluation")
            trainer(df, test_size, over_sample, vectorizer, model)
            button = st.button('Save Random Forest as Pickle')

    elif model == "Support Vector Machine":
        col1, col2 = st.columns(2)

        with col1:
            kernel = st.selectbox("**Select Kernel**", ("linear","poly","rbf", "sigmoid"))

        with col2:
            random_state = st.slider('**Select Random State**', 1, 1000, 42)

        model = SVC(
            kernel=kernel,
            random_state=random_state,
            probability=True
        )

        train = st.button("Train")

        if train:
            st.write("#### 4. Model Evaluation")
            trainer(df, test_size, over_sample, vectorizer, model)
            button = st.button('Save Support Vector Machine as Pickle')

    if button:
        data = {"model": model}
        with open('pickle/app_model.pkl', 'wb') as file:
            pickle.dump(data, file)

 #######################################################           

def show_predict_page():

    st.title("Predict If Job is Real or Fake")

    text = st.text_area('**Enter Job Description**')

    ok = st.button("Predict")

    if ok:

        st.write("**Input Text**")
        st.write(text)

        text = spacy_process(text)
        st.write("**After Text-Preprocessing**")
        st.write(text)

        data = {
            'text': [text]
        }

        df = pd.DataFrame(data)

        data = load_model()
        model = data["model"]
        vectorizer = data["vectorizer"]

        x = vectorizer.transform(df.loc[:,'text'])
        temp = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

        prediction = model.predict(temp)

        if prediction[0] == 1:
            st.markdown(
            body="""
            <div style='
                background-color:#ffe6e6;
                border: 2px solid red;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                text-align: center;
            '>
                <span style='color:red; font-size: 30px; font-weight:bold;'>
                    😟 <strong>Job is Fake!</strong>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        elif prediction[0] == 0:
            st.markdown(
                body="""
                <div style='
                    background-color:#e6ffe6;
                    border: 2px solid green;
                    border-radius: 10px;
                    padding: 20px;
                    margin-top: 20px;
                    text-align: center;
                '>
                    <span style='color:green; font-size: 30px; font-weight:bold;'>
                        😊 <strong>Job is Real!</strong>
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    




page = st.sidebar.selectbox("Explore Or Predict Or Else", ("Understanding the Data","Predict"))

if page == "Understanding the Data":
    show_explore_page()
elif page == "Predict":
    show_predict_page()