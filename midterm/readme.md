Web app is available at https://akuvshinova-ds.github.io/NNDL-2025/midterm/

Prompt (Manus AI):
Role: You are  an expert machine learning developer + web-developer.

Task: take csv files (superstore_test_unlabeled - with no target var, superstore_train and superstore_test_labeled - both with target var), take exactly what is coded in .ipynb. Transform this .ipynb into app.js, make sure it can be executed without bugs and errors on a web-page. Create index.html file to interact with these scripts on a website (stylish UI and design, but keep it minimalistic).

Overall workflow:
0. In index.html: Layout sections: Data Load, Exploratory Data Analysis, Preprocessing, Model definition, Training, Metrics of training (ROC-AUC), Prediction, Export, Ground-Truth Check. In the end, Include deployment note text: 'Create public GitHub repo, commit index.html/app.js, enable Pages (main/root), test URL.'
1. Data Load: Input 2 datasets (train and test_unlabeled) by a click of a button. Show data shapes. Preview first 10 rows of each dataset. On backend, concatenate training and unlabeled test data for exploratory analysis, adding the source column (train, test). In css, for table fitting the screen implement ```/* 1. Sets the boundaries for the scrollbar to appear */
.table-scroll {
overflow-x: auto; /* The critical property for horizontal scrolling */
width: 100%; /* Constrains the container to the page width */
margin-top: 10px;
margin-bottom: 10px;
}

/* 2. Allows the table to be as wide as its content */
table {
width: max-content; /* Allows the table content to flow outside the 100% container */
border-collapse: collapse;
margin-top: 15px;
}```
2. Exploratory Data Analysis: Show all graphs that are in EDA files.
3. Preprocessing: comment on drop_useless_features, remove_extreme_outliers,preprocessing transformers (StandardScaler, OneHotEncoder,SimpleImputer,using RandomOverSampler on the training data). 
4. Model in app.js: tf.sequential: Dense(256, 'relu'), BatchNormalization(), Dropout(0.1), Dense(128, 'relu'), BatchNormalization(), Dropout(0.1), Dense(2, 'softmax'). Compile optimizer=Adam(learningRate=0.002714707), loss='categoricalCrossentropy' (labelSmoothing=0.02), metrics=['categoricalAccuracy','AUC']. Print description of the model.
5. Training in app.js: 80/20 stratified split (it is done here: `X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`; train 50 epochs, batch 128; tfjs-vis fitCallbacks for live loss/accuracy plots; implement EarlyStopping on 'val_auc' (patience=5, restoreBestWeights=true) and a ReduceLROnPlateau-like scheduler (factor=0.5, patience=5, minLR=1e-5). 
6. Metrics in app.js: Compute ROC/AUC from val probs; plot ROC; slider (0-1) updates confusion matrix, Precision/Recall/F1 dynamically.
7. Predict & Export in app.js: Predict test.csv probs; apply threshold for Response; download submission.csv (Id, Response), probabilities.csv; model.save('downloads://superstore-tfjs').
8. Ground truth check: button "load labeled test data", load dataset "superstore_test_labeled" (id, Response). Merge with predicted data, calculate final metrics (as in kaggle competitions)


Output: Output two separate code files: first index.html (for HTML structure and UI), second app.js (for JavaScript logic). Include English comments in code. Make interactive with buttons. Handle errors (e.g., alerts for missing files/invalid data). Ensure reusable by commenting schema swap points.
