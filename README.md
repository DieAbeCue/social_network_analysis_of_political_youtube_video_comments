# social_network_analysis_of_political_youtube_video_comments
  This project aims at classifying the political bias of youtube comments and verify if they're substantially in line with the bias of the video source to check for potential existence of echo chambers. The classification is based on the site [AllSides](https://www.allsides.com/media-bias/ratings?field_featured_bias_rating_value=1&field_news_source_type_tid%5B%5D=2&field_news_bias_nid_1%5B1%5D=1&field_news_bias_nid_1%5B2%5D=2&field_news_bias_nid_1%5B3%5D=3&title=) and the bias that it assigns to the parent organization.
  
![AllBiasChart](https://www.allsides.com/sites/default/files/allsides-media-bias-chart-v9.1.png){ width=50% height=50% }

   Firstly, a dataset of classified articles found online was loaded to help train the model. The articles, the transcript of the videos and the studied comments were embedded using Doc2Vec along with TF-IDF to assign a weight to the words. Cosine similarity between the comments and the transcript of their respective video was used to discriminate the comments that were irrelevant to the study. Then, three different models, SVC, Random Forest Classifier and Logistic regression were trained (on a dataset that included the transcripts, the articles and hundreds of comments manually labeled), optimized and tested to see which would be capable of classifying text with the most accuracy. The chosen model was then used to classify the all the comments. Finally, a logistic regression model was employed to investigate the statistical significance of both the video's bias and the topic in relation to the rate of agreement. Whenever a video's topic and/or bias was found to be statistically significant, the coefficient was used to elaborate on which bias and/or topic was more likely to have a higher rate of agreement.
   
   To demonstrate the classification model, it was deployed on a simplistic local html template where, by inputting one or multiple opinions on one of the three political topics (abortion, affirmative action and gun control), as well as the bias and topic of the video that the opinion came from (if the input opinion is a comment found under a political YouTube video), the model will return the predicted bias of the opinion (left, center or right).

   The technologies used for this project include:  Python, Pandas, Google’s API, Azure translator, TF-IDF Vectorizer, Scikit-Learn, Doc2Vec, Json, Matplotlib, Stats models, Seaborn, Flask, html.
   
  The repository contains the following folders and files:
     - Diego_Cuevas_Capstone_Project.ipynb: main file for the project
     - Article_Bias_Prediction: The dataset of classified articles used to help train the classification model. The document with the              extension .tar.gz is the compressed document as found and downloaded. From that compressed file was extracted a folder (named                Article_Bias_Prediction) containing the compressed .zip file. From that zip file was extracted the folder containing the proper json        files used in the project (named Article_Bias_Prediction-main).
     - comments.csv: the manually classified comments used to train the model
     - augmented.csv: the back translated comments from comments.csv used to train the model
     - Note_20230821_1234_otter_ai.txt: transcript of one of the studied videos that didn't have captions
     - Bias_model_deployment.py: main code file for the deployment
     - templates: the folder containing the html file where the model was deployed
     - columns_for_model.csv: a reference dataframe used during the deployment of the model
     - bias_model.pkl: the file of the saved model
     - my_functions.py: a file containing string cleaning functions called during the deployment of the model
