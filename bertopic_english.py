from bertopic import BERTopic, representation

print(representation)

############
# LOAD DATA
############
import pandas as pd

def read_csv():
    df = pd.read_csv("hf://datasets/okite97/news-data/train.csv")
    documents = df['Excerpt'].tolist()
    return documents[:200] # limit to first 200 documents

documents = read_csv()


#############################
# STEP 1: Embedding Model [https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html]
#############################

# SentenceTransformer [https://www.sbert.net/docs/sentence_transformer/pretrained_models.html]
#from sentence_transformers import SentenceTransformer
#sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
sentence_model = "all-MiniLM-L6-v2"

# HuggingFace Transformers [https://huggingface.co/models]
#from transformers.pipelines import pipeline
#sentence_model = pipeline("feature-extraction", model="distilbert-base-cased")

# Spacy Transformer https://github.com/explosion/spaCy
# py -m spacy download en_core_web_md
#import spacy
#spacy.prefer_gpu()
#sentence_model = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

# Gensim Models e.g. Glove, Word2Vec, FastText
#import gensim.downloader as api
#sentence_model = api.load('fasttext-wiki-news-subwords-300')


###################################
# STEP 2: Dimensionality Reduction [https://maartengr.github.io/BERTopic/getting_started/dim_reduction/dim_reduction.html]
###################################

# UMAP [https://umap-learn.readthedocs.io/en/latest/parameters.html]
# maybe use cuml UMAP for GPU Acceleration [https://maartengr.github.io/BERTopic/getting_started/dim_reduction/dim_reduction.html#cuml-umap]
from umap import UMAP
dr_model = umap_model = UMAP(
    n_neighbors=10,  # Number of neighboring points used in local approximations
    n_components=2,  # Number of dimensions to reduce the data to
    metric='cosine',  # Distance metric to use for computing distances in high-dimensional space
    min_dist=0.0,  # Minimum distance between points in the low-dimensional space
    random_state=42  # Seed for reproducibility
)

# PCA [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html]
# Note from the author: PCA and k-Means have worked quite well in my experiments
#from sklearn.decomposition import PCA
#dr_model = PCA(
#    n_components=5,  # Number of principal components to keep
    #whiten=False,  # If True, the components are decorrelated and normalized to unit variance
    #svd_solver='auto',  # Solver selection based on input data and n_components
    #tol=0.0,  # Tolerance for singular values (used by 'arpack' solver)
    #iterated_power='auto',  # Number of iterations for the power method (used by 'randomized' solver)
    #n_oversamples=10,  # Additional number of random vectors for 'randomized' solver to ensure proper conditioning
    #power_iteration_normalizer='auto',  # Normalizer for the power iteration method (used by 'randomized' solver)
    #random_state=None  # Seed for reproducible results (used by 'arpack' or 'randomized' solver)
#)

# TruncatedSVD [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html]
#from sklearn.decomposition import TruncatedSVD
#dr_model = TruncatedSVD(
#    n_components=5,  # Number of dimensions to reduce the data to
    #algorithm='randomized',  # Algorithm to use for the SVD computation
    #n_iter=5,  # Number of iterations for the randomized SVD solver
    #n_oversamples=10,  # Number of oversamples for the randomized SVD solver
    #power_iteration_normalizer='auto',  # Normalizer for the power iteration in the randomized SVD solver
    #random_state=None,  # Seed for reproducibility
    #tol=0.0  # Tolerance for convergence of the 'arpack' solver
#)

# Skip Dimensionality Reduction entirely (Cluster right away from embeddings)
#from bertopic.dimensionality import BaseDimensionalityReduction
#dr_model = BaseDimensionalityReduction()

############################
# STEP 3: Cluster Algorithm [https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html]
############################

# HDBSCAN [https://hdbscan.readthedocs.io/en/latest/parameter_selection.html]
import hdbscan
clustering_model = hdbscan.HDBSCAN(
    min_cluster_size=4,  # Minimum size of clusters
    metric='euclidean',  # Distance metric to use for clustering
    #cluster_selection_epsilon=0.0,  # Distance threshold for clustering
    #alpha=1.0,  # Balance between point density and distance
    cluster_selection_method='eom',  # Method for selecting clusters
    prediction_data=True  # Whether to store data for cluster membership prediction
)

# K-Means
# It allows you to select how many clusters you would like and forces every single point to be in a cluster.
#  Therefore, no outliers will be created. This also has disadvantages. When you force every single point in a cluster,
#  it will mean that the cluster is highly likely to contain noise which can hurt the topic representations.
#from sklearn.cluster import KMeans
#clustering_model = KMeans(
#    n_clusters=8,  # Number of clusters to form
#    init='k-means++',  # Method for initialization
    #n_init='auto',  # Number of times the algorithm will run with different centroid seeds
#    max_iter=300,  # Maximum number of iterations for a single run
    #tol=0.0001,  # Relative tolerance with regards to inertia to declare convergence
    #verbose=0,  # Verbosity mode
    #random_state=None,  # Seed for reproducibility
    #copy_x=True,  # Whether to copy the data or perform in-place computations
    #algorithm='lloyd'  # K-means algorithm to use
#)

# Like k-Means, there are a bunch more clustering algorithms in sklearn that you can be using.
#  Some of these models do not have a .predict() method but still can be used in BERTopic.
#  However, using BERTopic's .transform() function will then give errors. 


##########################
# STEP 4: Vectorizer [https://maartengr.github.io/BERTopic/getting_started/vectorizers/vectorizers.html]
##########################

# CountVectorizer [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html]
from sklearn.feature_extraction.text import CountVectorizer

#if you want to add custom/additional stopwords
#import nltk
#nltk.download('stopwords')
#english_stop_words = set(nltk.corpus.stopwords.words('english'))
#english_stop_words = list(set(list(english_stop_words) + [])) # you can add more stopwords here (in that list)

english_stop_words = "english"

vect_model = CountVectorizer(
    #max_df=1.0,  # Ignore terms with a document frequency higher than this threshold
    #min_df=1,  # Ignore terms with a document frequency lower than this threshold
    #max_features=None,  # Maximum number of features to keep (highest term frequency)
    stop_words=english_stop_words,  # Remove stop words
    #ngram_range=(1, 1),  # Lower and upper boundary of the range of n-values for different n-grams
    #tokenizer= None # Interesting for non-western languages e.g. chinese, see: [https://maartengr.github.io/BERTopic/getting_started/vectorizers/vectorizers.html#tokenizer]
)

# OnlineCountVectorizer [https://maartengr.github.io/BERTopic/getting_started/vectorizers/vectorizers.html#onlinecountvectorizer]
#from bertopic.vectorizers import OnlineCountVectorizer
#vect_model = OnlineCountVectorizer()

###############################
# STEP 5: Topic Representation [https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html]
###############################

# c - TF - IDF
from bertopic.vectorizers import ClassTfidfTransformer
tr_model = ClassTfidfTransformer(
    #bm25_weighting= True,
    #reduce_frequent_words= True
)

###############################
# STEP 6: Finetuning [https://maartengr.github.io/BERTopic/getting_started/representation/representation.html]
###############################

# KeyBERTInspired
from bertopic.representation import KeyBERTInspired
representation_model = KeyBERTInspired()

# POS
#from bertopic.representation import PartOfSpeech
#pos_patterns = [
#            [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
#            [{'POS': 'NOUN'}], [{'POS': 'ADJ'}]
#]
#representation_model = PartOfSpeech("en_core_web_md", pos_patterns=pos_patterns) # py -m spacy download en_core_web_md

# Maximal Marginal Relevance
#from bertopic.representation import MaximalMarginalRelevance
#representation_model = MaximalMarginalRelevance(diversity=0.3)

# Zer-Shot Classification
#from bertopic.representation import ZeroShotClassification
#candidate_topics = ["space and nasa", "bicycles", "sports"]
#representation_model = ZeroShotClassification(candidate_topics, model="facebook/bart-large-mnli")

# There are also approaches with LLM's as described in here: [https://maartengr.github.io/BERTopic/getting_started/representation/llm.html]

###########################
# APPLY, FIT AND TRANSFORM
###########################

# Create BERTopic model
topic_model = BERTopic(
    # Step 1: Embedding
    #language="english", # if this is set, embedding_model will be ignored
    embedding_model=sentence_model,
    # Step 2: Dimensionality Reduction
    umap_model=dr_model,
    # Step 3: Clustering
    hdbscan_model=clustering_model,
    # Step 4: Tokenize Topics
    vectorizer_model=vect_model, # can also be passed afterwards, see: [https://maartengr.github.io/BERTopic/getting_started/vectorizers/vectorizers.html#countvectorizer]
    # this allows you to tweak the topic representations without re-training your model
    # Step 5: Extract topic words
    ctfidf_model=tr_model, 
    # Step 6: (Optional) Fine-tune topic representations from Step 5
    representation_model=representation_model,

    calculate_probabilities=True,  # Whether to calculate topic probabilities for each document.
    #top_n_words=10,  # Number of top words to represent each topic.
    #n_gram_range=(1, 1),  # Range of n-grams to consider when forming topics.
    #min_topic_size=10,  # Minimum size of a topic; smaller clusters will be merged.
    nr_topics=None,  # Number of topics to find; if None, it will be inferred from the data.
    #seed_topic_list=None,  # List of seed words or topics to guide the topic modeling.
    #zeroshot_topic_list=None,  # List of topics for zero-shot topic modeling.
    #zeroshot_min_similarity=0.7,  # Minimum similarity for assigning documents to zero-shot topics.
    verbose=True,  # If True, additional information will be printed during the process.
)

# Fit the model and transform the documents using the pre-computed embeddings
doc_topics, probs = topic_model.fit_transform(documents)
topics = topic_model.get_topics()

#Updating with vectorizer afterwards would look like this
#topic_model.update_topics(documents, vectorizer_model=vect_model)

############################
# VISUAlIZATIONS
############################

# Visualize the topics
#fig_topics = topic_model.visualize_topics()
#fig_topics.show()

# Visualize the distribution of topics for a given document (e.g., the first document)
#fig_distribution = topic_model.visualize_distribution(probs[0])
#fig_distribution.show()

# Visualize the hierarchical structure of topics
#fig_hierarchy = topic_model.visualize_hierarchy()
#fig_hierarchy.show()

# Visualize the similarity between topics as a heatmap
#fig_heatmap = topic_model.visualize_heatmap()
#fig_heatmap.show()

# Visualize the top words for each topic as a bar chart
#fig_barchart = topic_model.visualize_barchart()
#fig_barchart.show()

# Visualize the decline in term scores across topics
#fig_term_rank = topic_model.visualize_term_rank()
#fig_term_rank.show()

# Additional visualizations:

# Visualize the topics with reduced dimensionality (e.g., UMAP)
#fig_reduced = topic_model.visualize_documents(documents)
#fig_reduced.show()

# Wordcloud also possible (just for fun)
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt

#print("Word Clouds for Topics:")
#for topic_id, words in topics.items():
#    word_freq = {word: weight for word, weight in words}
#    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
#    
#    plt.figure(figsize=(10, 5))
#    plt.imshow(wordcloud, interpolation='bilinear')
#    plt.axis('off')
#    plt.title(f"Topic {topic_id}")
#    plt.show()

##########################
# TEXTUAL REPRESENTATIONS
##########################

# All topics with the top 5 words
#print("Topics:")
#for topic_id, words in topics.items():
#    print(f"Topic {topic_id}: {', '.join([word for word, _ in words[:5]])}")

# All topics with top 5 words with count
#print("Topics and Top Words Count:")
#for topic_id, words in topics.items():
#    word_counts = ', '.join([f"{word} ({count})" for word, count in words[:5]])
#    print(f"Topic {topic_id}: {word_counts}")

# All topics with weights for top 5 words
print("Topics with Weights:")
for topic_id, words in topics.items():
    word_weights = ', '.join([f"{word} ({weight:.2f})" for word, weight in words[:5]])
    print(f"Topic {topic_id}: {word_weights}")

# All documents with resp. Topic
print("\nDocument Topic Assignments:")
for i, topic in enumerate(doc_topics):
    topic_words = ', '.join([word for word, _ in topics[topic][:5]])
    print(f"Document {i}: Topic {topic} - {topic_words}")

# All documents with resp Topics and probability for other topics
#print("\nDocument Topic Assignments:")
#for i, (topic, prob_dist) in enumerate(zip(doc_topics, probs)):
#    topic_words = ', '.join([word for word, _ in topics[topic][:5]])
#    print(f"Document {i}: Topic {topic} - {topic_words}")
#    print(f"Probabilities: {', '.join([f'Topic {t}: {p:.4f}' for t, p in enumerate(prob_dist)])}\n")

