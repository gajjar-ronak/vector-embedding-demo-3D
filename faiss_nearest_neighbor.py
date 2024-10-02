import numpy as np
import faiss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time

# Set up seaborn style for better plots
sns.set(style="whitegrid")

# ============================
# Real-World Example: Text Embeddings
# ============================

# Example sentences (our "real-world" data)
# Example sentences (simulating a real-world dataset with 100 sentences)
sentences = [
    "I love programming in Python.", "The weather is great today.", "Python is great for data analysis.",
    "The sky is blue and the sun is shining.", "Data science is fascinating.", "I enjoy learning new programming languages.",
    "It's sunny outside, perfect for a walk.", "Python is widely used in machine learning.", "Let's go for a hike this weekend.",
    "I love exploring new technologies.", "Deep learning is a subset of machine learning.", "Data is the new oil.",
    "I am working on a machine learning project.", "The future of AI is exciting.", "Reinforcement learning is a fascinating topic.",
    "I am learning how to build neural networks.", "The cloud is revolutionizing computing.", "Big data analysis is key for business insights.",
    "Python makes it easy to implement machine learning algorithms.", "Artificial intelligence will change the world.", "I am analyzing some stock market data.",
    "Neural networks are inspired by the human brain.", "A convolutional neural network is great for image processing.", "Natural language processing is an important field in AI.",
    "I am exploring the field of generative models.", "Data visualization tools help communicate insights effectively.", "Automated systems can reduce human error.",
    "I am studying how to optimize machine learning models.", "The concept of overfitting is crucial in machine learning.", "AI ethics is an important area of research.",
    "Recurrent neural networks are useful for sequential data.", "Machine learning can be used to predict future trends.", "Data cleaning is essential for accurate analysis.",
    "I am working on a time-series forecasting problem.", "The importance of feature selection cannot be underestimated.", "Understanding the bias-variance tradeoff is critical.",
    "Linear regression is a simple yet powerful algorithm.", "I am learning how to tune hyperparameters for my models.", "The rise of edge computing will change data processing.",
    "Data science is an interdisciplinary field.", "I am researching AI-based healthcare solutions.", "The importance of data privacy is growing.",
    "I am using Python’s scikit-learn library for machine learning.", "Big data frameworks like Hadoop and Spark are popular.", "I enjoy reading papers on the latest advancements in AI.",
    "Neural networks are at the core of deep learning.", "The role of data in decision-making is growing.", "I am building a recommendation system.",
    "Clustering algorithms are useful for unsupervised learning.", "Data preprocessing is an essential step in machine learning.", "Decision trees are easy to interpret models.",
    "I am learning about the different types of neural networks.", "Gradient descent is a key optimization algorithm.", "I am exploring unsupervised learning techniques.",
    "Natural language processing has a wide range of applications.", "AI is transforming industries like healthcare and finance.", "I am learning how to deploy machine learning models.",
    "The future of autonomous vehicles relies on AI.", "Data lakes help store large volumes of data.", "AI in healthcare is revolutionizing diagnosis and treatment.",
    "I am studying the application of AI in robotics.", "The concept of transfer learning is gaining popularity.", "Supervised learning is a common approach in machine learning.",
    "I am using machine learning to detect fraudulent activities.", "AI models need to be regularly updated with new data.", "Computer vision is an exciting area in AI research.",
    "Data privacy laws like GDPR are important for businesses.", "The demand for data scientists is growing rapidly.", "I am studying the ethical implications of AI."
] * 2

# For this example, we simulate the text embeddings
# Normally you would use a model like BERT, GPT, or any sentence embedding model
# For simplicity, we'll generate random embeddings as placeholders
dimension = 3  # 3D vectors for visualization
num_sentences = len(sentences)

# Generate random embeddings for each sentence (this simulates the embeddings)
np.random.seed(42)
data_vectors = np.random.rand(num_sentences, dimension).astype('float32')

# ============================
# Generate Query Vector (Random)
# ============================

# Simulate a random query vector, which could be a sentence embedding
query_vector = np.random.rand(1, dimension).astype('float32')

# ============================
# Plotting Function for 3D
# ============================

def plot_3d_vectors(vectors, query, indices=None, title="3D Vector Plot"):
    """
    Plot the data vectors in 3D space and highlight the query vector and nearest neighbors.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data vectors
    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], color='blue', s=50, label="Data Vectors")

    # Plot the query vector
    ax.scatter(query[0, 0], query[0, 1], query[0, 2], color='red', s=100, label="Query Vector", marker='x')

    if indices is not None:
        # Highlight nearest neighbors (indices)
        nearest_neighbors = vectors[indices[0]]
        ax.scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], nearest_neighbors[:, 2], color='green', s=100, label="Nearest Neighbors")

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(title)
    ax.legend()
    plt.show()

# Plot the data vectors in 3D
plot_3d_vectors(data_vectors, query_vector, title="3D Data Vectors")

# ============================
# Brute-force Indexing (IndexFlatL2)
# ============================

# Create a simple FAISS index (brute-force search)
index = faiss.IndexFlatL2(dimension)
index.add(data_vectors)  # Add vectors to the index

# Perform search for the top 3 nearest neighbors
k = 3
start_time = time.time()
distances, indices = index.search(query_vector, k)
end_time = time.time()

print(f"Brute-force search time: {end_time - start_time:.4f} seconds")

# Plot the search results for brute-force indexing
plot_3d_vectors(data_vectors, query_vector, indices, title="Brute-Force Search - Nearest Neighbors")

# ============================
# Advanced Indexing (IVF)
# ============================

# Create an IVF index for more efficient search
nlist = 3  # Number of clusters
quantizer = faiss.IndexFlatL2(dimension)  # Quantizer for clustering
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

# Train the IVF index (this step is needed for IVF)
index_ivf.train(data_vectors)

# Add vectors to the IVF index
index_ivf.add(data_vectors)

# Perform search for the top 3 nearest neighbors using IVF
start_time = time.time()
distances_ivf, indices_ivf = index_ivf.search(query_vector, k)
end_time = time.time()

print(f"IVF search time: {end_time - start_time:.4f} seconds")

# Plot the search results for IVF indexing
plot_3d_vectors(data_vectors, query_vector, indices_ivf, title="IVF Search - Nearest Neighbors")