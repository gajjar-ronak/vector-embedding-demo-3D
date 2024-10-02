# FAISS Nearest Neighbor Search with 3D Visualization

This project demonstrates how to use **FAISS** for efficient nearest neighbor search in 3D vector space. The code compares two different techniques for nearest neighbor search: **brute-force** and **advanced indexing (IVF)**, and visualizes the results in 3D. 

We simulate a real-world use case by creating a set of **random text embeddings** as 3D vectors and search for similar vectors given a query. The project uses FAISS, a library designed for efficient similarity search, and visualizes the vector data and search results using **Matplotlib**.

---

### Table of Contents
- [Project Description](###project-description)
- [One-Line Explanation](###one-line-explanation)
- [Technologies Used](###technologies-used)
- [Installation](###installation)
- [Usage](###usage)
- [Use Cases](###use-cases)
- [Methods Used](###methods-used)
- [License](###license)
- [Acknowledgements](###Acknowledgements)

---

### Project Description

This project demonstrates **nearest neighbor search** using **FAISS**, a library for efficient similarity search. The example is based on 3D vector embeddings, simulating a real-world dataset of text embeddings.

- **Dataset**: A simulated collection of 100 sentences representing typical text data.
- **Query**: A random vector is used to query the nearest neighbors in the dataset.
- **Indexing**: It compares **brute-force search** (simple L2 distance calculation) and **IVF indexing** (clustered indexing technique) to find the nearest neighbors.
- **Visualization**: The search results are visualized in 3D to better understand vector relationships.

---

### One-Line Explanation

This code demonstrates nearest neighbor search with **FAISS** using 3D vector embeddings, comparing brute-force and IVF indexing techniques to optimize search performance.

---

### Technologies Used

```json
["FAISS", "NumPy", "Matplotlib", "Python", "Seaborn"]
```

### Installation

To run this project, you'll need to install the necessary Python libraries. You can install the dependencies using the following command:

```bash
pip install faiss-cpu numpy matplotlib seaborn
```

---

### Usage

**Clone the repository**:
   
   ```bash
   git clone https://github.com/gajjar-ronak/vector-embedding-demo-3D
   cd faiss-nearest-neighbor-search
   python faiss_nearest_neighbor.py
```

### Use Cases

- **Text Search**: Find semantically similar sentences or documents based on vector embeddings using FAISS for large-scale text search.
- **Recommendation Systems**: Use nearest neighbor search for recommending similar products, movies, or content based on user preferences.
- **Clustering and Classification**: Efficiently classify or cluster similar data points by using advanced indexing techniques like IVF in high-dimensional vector spaces.

### Methods Used

- **Brute-force Search (IndexFlatL2)**: Simple nearest neighbor search using **Euclidean distance** for small datasets.
- **IVF Indexing (IndexIVFFlat)**: Advanced indexing technique for efficient nearest neighbor search by clustering data points into partitions, allowing faster searches on large datasets.
- **3D Visualization**: Visualize vector data and query results in a **3D space** to intuitively understand the vector relationships and nearest neighbor search.

### Example

You can visualize the search results in 3D, where the following will be displayed:
- **Blue points**: Data vectors.
- **Red point**: The query vector.
- **Green points**: The nearest neighbors to the query vector.

Here’s a sample 3D visualization result:

- **Brute-force search** will show the nearest neighbors calculated by simple distance measures.
- **IVF search** will show the optimized nearest neighbors, with faster performance on larger datasets.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

- **FAISS** for providing a powerful and efficient library for similarity search.
- **Matplotlib** and **Seaborn** for the beautiful data visualizations.
