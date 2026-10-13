# Clothes Recommender System

A multimodal, intelligent clothing recommendation system using text, images, and user behavior — now with **S3 integration**, **Lambda-style triggers via LocalStack**, and a vision for collaborative recommendations powered by **Pinecone**.

---

## 🧠 Overview

Clothy is a smart clothes recommender that provides:

- 🧾 **Search by text** – Type something like _"black bra"_ and retrieve visually and semantically similar products.
- 🖼️ **Search by image** – Upload an image and find similar clothing items based on visual features.
- 👤 **User-based recommendations** – Generate recommendations based on historical purchases and inferred preferences.
- 💬 **AI Chatbot** – Use natural language to search, ask questions, or explore products with metadata-aware filters.
- ☁️ **S3-based pipeline triggering** – Automatically rebuild embeddings/vector store when new data lands in your S3 bucket (via LocalStack emulation).
- 🔄 **Collaborative recommendations (upcoming)** – Leverages both content-based and collaborative signals by fusing embeddings from user activity and interests.

---

## ⚙️ Architecture & Components

- **Streamlit UI** – Clean, tabbed interface for searching, browsing, and chatting.
- **FastAPI Backend** – REST API to power integrations, upload CSV/JSON, serve user data, etc.
- **Vector store** – Currently **ChromaDB**, migrating to **Pinecone** for better scaling and real-time queries.
- **Image/text embedding** – Uses **OpenCLIP** for visual and textual understanding.
- **LangChain + OpenAI** – Metadata filtering and chatbot functionality.
- **LocalStack** – Emulates **AWS S3 + Lambda** locally to trigger data processing pipelines.

---

## 🚀 Features

- **Text-to-Image Search:** Convert free-form text queries into OpenCLIP embeddings and match against product images.
- **Image-Based Search:** Upload a photo and retrieve the most visually similar clothing items.
- **User Recommendations:**
  - Personalized results based on historical purchases.
  - Gender-aware grouping (e.g., "Men", "Women").
- **Chatbot Search:**
  - Uses LangChain’s SelfQueryRetriever to interpret structured filters from natural queries.
  - Offers interactive and visual responses.
- **S3 Integration with Triggers:**
  - Uploading a data file to S3 (LocalStack) triggers a Lambda-like function to rebuild your vector index automatically.
- **Collaborative Filtering (Planned):**
  - Combines user behavior and interest profiles into a single embedding to find similar users/products.

---

## 🚀 Features

- **Text-to-Image Search:** Convert free-form text queries into OpenCLIP embeddings and match against product images.
- **Image-Based Search:** Upload a photo and retrieve the most visually similar clothing items.
- **User Recommendations:**
  - Personalized results based on historical purchases.
  - Gender-aware grouping (e.g., "Men", "Women").
- **Chatbot Search:**
  - Uses LangChain’s SelfQueryRetriever to interpret structured filters from natural queries.
  - Offers interactive and visual responses.
- **S3 Integration with Triggers:**
  - Uploading a data file to S3 (LocalStack) triggers a Lambda-like function to rebuild your vector index automatically.
- **Collaborative Filtering (Planned):**
  - Combines user behavior and interest profiles into a single embedding to find similar users/products.

---

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yo1am1/clothes_recommender.git
   cd clothes_recommender
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data (Optional):**

   A sample dataset is included in `data/` in archive: `images.zip`. To use your own data:

   - Replace the CSV/image folder.
   - Then run the embedding process:

   ```bash
   python ./utils/data_processing.py --force  # or --cont or --validate
   python ./utils/embeddings/image_embeddings.py
   ```

5. **Start the System:**
   - Start with Docker compose:

    ```bash
      docker-compose build --no-cache --parallel
      docker-compose up
    ```

6. **Initialize Localstack bucket:**
    - Run shell command to start LocalStack:

    ```shell
     ./localstack_setup/init_localstack.sh
    ```

    - Use `awslocal` inside localstack container to test:

    ```bash  
    awslocal s3 mb s3://my-clothes-bucket
    awslocal s3 cp sample.csv s3://my-clothes-bucket/
    ```

---

## 🧪 FastAPI API Docs

You can access FastAPI interactive docs here:

```
http://127.0.0.1:8502/docs
```

---

## 📍 Next Steps & Roadmap

✅ Working:

- ChromaDB + Streamlit UI
- LocalStack S3 + Lambda-style triggers
- Location-aware recommendations

🧩 In Progress:

- Pinecone vector store migration
- Collaborative recommendation via fused embeddings (user + interest)
- S3-triggered real-time learning updates
- Automated ETL pipeline from e-commerce site to our system

---

## 💡 Inspiration

> "When was the last time you looked in your wardrobe and thought — how many of these items were actually recommended to me?"

Clothy aims to close that loop — with a system that learns, adapts, and helps users discover what fits them, both in size and in style.
