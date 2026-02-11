import express from "express";
import { QdrantClient } from "@qdrant/js-client-rest";
import { pipeline } from "@xenova/transformers";
import { GoogleGenerativeAI } from "@google/generative-ai";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());

app.use(express.json());

const COLLECTION_NAME = "logica";

// Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

// Embeddings
const embedder = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);

// Qdrant
const qdrant = new QdrantClient({
  url: "https://47592da0-9385-46e2-aeef-f6ecc8e84bd3.europe-west3-0.gcp.cloud.qdrant.io",
  apiKey: process.env.QDRANT_API_KEY,
  checkCompatibility: false
});


app.post("/chat", async (req, res) => {
  try {
    const question = req.body.message;

    if (!question) {
      return res.status(400).json({ error: "Mensaje vacío" });
    }

    // 1️⃣ Embedding
    const emb = await embedder(question, {
      pooling: "mean",
      normalize: true,
    });

    // 2️⃣ Buscar en Qdrant
    const results = await qdrant.search(COLLECTION_NAME, {
      vector: Array.from(emb.data),
      limit: 5,
    });

    const context = results.map(r => r.payload.texto).join("\n");

    // 3️⃣ Prompt RAG
    const prompt = `
Eres un asistente virtual de la pizzería.

Contexto:
${context}

Pregunta:
${question}

Responde únicamente usando el contexto proporcionado.
Si no existe información suficiente, dilo claramente.
`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    res.json({ answer: text });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error interno del servidor" });
  }
});

app.listen(3000, () => {
  console.log("Backend RAG escuchando en puerto 3000");
});
