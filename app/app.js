import express from "express";
import bodyParser from "body-parser";
import { QdrantClient } from "@qdrant/js-client-rest";
import { pipeline } from "@xenova/transformers";

const COLLECTION_NAME = "logica";
const app = express();
app.use(bodyParser.json());

// Cargar modelo UNA vez
const embedder = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);

// Cliente Qdrant
const qdrant = new QdrantClient({
  url: "https://47592da0-9385-46e2-aeef-f6ecc8e84bd3.europe-west3-0.gcp.cloud.qdrant.io:6333",
  apiKey: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.e0f9F-rKs6Pmkhk4_MrZl51CcC3YbYLatE0tDN26Mls",
});

app.post("/chat", async (req, res) => {
  const question = req.body.message;

  if (!question) {
    return res.status(400).json({ error: "Mensaje vacío" });
  }

  // Embedding
  const emb = await embedder(question, {
    pooling: "mean",
    normalize: true,
  });

  // Qdrant
  const results = await qdrant.search(COLLECTION_NAME, {
    vector: Array.from(emb.data),
    limit: 5,
    score_threshold: 0.35,
  });

  if (results.length === 0) {
    return res.json({
      answer: "No tengo información suficiente para responder a esa pregunta.",
    });
  }

  const context = results.map(r => r.payload.texto).join("\n");

  // Gemini
  const answer = await askGemini(question, context);

  res.json({ answer });
});


app.listen(3000, () => {
  console.log("Backend RAG escuchando en http://localhost:3000");
});
