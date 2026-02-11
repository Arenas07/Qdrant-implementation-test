import { QdrantClient } from "@qdrant/js-client-rest";
import { pipeline } from "@xenova/transformers";

const COLLECTION_NAME = "logica";
const VECTOR_SIZE = 384;

async function main() {
  console.log("Cargando modelo de embeddings...");
  const embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );
  console.log("Modelo cargado correctamente.");

  console.log("Conectando con Qdrant...");
  const client = new QdrantClient({
    url: "https://47592da0-9385-46e2-aeef-f6ecc8e84bd3.europe-west3-0.gcp.cloud.qdrant.io:6333",
    apiKey:
      "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.e0f9F-rKs6Pmkhk4_MrZl51CcC3YbYLatE0tDN26Mls",
  });

  const collections = await client.getCollections();
  const exists = collections.collections.some(
    (c) => c.name === COLLECTION_NAME
  );

  if (!exists) {
    await client.createCollection(COLLECTION_NAME, {
      vectors: {
        size: VECTOR_SIZE,
        distance: "Cosine",
      },
    });
    console.log("Colección creada.");
  } else {
    console.log("La colección ya existe.");
  }

  const textos = [
    "La empresa es una pizzería local que opera exclusivamente en la ciudad de Bucaramanga, Colombia.",
    "No realiza envíos ni presta servicios fuera del área urbana de Bucaramanga.",
    "El objetivo del negocio es ofrecer pizzas artesanales, bebidas y combos familiares con servicio a domicilio y atención en sucursal.",
    "La pizzería solo atiende pedidos dentro de Bucaramanga.",
    "Si un cliente se encuentra fuera de la ciudad, el pedido no puede ser procesado.",
    "No se realizan envíos a Floridablanca, Girón, Piedecuesta ni otras ciudades.",
    "Horario general de atención: Lunes a jueves: 12:00 p.m. a 9:30 p.m. Viernes y sábado: 12:00 p.m. a 11:00 p.m. Domingo: 1:00 p.m. a 9:00 p.m.",
    "Los pedidos a domicilio se aceptan hasta 30 minutos antes del cierre.",
    "Sucursal Centro: Dirección Carrera 15 #34-45, Centro, Bucaramanga",
    "Sucursal Cabecera: Dirección Calle 48 #35-20, Cabecera del Llano, Bucaramanga",
    "Sucursal Provenza: Dirección: Avenida Bucarica #105-18, Provenza, Bucaramanga",
    "Pizza Pepperoni Clásica Pizza de salsa de tomate artesanal, queso mozzarella y pepperoni. Disponible en tamaños personal, mediana y familiar.",
    "Pizza Hawaiana Pizza con jamón, piña y queso mozzarella. Disponible en tamaños mediana y familiar.",
    "Pizza Cuatro Quesos Mezcla de mozzarella, parmesano, queso azul y queso crema. Disponible en tamaño mediano y familiar.",
    "Pizza Vegetariana Pizza con pimentón, champiñones, cebolla, aceitunas y tomate. Disponible en todos los tamaños.",
    "Bebidas disponibles: Gaseosa Coca-Cola 400 ml. Gaseosa Coca-Cola 1.5 L. Gaseosa Colombiana 400 ml. Agua sin gas 600 ml. Jugo natural de mora. Jugo natural de mango",
    "Costos de envío: El costo de envío dentro de Bucaramanga es de $5.000 COP. En pedidos superiores a $50.000 COP el envío es gratuito.",
    "El agente de IA actúa como asistente virtual de la pizzería.",
    "Sus funciones incluyen: Responder preguntas sobre productos, precios y horarios. Informar direcciones y teléfonos de las sucursales. Explicar promociones vigentes. Confirmar si un pedido puede realizarse según la ubicación del cliente. No inventar información que no esté en el contexto",
    "El agente no debe aceptar pedidos directamente ni procesar pagos.",
    "No debe responder preguntas fuera del contexto del negocio. No debe dar información de otras ciudades. No debe asumir precios no especificados. Si la información no está disponible, debe indicarlo claramente",
  ];

  console.log("Generando embeddings...");
  const vectors = await Promise.all(
    textos.map(async (t) => {
      const emb = await embedder(t, {
        pooling: "mean",
        normalize: true,
      });
      return Array.from(emb.data);
    })
  );

  const points = textos.map((texto, i) => ({
    id: i,
    vector: vectors[i],
    payload: { texto },
  }));

  await client.upsert(COLLECTION_NAME, {
    wait: true,
    points,
  });

  console.log("Textos insertados correctamente.");


}

main().catch(console.error);
