package main

import (
	"context"
	"github.com/joho/godotenv"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/googleai"
	"github.com/tmc/langchaingo/vectorstores/weaviate"
	"go-langchain/src"
	"log"
	"net/http"
	"os"
)

const embeddingModelName = "text-embedding-004"

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file: %v", err)
	}
	ctx := context.Background()
	apiKey := os.Getenv("GEMINI_API_KEY")
	geminiClient, err := googleai.New(ctx,
		googleai.WithAPIKey(apiKey),
		googleai.WithDefaultEmbeddingModel(embeddingModelName))
	if err != nil {
		log.Fatal(err)
	}

	emb, err := embeddings.NewEmbedder(geminiClient)
	if err != nil {
		log.Fatal(err)
	}

	wvStore, err := weaviate.New(
		weaviate.WithEmbedder(emb),
		weaviate.WithScheme("http"),
		weaviate.WithHost("localhost:"+os.Getenv("WVPORT")),
		weaviate.WithIndexName("Document"),
	)

	server := &src.RagServer{
		Ctx:          ctx,
		WvStore:      wvStore,
		GeminiClient: geminiClient,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /add", server.AddDocumentHandler)
	mux.HandleFunc("POST /query/", server.QueryHandler)

	port := os.Getenv("SERVERPORT")
	address := "localhost:" + port
	log.Printf("Listening on %s ...\n", address)
	log.Fatal(http.ListenAndServe(address, mux))
}
