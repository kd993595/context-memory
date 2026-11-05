package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

// All hugot-specific code lives in this file. If a hugot upgrade changes its
// API, this is the only file that should need touching.

const (
	// embeddingModel is the Hugging Face repo of the model. It is downloaded
	// once (weights only) into the models directory and then runs fully
	// in-process; no inference service is ever called.
	embeddingModel = "BAAI/bge-small-en-v1.5"

	// onnxFilePath is the ONNX export inside that repo.
	onnxFilePath = "onnx/model.onnx"
)

// embedder turns text into embeddingDim-sized vectors using bge-small-en-v1.5
// running on hugot's pure-Go backend (no onnxruntime shared library needed).
// It is created once during startup (see setup in main.go). The mutex
// serializes access because hugot pipelines aren't documented as
// goroutine-safe, and the TUI can have more than one search in flight.
type embedder struct {
	mu       sync.Mutex
	session  *hugot.Session
	pipeline *pipelines.FeatureExtractionPipeline
}

// modelsDir returns $XDG_DATA_HOME/<appName>/models.
func modelsDir() (string, error) {
	dir, err := dataDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "models"), nil
}

// ensureModel returns the local model directory, downloading it on first use.
// logf may be nil. It receives status messages such as the first-run download
// notice. The CLI routes them to stderr; the TUI routes them to its loading
// screen, because writing to stderr would corrupt the alternate screen.
func ensureModel(ctx context.Context, logf func(string)) (string, error) {
	root, err := modelsDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", fmt.Errorf("creating models dir: %w", err)
	}

	// hugot stores a downloaded model in <dest>/<repo with "/" replaced by "_">.
	modelPath := filepath.Join(root, strings.ReplaceAll(embeddingModel, "/", "_"))
	if _, err := os.Stat(filepath.Join(modelPath, "model.onnx")); err == nil {
		return modelPath, nil
	}

	if logf != nil {
		logf(fmt.Sprintf("Downloading %s (first run only)...", embeddingModel))
	}
	opts := hugot.NewDownloadOptions()
	opts.OnnxFilePath = onnxFilePath
	downloaded, err := hugot.DownloadModel(ctx, embeddingModel, root, opts)
	if err != nil {
		return "", fmt.Errorf("downloading %s: %w", embeddingModel, err)
	}
	return downloaded, nil
}

func newEmbedder(ctx context.Context, logf func(string)) (*embedder, error) {
	modelPath, err := ensureModel(ctx, logf)
	if err != nil {
		return nil, err
	}

	session, err := hugot.NewGoSession(ctx)
	if err != nil {
		return nil, fmt.Errorf("starting hugot session: %w", err)
	}

	pipeline, err := hugot.NewPipeline(session, hugot.FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      "bge-small-en-v1.5",
		Options: []hugot.FeatureExtractionOption{
			pipelines.WithNormalization(), // unit-length vectors, matches the cosine index
		},
	})
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("loading embedding pipeline: %w", err)
	}

	return &embedder{session: session, pipeline: pipeline}, nil
}

// Embed returns the embedding for text.
func (e *embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	out, err := e.pipeline.RunPipeline(ctx, []string{text})
	if err != nil {
		return nil, fmt.Errorf("running embedding model: %w", err)
	}
	if len(out.Embeddings) != 1 {
		return nil, fmt.Errorf("embedding model returned %d vectors, want 1", len(out.Embeddings))
	}
	vec := out.Embeddings[0]
	if len(vec) != embeddingDim {
		return nil, fmt.Errorf("embedding has %d dimensions, want %d", len(vec), embeddingDim)
	}
	return vec, nil
}

// Close releases the model. It waits for an in-flight Embed to finish first,
// so the session is never destroyed while it is being used.
func (e *embedder) Close() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}
}

