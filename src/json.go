package src

import (
	"encoding/json"
	"fmt"
	"mime"
	"net/http"
)

func readRequestJSON(req *http.Request, target any) error {
	contentType := req.Header.Get("Content-Type")
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return err
	}
	if mediaType != "application/json" {
		return fmt.Errorf("expect application/json Content-TYpe, got %s", mediaType)
	}

	decoder := json.NewDecoder(req.Body)
	decoder.DisallowUnknownFields()
	return decoder.Decode(target)
}

func renderJSON(w http.ResponseWriter, v any) {
	js, err := json.Marshal(v)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(js)
}