# Run 'make help' for usage information

SHELL := /bin/sh

DIST_DIR := dist
APP_DIR := app

.PHONY: help
help:
	@echo; grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'; echo


.PHONY: install
install:  ## Create venv and install dependencies
	poetry install



.PHONY: train
train:  ## Train the model
	poetry run python trainer.py
	@echo "Training complete."



.PHONY: clean
clean:  ## Clean up files from prior builds
	mkdir -p $(DIST_DIR)

	@if command -v trash > /dev/null; then \
		trash $(DIST_DIR)/* > /dev/null 2>&1; \
	else \
		rm -rf $(DIST_DIR)/*; \
	fi

	@echo "Cleaning complete."



.PHONY: build
build:  ## Build the inference app
	cp -r $(APP_DIR)/* $(DIST_DIR)/

	cp module.py datamodule.py config.py utils.py $(DIST_DIR)/

	mkdir -p $(DIST_DIR)/checkpoints
	poetry run python -c 'import os; from utils import best_checkpoint; os.system(f"cp {best_checkpoint()} ./$(DIST_DIR)/checkpoints/")'

	mkdir -p $(DIST_DIR)/data
	cp -r data/models--* $(DIST_DIR)/data/

	@echo "Build complete."



.PHONY: serve
serve:  ## Serve the inference app on localhost
	@echo "Serving the inference app on localhost..."
	cd $(DIST_DIR) && poetry run python main.py



.PHONY: deploy
deploy:  ## Deploy the inference app to GCP Cloud Run
	@echo "Deploying to GCP Cloud Run..."
	cd $(DIST_DIR) && gcloud run deploy ai-text-detect-app --source . --cpu 4000m --memory 4Gi
	@echo "Deployment complete."

