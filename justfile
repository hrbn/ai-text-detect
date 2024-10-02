# This file is used to build and deploy the inference app to GCP Cloud Run
# Run using the "just" command runner (https://github.com/casey/just)
# Alternatively, these commands can be executed manually in the terminal

build:
    #!/usr/bin/env zsh
    setopt extended_glob
    cd {{justfile_directory()}}

    # Create dist directory if it doesn't exist
    mkdir -p dist

    # Clean up files from previous builds
    if command -v trash > /dev/null; then
        trash dist/* > /dev/null
    else
        rm -rf dist/*
    fi

    # Copy GCP service files from app directory
    cp -r app/* dist/

    # Copy the model config files
    cp module.py dist/
    cp datamodule.py dist/
    cp config.py dist/
    cp utils.py dist/

    # Copy the saved model
    mkdir -p dist/checkpoints
    python -c "import os; from utils import best_checkpoint; os.system(f'cp {best_checkpoint()} ./dist/checkpoints/')"


    mkdir -p dist/data
    cp -r data/models--distilbert--distilbert-base-cased dist/data/

deploy:
    # Change to the dist directory and deploy to gcloud
    cd {{justfile_directory()}}/dist && gcloud run deploy ai-text-detect-app --source . --cpu 4000m --memory 4Gi