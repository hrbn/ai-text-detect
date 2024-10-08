# AI-Generated Text Detector

A PyTorch Lightning model designed to detect text generated by Large Language Models (LLMs). This project fine-tunes the DistilBERT model on a dataset comprising AI-generated and human-written text, compiled from the following sources:

- [artem9k/ai-text-detection-pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile)
- [thedrcat/daigt-v2-train-dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)
- [sunilthite/llm-detect-ai-generated-text-dataset](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset)

The model is deployed as a web application and API on Google Cloud Platform (GCP).

## Installation

[Poetry](https://python-poetry.org/docs/#installation) is used for dependency management.

```bash
git clone https://github.com/hrbn/ai-text-detect.git
cd ai-text-detect
make install
```

## Training

```
make train
```

or, for additional options, run trainer.py with optional command-line arguments:

```
# display available options for training
poetry run python trainer.py --help
```


## Deployment

```
# Build the Flask app
make clean
make build

# Serve the app locally
make serve

# Deploy to GCP (assumes gcloud is installed and a project is already set up)
make deploy
```

## Usage

Once the app is running, you can feed it text to evaluate via the web interface or API.

### Web Interface

Access the web interface at http://localhost:8080.

### API

Send a POST request with your text to the /predict endpoint. For example:

```bash

curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text here"}'

```
