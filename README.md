# Hateful-Meme-Detection

<img width="307" height="411" alt="download (1)" src="https://github.com/user-attachments/assets/118fa844-0644-47df-a4a4-6a04b5edae96" />

This project implements a multimodal hateful meme detection system using CLIP (ViT-B/32) to understand both images and text.
A lightweight neural classifier is trained on top of CLIP embeddings to classify memes as:
 Hateful or Not Hateful
 

# Goal: Build a practical baseline for automated hateful content moderation.

# Dataset

Facebook Hateful Memes Dataset

🟢 Train: train.jsonl

🔵 Validation: dev.jsonl

⚫ Test: test.jsonl (no labels, inference only)

⚡ For fast experimentation, a filtered subset (~900 samples) was used.
All JSON files were filtered to match existing images to avoid broken samples.

# Model Architecture

Backbone:

🔥 CLIP ViT-B/32 (Frozen Feature Extractor)

#Pipeline:

🖼️ Image → clip.encode_image

📝 Text → clip.encode_text

📏 Feature Normalisation

🔗 Concatenation (Image + Text)

# Classifier Head:

Linear(1024 → 256) + ReLU

Linear(256 → 2)

⚙️ Training Setup

🧪 Loss: Cross-Entropy Loss

🚀 Optimiser: AdamW

🖥️ Acceleration: CUDA (GPU)

🧯 Stability: Gradient Clipping

📊 Metrics: Accuracy, Precision, Recall, F1-score

🧠 CLIP Backbone: Frozen for stability

📈 Results (Subset)

✅ Training Accuracy: Improved steadily

📊 Validation Accuracy & F1: Improved across epochs

# Interpretation: Model learns meaningful multimodal patterns

⚠️ Note: Scores are lower due to small dataset size and frozen CLIP

🖼️ Inference Demo (Human Evaluation)

A demo script is included to:

🎲 Randomly pick a meme from test.jsonl

👀 Display the image + text

🤖 Show the model’s prediction (Hateful / Not Hateful)

🧪 This allows visual inspection of model behaviour without needing labels.

# Limitations

📉 Trained on a small subset of data

🧊 CLIP backbone not fine-tuned yet

⚖️ Dataset is class-imbalanced

🧪 Test set has no labels (dev set used for evaluation)

🚧 Future Work (Work in Progress)

📈 Train on the full dataset

🔓 Fine-tune last layers of CLIP

⚖️ Handle class imbalance with weighted loss

🔍 Add error analysis + confusion matrix

🧩 Ensemble Moderation System (Planned):

Combine CLIP classifier + Vision-Language LLM (BLIP-2 / LLaVA)

Add rule-based heuristics for sensitive symbols & protected groups

Fuse predictions using a meta-classifier / decision logic

▶️ How to Run

1️⃣ Download & unzip dataset
2️⃣ Update paths for JSON + image folders
3️⃣ Train model
4️⃣ Run inference demo on test samples

# Acknowledgements

🔗 OpenAI CLIP

📚 Facebook Hateful Memes Dataset

⚙️ PyTorch, Google Colab, Kaggle
