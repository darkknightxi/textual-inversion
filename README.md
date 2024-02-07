# Textual Inversion
This repository implements the Textual Inversion technique ([paper](https://textual-inversion.github.io/)), empowering you to personalize image generation models like Stable Diffusion using just a handful of example images. 

## Description
Textual Inversion is a training technique for personalizing image generation models with just a few example images of what you want it to learn. 
This technique works by learning and updating the text embeddings (the new embeddings are tied to a special word you must use in the prompt) to match the example images you provide.
Textual Inversion is built on top of the Latent Diffusion Models by modifying the learned embeddings by associating a particular token to some images (using only 3-5 images
work well) which represent a certain style or concept that needs to be captured in the output image.
![schematic by the authors of the paper describing how Textual Inversion works](https://textual-inversion.github.io/static/images/training/training.JPG)
This project first builds a typical Stable Diffusion image-to-image inference pipeline by putting together its components, namely Variational Autoencoder, U-Net, Scheduler
and Text Encoder, allowing for better control on the output and better understanding of the workings of the model. Then, we modify the text embeddings by the replacing a less-used token ('sd</w>' in this case) with the embedding vectors corresponding to the 
example concept (GTA 5 artwork) images. 

## Getting Started
1. CLone this repository:
   `git clone https://github.com/darkknightxi/textual-inversion.git`
3. Install Dependencies:
   `pip install -r requirements.txt`

## Explore the Deployed Web App:
Check out the deployed [web app](https://huggingface.co/spaces/darkknightxi/grand-theft-photo) over on 🤗 spaces and create your own GTA 5 characters.

## Example
Here's SRK if he were a character in GTA 5.

<img src="https://forums.fast.ai/uploads/default/optimized/3X/9/d/9dab8fbf4cd88cffd071d0c8dae1fdd4b5a11cd9_2_373x562.jpeg" alt="SRK" height = "400" width="400"/> <img src="https://forums.fast.ai/uploads/default/optimized/3X/a/f/af9887d7c49428a7ee75048d7b278283d553f6c6_2_375x375.png" alt="SRK in GTA 5" width="400"/>


