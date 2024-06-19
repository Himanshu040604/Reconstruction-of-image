# Reconstruction-of-image

**Image Autoencoder**
This project is an implementation of an image autoencoder using PyTorch. The autoencoder is designed to learn efficient representations of images, allowing us to reconstruct images from a compressed form. This is useful in various applications such as image compression and feature extraction.

**Features**
Autoencoder Architecture: The model consists of an encoder, a decoder, and a message processor to transform input images into a compressed binary form and then reconstruct them.
Custom Loss Functions: The project uses Binary Cross-Entropy Loss for binary code generation, Mean Squared Error for image reconstruction, and a custom LPIPS loss for perceptual similarity.
Data Handling: The project can handle batches of images, binary codes, and latent vectors, and includes functionality for loading and preprocessing images from a directory.
Visualization: It includes functions for visualizing original and reconstructed images, as well as latent vectors at various steps in the training process.

**Prerequisites**
The following libraries should be installed:
PyTorch
NumPy
Pillow (PIL)
Matplotlib
LPIPS
