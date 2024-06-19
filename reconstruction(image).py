import glob
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from os.path import join
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import lpips  # Import LPIPS module

# Define paths
TRAIN_PATH = r"C:\Users\KIIT\Desktop\archive\dataset\Images\Train"  # Change this to your Celeb dataset path
LOGS_Path = "./logs/"
CHECKPOINTS_PATH = './checkpoints/'
SAVED_MODELS = './saved_models'
RECONSTRUCTED_IMAGES_PATH = './reconstructed_images/'


# Create directories if they do not exist
##os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
#os.makedirs(RECONSTRUCTED_IMAGES_PATH, exist_ok=True)
#os.makedirs(LOGS_PATH, exist_ok=True)
#os.makedirs(SAVED_MODELS, exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MessageProcessor(nn.Module):
    def __init__(self):
        super(MessageProcessor, self).__init__()
        # Define layers for message processing
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)  # Adjusted output channels to 8
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # Adjusted output channels to 8
        self.bn2 = nn.BatchNorm2d(8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Define layers for encoder
        self.conv1 = nn.Conv2d(111, 64, kernel_size=3, padding=1)  # Adjusted input channels
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.instance_norm = nn.InstanceNorm2d(64)  # Apply instance normalization

    def forward(self, x):
        x = self.conv1(x)
        x = self.instance_norm(x)  # Apply instance normalization
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Adjust the input size of fc1 to match the encoder output
        self.fc1 = nn.Linear(64 * 16 * 16, 256)

        # Define layers for binary code output
        self.fc_binary = nn.Linear(256, 100)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid for binary output

        # Define layers for image reconstruction
        self.fc_image = nn.Linear(256, 3 * 64 * 64)
        self.tanh = nn.Tanh()  # Add tanh for image output normalization

        # Define layers for latent vector recovery
        self.fc_latent = nn.Linear(256, 100)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.fc1(x)

        # Binary code branch
        binary_code = self.fc_binary(x)
        binary_code = self.sigmoid(binary_code)  # Apply sigmoid to get probabilities

        # Image reconstruction branch
        reconstructed_image = self.fc_image(x)
        reconstructed_image = self.tanh(reconstructed_image)
        reconstructed_image = reconstructed_image.view(-1, 3, 64, 64)  # Reshape to image size

        # Latent vector recovery branch
        latent_vector = self.fc_latent(x)

        return binary_code, reconstructed_image, latent_vector


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.fc = nn.Linear(100, 3 * 32 * 32)  # Adjusted to match the intermediate image size
        self.message_processor = MessageProcessor()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, noise, image, latent_vector):
        noise = noise.view(-1, 100)
        x = self.fc(noise)
        x = x.view(-1, 3, 32, 32)
        x = self.message_processor(x)
        image_resized = F.interpolate(image, size=(32, 32), mode='nearest')
        x = torch.cat((x, image_resized), dim=1)  # x now has 3+8=11 channels
        latent_vector_resized = latent_vector.view(-1, 100, 1, 1)  # Reshape latent vector
        latent_vector_resized = latent_vector_resized.expand(-1, -1, 32, 32)  # Expand latent vector to match image size
        x = torch.cat((x, latent_vector_resized), dim=1)  # Concatenate latent vector, x now has 111 channels
        x = self.encoder(x)
        binary_code, reconstructed_image, recovered_latent_vector = self.decoder(x)
        return binary_code, reconstructed_image, recovered_latent_vector



def get_img_batch(files_list, batch_size=2, size=(64, 64)):
    batch_cover = []

    for i in range(batch_size):
        img_cover_path = random.choice(files_list)
        try:
            img_cover = Image.open(img_cover_path).convert("RGB")
            img_cover = img_cover.resize(size)  # Resize the image to 64x64
            img_cover = np.array(img_cover, dtype=np.float32) / 255.
            img_cover = np.transpose(img_cover, (2, 0, 1))  # Transpose to (3, 64, 64)
        except:
            img_cover = np.zeros((3, size[0], size[1]), dtype=np.float32)
        batch_cover.append(img_cover)

    batch_cover = np.array(batch_cover)
    return batch_cover


def get_binary_batch(batch_size=2, bit_length=100):
    batch = []
    for _ in range(batch_size):
        binary_tensor = torch.randint(0, 2, size=(100,), dtype=torch.float32)
        batch.append(binary_tensor)
    return batch


def get_latent_batch(batch_size=2, bit_length=100):
    batch = []
    for _ in range(batch_size):
        latent_tensor = torch.randn(bit_length, dtype=torch.float32)  # Generate random latent vectors
        batch.append(latent_tensor)
    return batch


def plot_images(original, reconstructed, step, latent_vector=None):
    fig, axes = plt.subplots(nrows=2, ncols=original.size(0), figsize=(15, 5))

    for i in range(original.size(0)):
        axes[0, i].imshow(original[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).cpu().detach().numpy())
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.suptitle(f"Step {step}")
    plt.show()

    if latent_vector is not None:
        print(f"Latent Vector at Step {step}:\n", latent_vector.cpu().detach().numpy())


def main(binary_input, latent_input, batch_size):
    num_steps = 50000
    lr = 0.0001
    files_list = glob.glob(join(TRAIN_PATH, "**/*"))
    print("Number of files found:", len(files_list))

    model = Autoencoder().to(device)
    criterion_binary = nn.BCELoss()
    criterion_image = nn.MSELoss()
    criterion_latent = nn.MSELoss()  # Criterion for latent vector recovery
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize LPIPS loss function
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)  # Use GPU if available

    for global_step in range(num_steps):
        images = get_img_batch(files_list, batch_size)
        if images is None:
            continue

        images_tensor = torch.from_numpy(images).to(device)
        noise_tensor = torch.randn(batch_size, 100).to(device)

        binary_input_tensor = torch.stack(binary_input).to(device)
        latent_input_tensor = torch.stack(latent_input).to(device)

        optimizer.zero_grad()
        binary_output, image_output, latent_output = model(noise_tensor, images_tensor, latent_input_tensor)

        loss_binary = criterion_binary(binary_output, binary_input_tensor)
        loss_image = criterion_image(image_output, images_tensor)
        loss_latent = criterion_latent(latent_output, latent_input_tensor)

        loss_lpips = loss_fn_lpips(image_output, images_tensor).mean()

        total_loss = loss_binary + loss_image + loss_latent + loss_lpips

        total_loss.backward()
        optimizer.step()

        if global_step % 100 == 0:
            print(f"Step [{global_step}/{num_steps}], Loss: {total_loss.item():.4f}")
            plot_images(images_tensor, image_output, global_step, latent_vector=latent_output)
            # Save the reconstructed images
            reconstructed_images = image_output.cpu().detach()
            for i in range(batch_size):
                img_path = os.path.join(RECONSTRUCTED_IMAGES_PATH, f"reconstructed_{global_step}_{i}.png")
                vutils.save_image(reconstructed_images[i], img_path)

    # Save the model
    torch.save(model.state_dict(), os.path.join(SAVED_MODELS, 'autoencoder.pth'))


if __name__ == "__main__":
    binary_input = get_binary_batch(batch_size=2, bit_length=100)
    latent_input = get_latent_batch(batch_size=2, bit_length=100)
    batch_size = 2
    main(binary_input, latent_input, batch_size)
