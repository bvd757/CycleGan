import torch
import streamlit as st
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tr
import numpy as np
from PIL import Image

MEAN_A = [0.4767, 0.4676, 0.4534]
STD_A = [0.2642, 0.2581, 0.2524]
MEAN_B = [0.2509, 0.2932, 0.2553]
STD_B = [0.1655, 0.1729, 0.1698]

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_res_blocks=6, ngf=32):
        super(Generator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        in_features = ngf
        out_features = in_features * 2

        model += [
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True),
        ]

        in_features = out_features
        out_features = in_features * 2

        model += [
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True),
        ]
        in_features = out_features
        out_features = in_features * 2

        for _ in range(n_res_blocks):
            model += [ResnetBlock(in_features)]

        out_features = in_features // 2

        model += [
            nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True),
        ]
        in_features = out_features
        out_features = in_features // 2

        model += [
            nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True),
        ]
        in_features = out_features
        out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = Path(__file__).parent

    G_AB = Generator().to(device)
    G_BA = Generator().to(device)

    G_AB.load_state_dict(torch.load(f"{base_path}/G_AB_5.pth", map_location=device))
    G_BA.load_state_dict(torch.load(f"{base_path}/G_BA_5.pth", map_location=device))

    return G_AB, G_BA, device


def de_normalize_a(tensor):
    mean = torch.tensor(MEAN_A, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(STD_A, device=tensor.device).view(3, 1, 1)
    normalized = tensor * std + mean
    return normalized.permute(1, 2, 0).cpu().numpy()

def de_normalize_b(tensor):
    mean = torch.tensor(MEAN_B, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(STD_B, device=tensor.device).view(3, 1, 1)
    normalized = tensor * std + mean
    return normalized.permute(1, 2, 0).cpu().numpy()

def predict(image, gen, device, from_reality=True) -> Image:
    original_size = image.size
    
    test_transforms = tr.Compose([
        tr.Resize((256, 256)),
        tr.ToTensor(),
        tr.Normalize(MEAN_A if from_reality else MEAN_B, 
                     STD_A if from_reality else STD_B),
    ])
    
    transformed_image = test_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        gen.eval()
        output = gen(transformed_image)
    
    output = output.squeeze(0)
    result = de_normalize_a(output) if from_reality else de_normalize_b(output)
    
    result_image = Image.fromarray((result * 255).astype(np.uint8))
    result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)
    
    return result_image

def main():
    st.title("CycleGAN: Reality ↔ GTA Style Transfer")
    
    G_AB, G_BA, device = get_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    direction = st.radio("Transfer Direction", ["From Reality to GTA", "From GTA to Reality"])
    
    generate_button = st.button("Generate")
    
    if generate_button and uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            from_reality = direction == "From Reality to GTA"
            gen_model = G_AB if from_reality else G_BA
            
            with st.spinner("Generating..."):
                result_image = predict(image, gen_model, device, from_reality)
                
                with col2:
                    st.image(result_image, 
                            caption="Transformed Image", 
                            use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    elif generate_button and uploaded_file is None:
        st.error("Please upload an image first")


if __name__ == "__main__":
    main()
