import clip
import torch 

from .utils import class_names


if __name__ == "__main__":
    device = "cuda"
    model, _ = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        text_prompts = [f"a photo of a {label}" for label in class_names]
        text_tokens = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(device)
        text_features = model.encode_text(text_tokens).cpu()

    text_features /= text_features.norm(dim=-1, keepdim=True)  
    torch.save(text_features, "cifar100_text_embeddings.pt")