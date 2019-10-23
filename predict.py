import argparse
import torch
from func import load_checkpoint, process_image, predict
import json

def arg_parser():
    parser = argparse.ArgumentParser(
        description="Predict Class of Image"
    )
    
    parser.add_argument("image_dir", type=str, help="Path to Image")
    parser.add_argument("checkpoint_dir", type=str, help="Path to Checkpoint")
    parser.add_argument("--top_k", type=int, help="Top K Classes", default=1)
    parser.add_argument("--category_names", type=str, help="Mapping of Categories to Real Names File Path")
    parser.add_argument('--gpu', action='store_true', help='Enable GPU')
    
    arguments = parser.parse_args()
    return arguments

if __name__ == "__main__":
    
    arguments = arg_parser()
    
    with open(arguments.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    image_dir = arguments.image_dir
    checkpoint_dir = arguments.checkpoint_dir
    topk = arguments.top_k
    gpu = arguments.gpu
    device = torch.device("cuda" if gpu else "cpu")

   
    model, class_to_idx = load_checkpoint(checkpoint_dir, device)
    
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    top_p, top_label = predict(image_dir, model, topk,device,idx_to_class)

    print("Prediction:")
    for probabilities, classes in zip(top_p[0].tolist(), top_label):
        print(cat_to_name[classes],"{:.4f}".format(probabilities))