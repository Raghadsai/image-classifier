import argparse
import torch
from func import load_data, build_model, train_model, calculate_acc, save_checkpoint
def arg_parser():
    parser = argparse.ArgumentParser(
        description='Image Classifier',
    )

    parser.add_argument("data_dir", type=str, help="Data Directory")
    parser.add_argument("--save_dir", type=str, default='/home/workspace/ImageClassifier/')
    parser.add_argument("--arch", type=str,  help="Model Architecture", default='densenet121')
    parser.add_argument("--learning_rate", help="Learning Rate", type=float,default=0.03)
    parser.add_argument("--hidden_units", help="Hidden Units", type=int, default=256)
    parser.add_argument("--epochs", type=int, help="Epochs", default=7)
    parser.add_argument('--gpu', action="store_true", help='Enable GPU')

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    arguments = arg_parser()
    data_dir = arguments.data_dir
    checkpoint_dir = arguments.save_dir
    architecture = arguments.arch
    learning_rate = arguments.learning_rate
    hidden_units = arguments.hidden_units
    epochs = arguments.epochs
    gpu = arguments.gpu
    device = torch.device("cuda" if gpu else "cpu")
    

    dataloaders, validloaders, testloader, image_datasets = load_data(data_dir)
    model, criterion, optimizer = build_model(architecture, learning_rate, hidden_units, epochs,device,image_datasets.class_to_idx)

    train_model(epochs, dataloaders, validloaders, model, criterion, optimizer, device)

    calculate_acc(model, testloader, device)

    save_checkpoint(architecture, model,image_datasets,optimizer,epochs,checkpoint_dir)