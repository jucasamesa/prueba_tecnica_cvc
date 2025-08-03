# Contents of /image-classification-app/image-classification-app/src/evaluation/evaluate.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.data.dataset import CustomDataset  # Adjust the import based on your dataset class
from src.models.model import MyModel  # Adjust the import based on your model class

def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall

def main():
    # Load your model and data here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel()  # Initialize your model
    model.load_state_dict(torch.load('path_to_your_model.pth'))  # Load your trained model weights
    model.to(device)

    # Create your dataloader
    test_dataset = CustomDataset('path_to_your_test_data')  # Adjust the path and dataset class
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    accuracy, precision, recall = evaluate_model(model, test_dataloader, device)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

if __name__ == "__main__":
    main()