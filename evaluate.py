# evaluate.py
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, test_loader, device, dataset_name="Test"):
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f'Evaluating on {dataset_name} Set')
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            test_pbar.set_postfix({
                'Acc': f'{100.*test_correct/test_total:.2f}%'
            })

    test_accuracy = 100 * test_correct / test_total
    return test_accuracy, all_predictions, all_labels


def print_metrics(labels, predictions):
    print(f"1. Accuracy: {accuracy_score(labels, predictions):.4f}")
    print(f"2. Precision (Weighted): {precision_score(labels, predictions, average='weighted'):.4f}")
    print(f"3. Recall (Weighted): {recall_score(labels, predictions, average='weighted'):.4f}")
    print(f"4. F1-Score (Weighted): {f1_score(labels, predictions, average='weighted'):.4f}")