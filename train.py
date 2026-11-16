# train.py
import os
import json
import time
import torch
from tqdm import tqdm

from config import CHECKPOINT_PATH, BEST_MODEL_PATH, HISTORY_FILE, PATIENCE


def resume_training_fresh(
    model, train_loader, val_loader, criterion, optimizer,
    additional_epochs=5,
    checkpoint_path=CHECKPOINT_PATH,
    best_model_path=BEST_MODEL_PATH,
    history_file=HISTORY_FILE,
    patience=PATIENCE,
    device='cuda'
):
    if os.path.exists(checkpoint_path):
        print("Checkpoint found. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_accuracies = checkpoint['val_accuracies']
        best_val_acc = checkpoint['best_val_acc']
        best_epoch = checkpoint.get('best_epoch', start_epoch)

        print(f"Resuming from epoch {start_epoch}")
        print(f"Previous best val acc: {best_val_acc:.2f}%")
    else:
        print("No checkpoint found. Starting fresh training...")
        start_epoch = 0
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        best_val_acc = 0.0
        best_epoch = 0

    no_improve = 0
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + additional_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{start_epoch+additional_epochs} - Training')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{start_epoch+additional_epochs} - Validation')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })

        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_acc = 100. * val_correct / val_total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        print(f'Epoch [{epoch+1}/{start_epoch+additional_epochs}]:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated! Val Acc = {best_val_acc:.2f}%")
            no_improve = 0
        else:
            no_improve += 1
            print(f"No improvement for {no_improve}/{patience} epochs.")
            if no_improve >= patience:
                print("Early stopping triggered!")
                break

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch
        }, checkpoint_path)
        print("Checkpoint saved.")
        print('-' * 60)

        with open(history_file, 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch
            }, f)

    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\nTotal Training Time: {int(minutes)} min {int(seconds)} sec")
    print(f"Best Model - Epoch: {best_epoch}, Val Acc: {best_val_acc:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies