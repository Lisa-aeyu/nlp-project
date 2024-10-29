import torch
import string
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import string
import os
from torch.utils.data import DataLoader
from time import time
from nltk.corpus import stopwords
stop_words = set(stopwords.words('russian'))
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
mystem = Mystem() 
import torch.nn.functional as F


def l_predict_class(model, text, preprocess_fn, padding, seq_len, vocabulary, device='cuda'):
    model.to(device)
    model.eval()  # Устанавливаем модель в режим оценки
    
    # Предобработка текста
    text = preprocess_fn(text)
    text_list = [vocabulary[word] for word in text if word in vocabulary]

    # Пэддинг и преобразование в тензор
    padded_text = padding([text_list], seq_len, padding='right', truncating='right')[0]
    text_tensor = torch.tensor(padded_text, dtype=torch.long).to(device).unsqueeze(0)

    # Прогон через модель
    with torch.no_grad():  # Отключаем градиенты
        output = model(text_tensor)  # Получаем только один выходной тензор
    
    # Преобразуем логиты в вероятности
    probabilities = F.softmax(output, dim=1)
    
    # Получаем предсказанный класс и вероятность
    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()  # Вероятность предсказанного класса
    
    # Преобразуем числовой класс в строковый
    pred_dict = {
        0: 'Bad',
        1: 'Neutral',
        2: 'Good'
    }
    predicted_label = pred_dict.get(predicted_class, "Unknown")
    
    return predicted_label, confidence


def data_preprocessing(text):
    text = re.sub(r"[^а-яА-ЯёЁ.,!?\":\s]", "", text)
    text = re.sub(r"([.,!?\":])\1+", r"\1", text)
    text = re.sub(r"\s+([.,!?\":])", r"\1", text)
    text = text.lower()
    tokens = mystem.lemmatize(text)
    tokens = [token for token in tokens if token not in stop_words and token != " " and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    return text

def get_words_by_freq(sorted_words: list, n: int = 10) -> list:
    return list(filter(lambda x: x[1] > n, sorted_words))


def logs_dict(logs) -> pd.DataFrame:
    tl, ta, vl, va = logs
    logs_dict_ = {
    'train_losses': tl,
    'train_accs': ta,
    'valid_losses': vl,
    'valid_accs':  va
    }  
    

def logs_dict_multi_metrics(logs) -> pd.DataFrame:

    train_losses, val_losses, train_metrics, val_metrics = logs
    logs_dict_ = {
        'train_loss': [round(value, 2) for value in train_losses],
        'val_loss': [round(value, 2) for value in val_losses],
    }

    for metric_name, metric_values in train_metrics.items():
        rounded_values = [round(value, 2) for value in metric_values]
        logs_dict_[f'train_{metric_name}'] = rounded_values

    for metric_name, metric_values in val_metrics.items():
        rounded_values = [round(value, 2) for value in metric_values]
        logs_dict_[f'val_{metric_name}'] = rounded_values

    df = pd.DataFrame.from_dict(logs_dict_, orient='index')

    num_epochs = len(train_losses)
    df.columns = [f'Epoch {i+1}' for i in range(num_epochs)]

    return df


def plot_history(logs, label_1, logs_2=None, label_2=None, grid=True):
    tl, ta, vl, va = logs
    if logs_2 is not None:
        tl_2, ta_2, vl_2, va_2 = logs_2
    fig, ax = plt.subplots(1,2, figsize=(14,5))
    
    ax[0].plot(tl, label=f'Train loss {label_1}', color='green')
    ax[0].plot(vl, label=f'Valid loss {label_1}', linestyle='-.', color='green')
    if logs_2 is not None:
        ax[0].plot(tl_2, label=f'Train loss {label_2}', color='red')
        ax[0].plot(vl_2, label=f'Valid loss {label_2}', linestyle='-.', color='red')
    ax[0].set_title(f'Loss on epoch {len(tl)}')
    ax[0].grid(grid)
    ax[0].legend()
    
    ax[1].plot(ta, label=f'Train Accuracy {label_1}', color='green')
    ax[1].plot(va, label=f'Valid Accuracy {label_1}', linestyle='-.', color='green')
    if logs_2 is not None:
        ax[1].plot(ta_2, label=f'Train Accuracy {label_2}', color='red')
        ax[1].plot(va_2, label=f'Valid Accuracy {label_2}', linestyle='-.', color='red')
    ax[1].set_title(f'Accuracy on epoch {len(ta)}')
    ax[1].grid(grid)
    ax[1].legend()
    
    plt.show()


def plot_history(logs, label_1, logs_2=None, label_2=None, grid=True):
    tl, ta, vl, va = logs
    if logs_2 is not None:
        tl_2, ta_2, vl_2, va_2 = logs_2
        
    # Определяем количество метрик для правильного количества подграфиков
    num_metrics = len(ta)
    fig, ax = plt.subplots(1, num_metrics + 1, figsize=(7 * (num_metrics + 1), 5))
    
    # График потерь
    ax[0].plot(tl, label=f'Train Loss {label_1}', color='green')
    ax[0].plot(vl, label=f'Valid Loss {label_1}', linestyle='-.', color='green')
    if logs_2 is not None:
        ax[0].plot(tl_2, label=f'Train Loss {label_2}', color='red')
        ax[0].plot(vl_2, label=f'Valid Loss {label_2}', linestyle='-.', color='red')
    ax[0].set_title(f'Loss over {len(tl)} epochs')
    ax[0].grid(grid)
    ax[0].legend()
    
    # Графики метрик
    for i, metric_name in enumerate(ta.keys()):
        ax[i + 1].plot(ta[metric_name], label=f'Train {metric_name} {label_1}', color='green')
        ax[i + 1].plot(va[metric_name], label=f'Valid {metric_name} {label_1}', linestyle='-.', color='green')
        
        if logs_2 is not None:
            ax[i + 1].plot(ta_2[metric_name], label=f'Train {metric_name} {label_2}', color='red')
            ax[i + 1].plot(va_2[metric_name], label=f'Valid {metric_name} {label_2}', linestyle='-.', color='red')
        
        ax[i + 1].set_title(f'{metric_name.capitalize()} over {len(ta[metric_name])} epochs')
        ax[i + 1].grid(grid)
        ax[i + 1].legend()
    
    plt.show()


def my_padding(
        review_int: list, seq_len: int,
        padding: str = 'right', truncating: str = 'right'
    ) -> np.array:
    
    features = np.zeros((len(review_int), seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            if padding == 'right':
                features[i, :len(review)] = np.array(review)
            elif padding == 'left':
                features[i, -len(review):] = np.array(review)
            else:
                raise ValueError("padding должен быть 'left' или 'right'")
        else:
            if truncating == 'right':
                truncated_review = review[:seq_len]
            elif truncating == 'left':
                truncated_review = review[-seq_len:]
            else:
                raise ValueError("truncating должен быть 'left' или 'right'")
            
            features[i, :] = np.array(truncated_review)
    return features


def train(
        epochs: int, 
        model: torch.nn.Module, 
        train_loader: DataLoader,
        valid_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        criterion, 
        metric,
        rnn_conf=None,
        patience=5,
        save_path='./models/best_model_weights.pth',
        attention=False,
        multiclass=False
    ) -> tuple: 

    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_train_metric = []
    epoch_valid_metric = []
    time_start = time()
    if not rnn_conf:
        device = 'cpu'
    else: 
        device = rnn_conf.device

    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(epochs):
        batch_losses = []
        batch_metric = []
        model.train()
        model.to(device)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)   

            if attention is False:
                output = model(inputs).squeeze()
            else:
                output, _ = model(inputs)
            loss = criterion(output, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            if multiclass is False:
                batch_metric.append(metric(output,labels).item())
            else:
                batch_metric.append(metric(output.argmax(1),labels).item())
        epoch_train_losses.append(np.mean(batch_losses))
        epoch_train_metric.append(np.mean(batch_metric))
        batch_losses = []
        batch_metric = []
        model.eval()
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                if attention is False:
                    output = model(inputs).squeeze()
                else:
                    output, _ = model(inputs)
            loss = criterion(output, labels.float())
            batch_losses.append(loss.item())
            if attention is False and multiclass is False:
                batch_metric.append(metric(output.squeeze(),labels).item())
            elif attention is True and multiclass is False:
                batch_metric.append(metric(output,labels).item())
            else:
                batch_metric.append(metric(output.argmax(1),labels).item())
        epoch_valid_losses.append(np.mean(batch_losses))
        epoch_valid_metric.append(np.mean(batch_metric))


        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_losses[-1]:.4f} val_loss : {epoch_valid_losses[-1]:.4f}')
        print(f'train_accuracy : {epoch_train_metric[-1]:.2f} val_accuracy : {epoch_valid_metric[-1]:.2f}')
            
        print(25*'==')

        if epoch_valid_losses[-1] < best_val_loss:
            best_val_loss = epoch_valid_losses[-1]
            torch.save(model.state_dict(), save_path)
            print(f'Сохранены лучшие веса модели после {epoch+1} эпохи с валид. лоссом: {best_val_loss:.4f}')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Ранняя остановка на {epoch+1} эпохе')
                break 
        training_time = time() - time_start
    return (epoch_train_losses, epoch_valid_losses, epoch_train_metric, epoch_valid_metric, training_time)

def train2(
        epochs: int, 
        model: torch.nn.Module, 
        train_loader: DataLoader,
        valid_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        criterion, 
        metric,
        rnn_conf=None,
        patience=5,
        save_path='./models/best_model_weights.pth',
        attention=False,
        multiclass=False,
        device='cuda',
        other_metrics=None
    ) -> tuple: 

    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_train_metric = []
    epoch_valid_metric = []
    time_start = time()
    
    if not rnn_conf:
        device = 'cpu'
    else: 
        device = rnn_conf.device

    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(epochs):
        batch_losses = []
        batch_metric = []
        model.train()
        model.to(device)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)   

            output = model(inputs)
            
            if not multiclass:
                output = output.squeeze()
            
            if multiclass:
                labels = labels.long()  # Метки классов для CrossEntropyLoss должны быть LongTensor
                loss = criterion(output, labels)
            else:
                labels = labels.float()  # Метки для бинарной классификации должны быть FloatTensor
                loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
            if multiclass:
                preds = output.argmax(dim=1)
                batch_metric.append(metric(preds, labels).item())
            else:
                preds = torch.round(torch.sigmoid(output))
                batch_metric.append(metric(preds, labels).item())
        epoch_train_losses.append(np.mean(batch_losses))
        epoch_train_metric.append(np.mean(batch_metric))
        batch_losses = []
        batch_metric = []
        model.eval()
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                output = model(inputs)
                if not multiclass:
                    output = output.squeeze()
                if multiclass:
                    labels = labels.long()
                    loss = criterion(output, labels)
                else:
                    labels = labels.float()
                    loss = criterion(output, labels)
                batch_losses.append(loss.item())
                if multiclass:
                    preds = output.argmax(dim=1)
                    batch_metric.append(metric(preds, labels).item())
                else:
                    preds = torch.round(torch.sigmoid(output))
                    batch_metric.append(metric(preds, labels).item())
        epoch_valid_losses.append(np.mean(batch_losses))
        epoch_valid_metric.append(np.mean(batch_metric))


        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_losses[-1]:.4f} val_loss : {epoch_valid_losses[-1]:.4f}')
        print(f'train_accuracy : {epoch_train_metric[-1]:.2f} val_accuracy : {epoch_valid_metric[-1]:.2f}')
            
        print(25*'==')

        if epoch_valid_losses[-1] < best_val_loss:
            best_val_loss = epoch_valid_losses[-1]
            torch.save(model.state_dict(), save_path)
            print(f'Сохранены лучшие веса модели после {epoch+1} эпохи с валид. лоссом: {best_val_loss:.4f}')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Ранняя остановка на {epoch+1} эпохе')
                break 
        training_time = time() - time_start
    return (epoch_train_losses, epoch_valid_losses, epoch_train_metric, epoch_valid_metric, training_time)

def train3(
        epochs: int, 
        model: torch.nn.Module, 
        train_loader: DataLoader,
        valid_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        criterion, 
        metrics: dict,
        rnn_conf=None,
        patience=5,
        save_path='./models/best_model_weights.pth',
        attention=False,
        multiclass=False
    ) -> tuple: 

    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_train_metrics = {name: [] for name in metrics.keys()}
    epoch_valid_metrics = {name: [] for name in metrics.keys()}
    val_predictions = []
    time_start = time()
    if not rnn_conf:
        device = 'cpu'
    else: 
        device = rnn_conf.device

    best_val_loss = float('inf')
    counter = 0

    model.to(device)
    # Перемещаем метрики на устройство
    for metric in metrics.values():
        metric.to(device)

    for epoch in range(epochs):
        batch_losses = []
        # Сброс метрик в начале эпохи
        for metric in metrics.values():
            metric.reset()

        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)   

            output = model(inputs)
            if not multiclass:
                output = output.squeeze()
            if multiclass:
                labels = labels.long()
                loss = criterion(output, labels)
            else:
                labels = labels.float()
                loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

            if multiclass:
                preds_classes = output.argmax(dim=1)  # Индексы предсказанных классов
                preds_logits = output  # Сырые логиты

                # Обновляем метрики
                for name, metric in metrics.items():
                    if name == 'accuracy':
                        metric.update(preds_classes, labels)
                    else:
                        metric.update(preds_logits, labels)
            else:
                preds = torch.round(torch.sigmoid(output))
                for metric in metrics.values():
                    metric.update(preds, labels)

        epoch_train_losses.append(np.mean(batch_losses))
        # Вычисляем метрики за эпоху
        for name, metric in metrics.items():
            epoch_train_metrics[name].append(metric.compute().item())
            metric.reset()

        # Валидация
        batch_losses = []
        for metric in metrics.values():
            metric.reset()
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                if not multiclass:
                    output = output.squeeze()
                if multiclass:
                    labels = labels.long()
                    loss = criterion(output, labels)
                    preds_classes = output.argmax(dim=1)
                    preds_logits = output
                else:
                    labels = labels.float()
                    loss = criterion(output, labels)
                    preds_classes = torch.round(torch.sigmoid(output)).long()

                val_predictions.extend(preds_classes.cpu().numpy())
                batch_losses.append(loss.item())

                if multiclass:
                    for name, metric in metrics.items():
                        if name == 'accuracy':
                            metric.update(preds_classes, labels)
                        else:
                            metric.update(preds_logits, labels)
                else:
                    preds = torch.round(torch.sigmoid(output))
                    for metric in metrics.values():
                        metric.update(preds, labels)

        epoch_valid_losses.append(np.mean(batch_losses))
        for name, metric in metrics.items():
            epoch_valid_metrics[name].append(metric.compute().item())
            metric.reset()

        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_losses[-1]:.4f} val_loss : {epoch_valid_losses[-1]:.4f}')
        for name in metrics.keys():
            train_metric_value = epoch_train_metrics[name][-1]
            val_metric_value = epoch_valid_metrics[name][-1]
            print(f'{name} - train: {train_metric_value:.4f}, val: {val_metric_value:.4f}')
        print(25*'==')

        if epoch_valid_losses[-1] < best_val_loss:
            best_val_loss = epoch_valid_losses[-1]
            torch.save(model.state_dict(), save_path)
            print(f'Сохранены лучшие веса модели после {epoch+1} эпохи с валид. лоссом: {best_val_loss:.4f}')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Ранняя остановка на {epoch+1} эпохе')
                break 
    training_time = time() - time_start
    return (epoch_train_losses, epoch_valid_losses, epoch_train_metrics, epoch_valid_metrics, training_time, val_predictions)
