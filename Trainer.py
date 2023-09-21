import torch.optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification,RobertaTokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
import seaborn as sns
from torch.nn import CrossEntropyLoss
import argparse
from dataloader import loading
import transformers
from Model import sentiment_model
import random
import time

transformers.logging.set_verbosity_error()
parser = argparse.ArgumentParser(description="Train a Sentiment classifier - via Transformers")
parser.add_argument("--BERT", type=bool, help="You are using BERT", default=False)
parser.add_argument("--RoBERTa", type=bool, help="You are using RoBERTa", default=False)
parser.add_argument("--DistilBERT", type=bool, help="You are using DistilBERTa", default=True)
parser.add_argument("--Augment", type=bool, help="Augment model classifier", default=False)  # model classifier update

parser.add_argument("--train_file", type=str, help="train dataset name", default='train.tsv')
parser.add_argument("--test_file", type=str, help="test dataset name", default='test.tsv')
parser.add_argument("--eval_file", type=str, help="eval dataset name", default='dev.tsv')
parser.add_argument("--normalized", type=bool, help="normalize the dataset labels frequency", default=True)  # data normalizer

parser.add_argument("--epoch", type=int, help="this is the number of epochs", default=3)
parser.add_argument("--hidden", type=int, help="this is the LSTM hidden_size", default=100)
parser.add_argument("--batch_size", type=int, help="number of samples in each iteration", default=20)
parser.add_argument("--lr", type=float, help="this is learning rate value", default=0.000001)
parser.add_argument("--num_labels", type=int, help="this is the total number of labels", default=3)
parser.add_argument("--max_length", type=int, help="this is maximum length of an utterance", default=100)

parser.add_argument("--L1_reg", type=bool, help="L1 regularizer", default=False)
parser.add_argument("--L2_reg", type=bool, help="L2 regularizer", default=False)
parser.add_argument("--drop_out", type=bool, help="implement a dropout to the model output", default=True)

parser.add_argument("--L1_lambda", type=int, help="Lambda value used for regularization", default=0.01)
parser.add_argument("--L2_lambda", type=int, help="Lambda value used for regularization", default=0.02)
parser.add_argument("--p", type=int, help="Lambda value used for regularization", default=0.5)
args = parser.parse_args()

if args.BERT:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.num_labels)
elif args.RoBERTa:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='RoBERT_CacheDir')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=args.num_labels)
elif args.DistilBERT:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=args.num_labels)

model_object = sentiment_model(model, args.p, args.hidden, args.Augment)
loss = CrossEntropyLoss()
if args.L2_reg:
    optimizer = torch.optim.Adam(model_object.parameters(), lr=args.lr, weight_decay= args.L2_lambda)
else:
    optimizer = torch.optim.Adam(model_object.parameters(), lr=args.lr)

loss_records = []
def train_classifier():
    patience = 0
    curr_loss = 0

    train_text, train_labels = loading(args.train_file, args.normalized)
    train_text = train_text[:50]
    train_labels = train_labels[:50]

    test_text, test_labels = loading(args.test_file, args.normalized)
    test_text = test_text[:20]
    test_labels = test_labels[:20]

    eval_text, eval_labels = loading(args.eval_file, args.normalized)
    eval_text = eval_text[:20]
    eval_labels = eval_labels[:20]

    for epoch_indx in range(args.epoch):
        prev_loss = curr_loss
        epoch_loss = 0
        acc_epoch_record = []
        loss_epoch_record = []
        train_text, train_labels = randomize(train_text, train_labels)
        for batch_indx in tqdm(range(0, len(train_labels), args.batch_size),desc=f"TRAINING DATASET: {epoch_indx + 1}/{args.epoch}"):
            batch_encoding = tokenizer.batch_encode_plus(
                train_text[batch_indx:batch_indx + args.batch_size], padding='max_length', truncation=True,
                max_length=args.max_length, return_tensors='pt')
            input_ids = batch_encoding['input_ids']
            attention_mask = batch_encoding['attention_mask']
            predicted = model_object.forward(input_ids, attention_mask, args.drop_out)
            gold_label_tensor = torch.tensor(train_labels[batch_indx:batch_indx + args.batch_size])
            preds = torch.argmax(predicted, dim=1)
            accuracy = calculate_accuracy(gold_label_tensor, preds)
            acc_epoch_record.append(accuracy)
            loss_value = loss(predicted, gold_label_tensor)
            if args.L1_reg:
                for param in model.parameters():
                    loss_value += torch.sum(torch.abs(param)) * args.L1_lambda
            loss_epoch_record.append(loss_value.item())
            epoch_loss += loss_value.item()
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Ave TRAIN acc: Epoch {epoch_indx + 1}: {sum(acc_epoch_record) / len(acc_epoch_record)}")
        print(f"Ave TRAIN loss: Epoch {epoch_indx + 1}: {sum(loss_epoch_record) / len(loss_epoch_record)}")
        loss_records.append(epoch_loss)
        eval_text, eval_labels = randomize(eval_text, eval_labels)
        acc, ave_loss, _,_ = evaluation(model_object, eval_text, eval_labels)
        print(f"Ave DEV acc: Epoch {epoch_indx + 1}: {acc}")
        print(f"Ave DEV loss: Epoch {epoch_indx + 1}: {ave_loss}")
        curr_loss = ave_loss
        if curr_loss >= prev_loss:
            patience += 1
            if patience > 1:
                acc, ave_loss,total_predicted_label, total_gold_label = evaluation(model_object, test_text, test_labels)
                print(f"Ave TEST acc: {acc}")
                print(f"Ave TEST loss: Epoch {epoch_indx + 1}: {ave_loss}")
                matrices(total_gold_label, total_predicted_label)
                return
        else:
            patience = 0
    acc, ave_loss, total_predicted_label, total_gold_label = evaluation(model_object, test_text, test_labels )
    print(f"Ave TEST acc: {acc}")
    print(f"Ave TEST loss: Epoch {epoch_indx + 1}: {ave_loss}")
    matrices(total_gold_label, total_predicted_label)

def evaluation(model,text,label):
    ave_loss_epoch = []
    total_predicted_label = []
    total_gold_label = []
    with torch.no_grad():
        for batch_indx in range(0, len(label), args.batch_size):
            batch_encoding = tokenizer.batch_encode_plus(
                text[batch_indx:batch_indx + args.batch_size], padding='max_length',max_length=args.max_length, truncation=True, return_tensors='pt', text_pair=True)

            input_ids = batch_encoding['input_ids']
            attention_mask = batch_encoding['attention_mask']
            predicted = model.forward(input_ids, attention_mask, False)

            gold_label_list = label[batch_indx:batch_indx + args.batch_size]
            total_gold_label.extend(gold_label_list)
            gold_label_tensor = torch.tensor(gold_label_list)
            loss_value = loss(predicted, gold_label_tensor)
            ave_loss_epoch.append(loss_value)
            preds = torch.argmax(predicted, dim=1)
            total_predicted_label.extend(preds)
            accuracy = calculate_accuracy(gold_label_tensor, preds)
        return accuracy, sum(ave_loss_epoch)/len(ave_loss_epoch), total_predicted_label, total_gold_label

def plotting(records):
    batchList = [i for i in range(len(records))]
    plt.plot(batchList, records, linewidth=5, label="Loss variation")
    plt.xlabel("Batch", color="green", size=20)
    plt.ylabel("Loss", color="green", size=20)
    plt.title("Progress Line for BERT Model", size=20)
    plt.grid()
    plt.show()

def randomize(list1, list2):
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    shuffled_list1, shuffled_list2 = zip(*combined)
    return shuffled_list1,shuffled_list2

def calculate_accuracy(gold_label, predicted):
    correct = torch.sum(gold_label== predicted).item()
    total = len(gold_label)
    accuracy = (correct / total) * 100
    return accuracy

def matrices(gold, predicted):
    predicted = [tensor.cpu() for tensor in predicted]
    results = classification_report(gold, predicted)
    print(results)
    print()
    cm = confusion_matrix(gold, predicted)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    train_classifier()
    plotting(loss_records)
    seconds = time.time() - start_time
    print('Time Taken:', time.strftime("%H:%M:%S", time.gmtime(seconds)))

