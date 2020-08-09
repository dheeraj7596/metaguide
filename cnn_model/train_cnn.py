import os
import sys
import torch.autograd as autograd
import torch.nn.functional as F
import torch
import torchtext.data as data
import torchtext.datasets as datasets
from sklearn.model_selection import train_test_split
from cnn_model.model import CNN_Text
from cnn_model.dataset import TrainValFullDataset


def train(train_iter, dev_iter, model, use_gpu, lr, num_epochs, early_stop=5, log_interval=100):
    if use_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    steps = 0
    best_epoch = 0
    best_model = None
    best_loss = float("inf")
    model.train()
    for epoch in range(1, num_epochs + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.t_()  # batch first
            if use_gpu:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             batch.batch_size))

        dev_loss = eval(dev_iter, model, use_gpu)
        if dev_loss <= best_loss:
            best_loss = dev_loss
            best_epoch = epoch
            best_model = model
        else:
            if epoch - best_epoch >= early_stop:
                print('early stop by {} epochs.'.format(early_stop))
                print("Best epoch: ", best_epoch, "Current epoch: ", epoch)
                break
    return best_model


def train_cnn(X, y, X_full, y_full, use_gpu):
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False,
                             use_vocab=False,
                             pad_token=None,
                             unk_token=None
                             )

    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.1)
    train_data, val_data, full_data = TrainValFullDataset.splits(text_field, label_field, train_X, train_y, val_X,
                                                                 val_y, X_full, y_full)

    text_field.build_vocab(train_data, val_data, full_data)
    label_field.build_vocab(train_data, val_data, full_data)

    train_iter, dev_iter, full_data_iter = data.BucketIterator.splits((train_data, val_data, full_data),
                                                                      batch_sizes=(256, 256, 256), sort=False,
                                                                      sort_within_batch=False)
    embed_num = len(text_field.vocab)
    class_num = len(label_field.vocab)
    kernel_sizes = [3, 4, 5]
    cnn = CNN_Text(
        embed_num=embed_num,
        class_num=class_num,
        kernel_sizes=kernel_sizes)
    if use_gpu:
        cnn = cnn.cuda()
    model = train(train_iter, dev_iter, cnn, use_gpu, lr=0.001, num_epochs=256)
    pred_labels, pred_probs, true_labels = test_eval(full_data_iter, model, use_gpu)
    return pred_labels, pred_probs, true_labels


def eval(data_iter, model, use_gpu):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_()  # batch first
        if use_gpu:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return avg_loss


def test_eval(data_iter, model, use_gpu):
    model.eval()
    pred_labels = []
    true_labels = []
    total_probs = []
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_()  # batch first
        if use_gpu:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        probs = F.softmax(logit, dim=-1)
        pred_labels.append(torch.max(logit, 1)[1].view(target.size()).data)
        total_probs.append(probs)
        true_labels.append(target.data)

    pred_probs = torch.cat(total_probs).contiguous().detach().cpu().numpy()
    pred_labels = torch.cat(pred_labels).contiguous().detach().cpu().numpy()
    true_labels = torch.cat(true_labels).contiguous().detach().cpu().numpy()
    return pred_labels, pred_probs, true_labels


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item() + 1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    X = ["As kids, we lived together",
         "we fnust20, we laughed, we cried",
         "we did not always show the love",
         "As kids, we lived together",
         "we fnust20, we laughed, we cried",
         "we did not always show the love",
         "As kids, we lived together",
         "we fnust20, we laughed, we cried",
         "we did not always show the love",
         "As kids, we lived together",
         "we fnust20, we laughed, we cried",
         "we did not always show the love"
         ]
    y = [0, 2, 0, 4, 5, 6, 0, 4, 5, 3, 2, 1]
    X_full = ["As kids, we lived together",
              "we fnust20, we laughed, we cried",
              "we did not always show the love",
              "As kids, we lived together",
              "we fnust20, we laughed, we cried",
              "we did not always show the love",
              "As kids, we lived together",
              "we fnust20, we laughed, we cried",
              "we did not always show the love",
              "As kids, we lived together",
              "we fnust20, we laughed, we cried",
              "we did not always show the love"
              ]
    y_full = [0, 2, 0, 4, 5, 6, 0, 4, 5, 3, 2, 1]
    use_gpu = False
    save_dir = "./data/"
    train_cnn(X, y, X_full, y_full, use_gpu)
