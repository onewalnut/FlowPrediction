# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     main
   Description :
   Author :       walnut
   date:          2021/1/5
-------------------------------------------------
   Change Activity:
                  2021/1/5:
-------------------------------------------------
"""
__author__ = 'walnut'

from classes import *
from paras import *


def flow_predict():
    # device configuration
    device = torch.device("cuda:" + GPU_ID if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = True

    train_data = SeqDataset(train=True, file=DATA_FILE)
    test_data = SeqDataset(train=False, file=DATA_FILE)

    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=TRAIN_BATCH, num_workers=4)
    test_loader = DataLoader(dataset=test_data, shuffle=False, batch_size=TEST_BATCH, num_workers=4)

    model = TemporalNet()
    model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=GPU_IDS)
    print(model)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=2e-4)

    # record train loss during iterative
    train_process_record = []

    save_flag = SAVE_FLAG
    best_res = None

    for epoch in range(1, EPOCH + 1):
        print("Epoch {}: (with LR = {}):".format(epoch, optimizer.param_groups[0]['lr']))

        train_loss, train_GT, train_PRED = train(epoch=epoch, model=model, train_loader=train_loader, optimizer=optimizer, loss_func=loss_func, device=device)

        test_loss, input_seq, test_GT, test_PRED = evaluate(model=model, test_loader=test_loader, loss_func=loss_func, device=device)

        train_process_record.append([epoch, train_loss, test_loss])

        if test_loss < save_flag:
            # best_res = np.concatenate((test_GT, test_PRED), axis=1)
            best_res = np.concatenate((np.expand_dims(test_GT, axis=1), np.expand_dims(test_PRED, axis=1)), axis=1)
            best_res = np.concatenate((input_seq, best_res), axis=1)
            # best_res = np.concatenate((np.squeeze(input_seq, axis=2), best_res), axis=1)
            # best_res = np.concatenate((np.expand_dims(test_GT, axis=1), np.expand_dims(test_PRED, axis=1)), axis=1)
            save_flag = test_loss

            save_res(file=RES_PATH + "res.csv", res=best_res)
            torch.save(model, RES_PATH + "net.pkl")

        if epoch % DECAY_PERIOD == 0:
            adjust_learning_rate(optimizer, LR_DECAY)

        save_res(RES_PATH + "train_process.csv", train_process_record)

    if best_res is None:
        print("all result below save flag!")



def train(epoch, model, train_loader, optimizer, loss_func, device):
    model.train()

    # record training data
    train_losses = []
    ground_truth = []
    prediction = []

    for step, (seq, label) in enumerate(train_loader):
        seq = seq.type(torch.FloatTensor).to(device)
        label = label.type(torch.FloatTensor).to(device)

        output = model(seq)
        loss = loss_func(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.data.item())
        ground_truth.extend(label.cpu().detach().numpy().tolist())
        prediction.extend(output.cpu().detach().numpy().tolist())

        print("\t{}-{}\t\tTrain_loss: {:.4f}".format(epoch, step, loss.data.item()))
    print("\tAverage train loss: {:.4f}".format(np.average(train_losses)))

    return np.average(train_losses), ground_truth, prediction


def evaluate(model, test_loader, loss_func, device, model_idx=None):
    model.eval()

    test_losses = []
    ground_truth = []
    prediction = []
    input_seq = []

    for step, (seq, label) in enumerate(test_loader):
        seq = seq.type(torch.FloatTensor).to(device)
        label = label.type(torch.FloatTensor).to(device)

        output = model(seq)
        loss = loss_func(output, label)

        test_losses.append(loss.data.item())
        input_seq.extend(seq.cpu().detach().numpy().tolist())
        ground_truth.extend(label.cpu().detach().numpy().tolist())
        prediction.extend(output.cpu().detach().numpy().tolist())

    print("\tAverage test loss:{:.4f}".format(np.average(test_losses)))

    return np.average(test_losses), input_seq, ground_truth, prediction


def save_res(file, res):
    with open(file, 'w', newline='') as csv_file:
        my_writer = csv.writer(csv_file)
        my_writer.writerows(res)


if __name__ == "__main__":
    flow_predict()

    pass
