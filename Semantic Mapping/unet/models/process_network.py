import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import itertools

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from PIL import Image
from label_visualize import *

#np.set_printoptions(threshold=np.inf)


# tensor to numpy
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))


    return img


# numpy to tensor
def np_to_tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))


    return img.float().div(255).unsqueeze(0)


# gaussian blurring function
def Gaussian_blur(in_img, kernel_size=(3, 3)):
    '''
    row_out = (int)((in_img.shape[0]) / 2)
    col_out = (int)((in_img.shape[1]) / 2)
    color_out = (int)(in_img.shape[2])
    out_img = np.zeros((row_out, col_out, color_out))
    p_Gaussian_blurred = cv2.GaussianBlur(in_img, kernel_size, 1, 1)
    for row in range(row_out):
        for col in range(col_out):
            out_img[row][col] = p_Gaussian_blurred[row * 2 + 1][col * 2 + 1]
    '''
    out_img = cv2.GaussianBlur(in_img, kernel_size, 1, 1)


    return out_img


# plot the loss
def plot_loss(hist_losses):
    plt.figure()
    x = range(len(hist_losses))
    plt.plot(x, hist_losses, color='blue', label='hist_losses')
    plt.legend()

    plt.title('Losses v.s. Iteration')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


# show the training result
def show_result(model, x_, y_, num_epoch):
    predict_images = model(x_)

    fig, ax = plt.subplots(x_.size()[0], 3, figsize=(12, 10))

    for i in range(x_.size()[0]):
        ax[i, 0].get_xaxis().set_visible(False)
        ax[i, 0].get_yaxis().set_visible(False)
        ax[i, 1].get_xaxis().set_visible(False)
        ax[i, 1].get_yaxis().set_visible(False)
        ax[i, 2].get_xaxis().set_visible(False)
        ax[i, 2].get_yaxis().set_visible(False)
        ax[i, 0].cla()
        ax[i, 0].imshow(process_image(x_[i]))
        ax[i, 1].cla()
        ax[i, 1].imshow(process_image(predict_images[i]))
        ax[i, 2].cla()
        ax[i, 2].imshow(process_image(y_[i]))

    plt.tight_layout()
    label_epoch = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0, label_epoch, ha='center')
    label_input = 'Input'
    fig.text(0.18, 1, label_input, ha='center')
    label_output = 'Output'
    fig.text(0.5, 1, label_output, ha='center')
    label_truth = 'Truth'
    fig.text(0.81, 1, label_truth, ha='center')

    plt.show()


# save the training results
def save_result(model, x_, y_, num_epoch):
    predict_images = model(x_)

    fig, ax = plt.subplots(x_.size()[0], 3, figsize=(12, 10)) # figsize ratio (w,h) (18, 15)--->(5)

    for i in range(x_.size()[0]):
        ax[i, 0].get_xaxis().set_visible(False)
        ax[i, 0].get_yaxis().set_visible(False)
        ax[i, 1].get_xaxis().set_visible(False)
        ax[i, 1].get_yaxis().set_visible(False)
        ax[i, 2].get_xaxis().set_visible(False)
        ax[i, 2].get_yaxis().set_visible(False)
        ax[i, 0].cla()
        ax[i, 0].imshow(process_image(x_[i]))
        ax[i, 1].cla()
        ax[i, 1].imshow(process_image(predict_images[i]))
        ax[i, 2].cla()
        ax[i, 2].imshow(process_image(y_[i]))

    plt.tight_layout()
    label_epoch = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0, label_epoch, ha='center')
    label_input = 'Input'
    fig.text(0.18, 1, label_input, ha='center')
    label_output = 'Output'
    fig.text(0.5, 1, label_output, ha='center')
    label_truth = 'Truth'
    fig.text(0.81, 1, label_truth, ha='center')

    plt.savefig(
        '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Training_results/Epoch %d.png' %
        (num_epoch), bbox_inches='tight'
    )#, bbox_inches='tight')

    plt.close()


# count parameters
def count_params(model):
    num_params = sum([item.numel() for item in model.parameters() if item.requires_grad])


    return num_params


# compute IOU
iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

def iou(img_true, img_pred):
    img_pred = (img_pred > 0).float()
    i = (img_true * img_pred).sum()
    u = (img_true + img_pred).sum()
    return i / u if u != 0 else u

def iou_metric(imgs_pred, imgs_true):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i] = (iou_thresholds <= iou(imgs_true[i], imgs_pred[i])).mean()
    return scores.mean()


# Dice loss
# Dice loss helper function
def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)


    return (2. * intersection + 100) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + 100)


# Dice loss caller
def dice_coef_loss(y_true, y_pred):
    return (-1)*dice_coef_np(y_true, y_pred)


# transform training datasets' H reduced img(0-420) to (512, 1024)
def crop_tensor_train(x, y):
    cropped_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    batch_size = y.shape[0]
    x_temp, y_temp = x[:, :, :420, :], y[:, :, :420, :]
    x0, y0 = cropped_transform(x_temp[0]), cropped_transform(y_temp[0])
    x0 = x0.unsqueeze(0)

    x_, y_ = x0.cuda(), y0.cuda()


    return (x_, y_)


'''
transfer the labelID image to proper format
'''
def pyr_down(p, kernel_size=3, sigma=0.8):
    '''
    Downsample the pyramid image to get the upper level.
    Input:
      p: M x N x C array
    Return:
      out: M/2 x N/2 x C
    '''
    p_ = p.copy()
    # out = gaussian_blur(p_, kernel_size, sigma)


    return cv2.resize(p_, (int(p.shape[1] / 2), int(p.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)


#input：labels  channel：1024*2048
#input：one_hot  channel：N*34*512*1024
def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   #reshape to vector
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)  #transform to one hot
    size.append(N)  #add label to the last col and reshape to origin size
    return ones.view(*size)


def ToOneShot(labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    labels = labels.cpu().numpy()
    labels = labels[0][0]
    #print(labels.shape)
    #print(np.max(labels))
    a = pyr_down(labels)  #sample to 512*1024
    label_true = (a*255).astype(int)  #convert to intergal within 0-33
    class_num_real = int(np.max(label_true))+1  #real label numbers
    #print('real:%d' % class_num_real)
    class_num = 34  #int(np.max(label_true))+1  #total categories
    H = label_true.shape[0]
    W = label_true.shape[1]
    L = torch.LongTensor(label_true).unsqueeze(0)  #convert to 1*512*1024 (the first size is batch_size)

    gt_one_hot = get_one_hot(L, class_num)  #the result is 1*512*1024*34
    #print(gt_one_hot)
    #print(gt_one_hot.shape)

    one_hot = gt_one_hot.permute(0,3,1,2)  #arrange label to the corresponding channel
    #print(one_hot.shape)


    return one_hot


# train process
def train_model(model, criterion, optimizer, train_label_loader, train_raw_loader, test_raw, test_label, num_epochs=20):
    start_time = time.time()
    # loss history
    hist_losses = []

    for epoch in range(num_epochs):
        print('Start training epoch %d' % (epoch + 1))
        losses_list = []
        epoch_start_time = time.time()
        num_iter = 0
        for (train_label, train_raw) in zip(train_label_loader, train_raw_loader):
            y, temp1 = train_label  #torch.Size([1, 3, 512, 1024])
            x, temp2 = train_raw  #torch.Size([1, 3, 512, 1024])
            #y = ToOneShot(y)  # transform the labels to one shot  #torch.Size([1, 34, 512, 1024])
            #print(y.shape)
            #print(torch.sum(torch.sum(y, axis=2), axis=2))

            x_, y_crop = crop_tensor_train(x, y)  #y_->(3, 512, 1024)
            #print(y_.shape)
            y_temp = y_crop[0].unsqueeze(0)
            #print(y_temp.shape)
            y_ = (y_temp * 255).long()
            #print(x_.shape)
            #print(y_.shape)
            #print(torch.max(y_))

            # Compute loss
            # forward
            outputs = model(x_)
            #print(preds)
            #print(preds.shape)
            loss = criterion(outputs, y_)
            # Bp and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_list.append(loss)
            hist_losses.append(loss)
            num_iter += 1

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - using time: %.2f' % ((epoch + 1), num_epochs, per_epoch_ptime))
        print('Loss: %.3f' % (torch.mean(torch.FloatTensor(losses_list))))

        # save test compare images
        '''
        if (epoch == 0) or (epoch % 5 == 0) or (epoch == (num_epochs - 1)):
            with torch.no_grad():
                save_result(model, test_raw.cuda(), test_label, (epoch + 1))
        '''
        with torch.no_grad():
            save_visualized_result(model, test_raw, test_label, (epoch + 1))

        # save train result
        torch.save(model.state_dict(), '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Final_Project/weights/weights_epoch%d.pth' % (epoch+1))

    end_time = time.time()
    total_ptime = end_time - start_time


    return (model, hist_losses)


# train the network
def train(model, train_label_loader, train_raw_loader, test_raw, test_label, num_epochs=20):
    # define LOSS functions
    criterion = nn.CrossEntropyLoss()  #nn.BCELoss().cuda()
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))  #lr = 1e-5  fair:0.0002 & (0.5, 0.999)
    print('Training start!')
    (model, hist_losses) = train_model(
        model, criterion, optimizer,
        train_label_loader, train_raw_loader, test_raw, test_label, num_epochs
    )


    return (model, hist_losses)


# keep training functions
# keep train process
def keep_train_model(model, criterion, optimizer, train_label_loader, train_raw_loader, test_raw, test_label, start_epoch, num_epochs=20):
    start_time = time.time()
    # loss history
    hist_losses = []

    for epoch in range(start_epoch-1, num_epochs):
        print('Start training epoch %d' % (epoch + 1))
        losses_list = []
        epoch_start_time = time.time()
        num_iter = 0
        for (train_label, train_raw) in zip(train_label_loader, train_raw_loader):
            y, temp1 = train_label  #torch.Size([1, 3, 512, 1024])
            x, temp2 = train_raw  #torch.Size([1, 3, 512, 1024])
            #y = ToOneShot(y)  # transform the labels to one shot  #torch.Size([1, 34, 512, 1024])
            #print(y.shape)
            #print(torch.sum(torch.sum(y, axis=2), axis=2))

            x_, y_crop = crop_tensor_train(x, y)  #y_->(3, 512, 1024)
            #print(y_.shape)
            y_temp = y_crop[0].unsqueeze(0)
            #print(y_temp.shape)
            y_ = (y_temp * 255).long()
            #print(x_.shape)
            #print(y_.shape)
            #print(torch.max(y_))

            # Compute loss
            # forward
            outputs = model(x_)
            #print(preds)
            #print(preds.shape)
            loss = criterion(outputs, y_)
            # Bp and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_list.append(loss)
            hist_losses.append(loss)
            num_iter += 1

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - using time: %.2f' % ((epoch + 1), num_epochs, per_epoch_ptime))
        print('Loss: %.3f' % (torch.mean(torch.FloatTensor(losses_list))))

        # save test compare images
        '''
        if (epoch == 0) or (epoch % 5 == 0) or (epoch == (num_epochs - 1)):
            with torch.no_grad():
                save_result(model, test_raw.cuda(), test_label, (epoch + 1))
        '''
        with torch.no_grad():
            save_visualized_result(model, test_raw, test_label, (epoch + 1))

        # save train result
        torch.save(model.state_dict(), '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Final_Project/weights/weights_epoch%d.pth' % (epoch+1))

    end_time = time.time()
    total_ptime = end_time - start_time


    return (model, hist_losses)


# train the network
def keep_train(model, train_label_loader, train_raw_loader, test_raw, test_label, start_epoch, num_epochs=20):
    # define LOSS functions
    criterion = nn.CrossEntropyLoss()  #nn.BCELoss().cuda()
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.5, 0.999))  #lr = 1e-4  fair:0.0002 & (0.5, 0.999)
    print('Keep training from epoch %d!' % (start_epoch))
    (model, hist_losses) = keep_train_model(
        model, criterion, optimizer,
        train_label_loader, train_raw_loader, test_raw, test_label, start_epoch, num_epochs
    )


    return (model, hist_losses)


# test the network  ??remaining to be writen
def test(model, test_label_loader, test_raw_loader):
    # select the best weights to apply
    model.load_state_dict(torch.load('weights_19.pth'))
    model.eval()
    plt.ion()

    with torch.no_grad():
        for (test_label, test_raw) in zip(test_label_loader, test_raw_loader):
            y_, temp1 = test_label
            x_, temp2 = test_raw
            x_, y_ = x_.cuda(), y_.cuda()

            show_result(model, x_, y_)