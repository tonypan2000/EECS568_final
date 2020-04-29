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
import matplotlib.image
from PIL import Image

# import self functions
import sys
sys.path.append('/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Final_Project/datasets')
sys.path.append('/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Final_Project/models')
#sys.path.append('/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Final_Project/models')
import preprocess
import network
import process_network


# resize kitti image (512, 1024, 3)->(375, 1242, 3)
def pyr_up(p):
    '''
    Upsample the image to get the upper level.
    Input:
      p: M x N x C array
    Return:
      out: 375 x 1242 x C
    '''
    p_ = p.copy()


    return cv2.resize(p_, (1242, 375), interpolation=cv2.INTER_NEAREST)


# transform kitti single image's H reduced img(0-420) to (512, 1024)
def crop_tensor_show_single(x):
    cropped_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    x_temp = x[:, :, :420, :]
    x0 = cropped_transform(x_temp[0])
    #x1 = cropped_transform(x_temp[1])
    #x2 = cropped_transform(x_temp[2])
    #x3 = cropped_transform(x_temp[3])
    #x4 = cropped_transform(x_temp[4])
    x_ = x0.unsqueeze(0)

    x_ = x_.cuda()


    return x_


# transform testing datasets' H reduced img(0-420) to (512, 1024)
def crop_tensor_test(x, y):
    cropped_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    batch_size = y.shape[0]
    x_temp, y_temp = x[:, :, :420, :], y[:, :, :420, :]
    x0, y0 = cropped_transform(x_temp[0]), cropped_transform(y_temp[0])
    x1, y1 = cropped_transform(x_temp[1]), cropped_transform(y_temp[1])
    x2, y2 = cropped_transform(x_temp[2]), cropped_transform(y_temp[2])
    x3, y3 = cropped_transform(x_temp[3]), cropped_transform(y_temp[3])
    x4, y4 = cropped_transform(x_temp[4]), cropped_transform(y_temp[4])
    x_ = torch.cat((x0.unsqueeze(0), x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)), 0)
    y_ = torch.cat((y0.unsqueeze(0), y1.unsqueeze(0), y2.unsqueeze(0), y3.unsqueeze(0), y4.unsqueeze(0)), 0)  #(5, 3, 512, 1024)

    x_, y_ = x_.cuda(), y_.cuda()


    return (x_, y_)


# transform testing datasets' H reduced img(0-420) to (512, 1024)  pixel2color
def crop_tensor_test_pixel2color(x, y):
    cropped_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    batch_size = y.shape[0]
    x_temp, y_temp = x[:, :, :420, :], y[:, :, :420, :]
    x0, y0 = cropped_transform(x_temp[0]), cropped_transform(y_temp[0])
    x1, y1 = cropped_transform(x_temp[1]), cropped_transform(y_temp[1])
    x2, y2 = cropped_transform(x_temp[2]), cropped_transform(y_temp[2])
    x3, y3 = cropped_transform(x_temp[3]), cropped_transform(y_temp[3])
    x4, y4 = cropped_transform(x_temp[4]), cropped_transform(y_temp[4])
    x_temp2 = torch.cat((x0.unsqueeze(0), x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)), 0)
    y_temp2 = torch.cat((y0.unsqueeze(0), y1.unsqueeze(0), y2.unsqueeze(0), y3.unsqueeze(0), y4.unsqueeze(0)), 0)  #(5, 3, 512, 1024)

    y_temp3 = y_temp2[:, 0, :, :]  #(5, 512, 1024)
    y_ = (y_temp3 * 255).long()

    x_, y_ = x_temp2.cuda(), y_.cuda()


    return (x_, y_)


# show images
def process_image(img):
    img = img.cpu().data.numpy().transpose(1, 2, 0)


    return img


# save the kitti prediction results
def show_visualized_result_single(model, x_, filename):
    predict_label = model(x_)
    (temp_max, preds) = torch.max(predict_label.data, 1)  #preds->(512, 1024)
    #print(preds.shape)
    preds_np = preds[0].cpu().data.numpy()
    #print(preds_np.shape)

    '''
    fig, ax = plt.subplots(1, 2, figsize=(12, 10)) # figsize ratio (w,h) (18, 15)--->(5)


    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[0].cla()
    ax[0].imshow(pyr_up(process_image(x_[0])))
    ax[1].cla()
    ax[1].imshow(pyr_up(visualize_prediction_kitti(preds[0])))

    plt.tight_layout()
    label_epoch = 'Image {0}'.format(filename)
    fig.text(0.5, 0, label_epoch, ha='center')
    label_input = 'Input'
    fig.text(0.18, 1, label_input, ha='center')
    label_output = 'Output'
    fig.text(0.5, 1, label_output, ha='center')

    plt.savefig(
        '/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Kitti_results_self' + '/' + filename,
        bbox_inches='tight'
    )

    plt.show()
    '''


    return (pyr_up(preds_np), pyr_up(visualize_prediction_kitti(preds[0])))


# save the training results
def save_visualized_result(model, x, y, num_epoch):
    (x_, y_) = crop_tensor_test(x, y)
    predict_label = model(x_)
    (temp_max, preds) = torch.max(predict_label.data, 1)  #preds->(5, 512, 1024)
    #print(preds.shape)

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
        ax[i, 1].imshow(visualize_prediction(preds[i]))
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


# visualize the results of categories  (512, 1024)->(512, 1024, 3)
def visualize_prediction(output_model):
    predict_img = output_model.cpu().data.numpy()
    #print(predict_img)

    color_array = np.array([
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [111, 74, 0],
        [81, 0, 81], [128, 64, 128], [244, 35, 232], [250, 170, 160], [230, 150, 140], [70, 70, 70],
        [102, 102, 156], [190, 153, 153], [180, 165, 180], [150, 100, 100], [150, 120, 90], [153, 153, 153],
        [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180],
        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90],
        [0, 0, 110], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ], dtype=np.uint8)

    visualization = np.zeros((512, 1024, 3), dtype=np.uint8)
    for h in range(predict_img.shape[0]):
        for w in range(predict_img.shape[1]):
            visualization[h][w] = color_array[(int)(predict_img[h][w])]
            #visualization[h][w][1] = color_array[predict_img[h][w]][1]
            #visualization[h][w][2] = color_array[predict_img[h][w]][2]


    return visualization


# visualize the labels  (512, 1024)->(512, 1024, 3)
def visualize_prediction_label(label):
    predict_img = process_image(label)
    #print(predict_img)

    color_array = np.array([
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [111, 74, 0],
        [81, 0, 81], [128, 64, 128], [244, 35, 232], [250, 170, 160], [230, 150, 140], [70, 70, 70],
        [102, 102, 156], [190, 153, 153], [180, 165, 180], [150, 100, 100], [150, 120, 90], [153, 153, 153],
        [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180],
        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90],
        [0, 0, 110], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ], dtype=np.uint8)

    visualization = np.zeros((512, 1024, 3), dtype=np.uint8)
    for h in range(predict_img.shape[0]):
        for w in range(predict_img.shape[1]):
            visualization[h][w] = color_array[int(predict_img[h][w])]


    return visualization


# visualize the results of kitti  (512, 1024)->(512, 1024, 3)
def visualize_prediction_kitti(output_model):
    predict_img = output_model.cpu().data.numpy()
    #print(predict_img)

    # road&goound->[128, 64, 128]; vegetation&terrain->[107, 142, 35]
    color_array = np.array([
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [111, 74, 0],
        [128, 64, 128], [128, 64, 128], [244, 35, 232], [250, 170, 160], [230, 150, 140], [70, 70, 70],
        [102, 102, 156], [190, 153, 153], [180, 165, 180], [150, 100, 100], [150, 120, 90], [153, 153, 153],
        [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [107, 142, 35], [70, 130, 180],
        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90],
        [0, 0, 110], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ], dtype=np.uint8)

    visualization = np.zeros((512, 1024, 3), dtype=np.uint8)
    for h in range(predict_img.shape[0]):
        for w in range(predict_img.shape[1]):
            visualization[h][w] = color_array[(int)(predict_img[h][w])]


    return visualization


# predict the input with trained unet - single image input
def predict(model, weights_file_path, dir_path, crop, test_raw, test_label, num_restrict):
    if crop == 0:
        '''
        # model.eval()
        model.load_state_dict(torch.load(weights_file_path))
        model = model.cuda()
        test_raw, test_label = crop_tensor_test(test_raw, test_label)

        with torch.no_grad():
            show_result(model, test_raw, test_label, 21)
        '''
        pass
    else:
        # transform the input image
        transform = transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        #model.eval()
        model.load_state_dict(torch.load(weights_file_path))
        model = model.cuda()
        img_count = 1
        with torch.no_grad():
            files = os.listdir(dir_path)
            files.sort()  #sort
            if num_restrict != 0:
                for filename in files:
                    if img_count < num_restrict+1:
                        #print(filename)
                        img_raw = Image.open(dir_path + "/" + filename)  #(1242, 375)
                        #print(img_raw.size)
                        img = transform(img_raw).unsqueeze(0)
                        (output_label, visualized_label) = show_visualized_result_single(model, img.cuda(), 68)
                        #outputs = model(img)
                        #output_label = process_image(outputs[0])
                        output_label = Image.fromarray(output_label.astype('uint8')).convert('RGB')
                        visualized_label = Image.fromarray(visualized_label.astype('uint8')).convert('RGB')
                        output_label.save('/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Kitti_prediction/label' + '/' + filename)
                        visualized_label.save('/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Kitti_prediction/label_RGB' + '/' + filename)
                        img_count += 1
                    else:
                        break
            else:
                for filename in files:
                    #print(filename)
                    img_raw = Image.open(dir_path + "/" + filename)  #(1242, 375)
                    #print(img_raw.size)
                    img = transform(img_raw).unsqueeze(0)
                    (output_label, visualized_label) = show_visualized_result_single(model, img.cuda(), 68)
                    #outputs = model(img)
                    #output_label = process_image(outputs[0])
                    output_label = Image.fromarray(output_label.astype('uint8')).convert('RGB')
                    visualized_label = Image.fromarray(visualized_label.astype('uint8')).convert('RGB')
                    output_label.save('/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Kitti_prediction/label' + '/' + filename)
                    visualized_label.save('/home/chendh/Pytorch_Projects/jupyter_notebook_files/EECS504_Files/Project/Kitti_prediction/label_RGB' + '/' + filename)