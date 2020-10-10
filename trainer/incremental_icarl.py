import torch
import tqdm
import numpy as np
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *
from utils.process_fp import process_inputs_fp
import torch.nn.functional as F

def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y


def incremental_train_and_eval(the_args, epochs, fusion_vars, ref_fusion_vars, b1_model, ref_model, b2_model, ref_b2_model, tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler, trainloader, testloader, iteration, start_iteration, X_protoset_cumuls, Y_protoset_cumuls, order_list,lamda, dist, K, lw_mr, fix_bn=False, weight_per_class=None, device=None):

    T = 2.0
    beta = 0.25
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_model.eval()

    num_old_classes = ref_model.fc.out_features
    if iteration > start_iteration+1:
        ref_b2_model.eval()

    for epoch in range(epochs):
        b1_model.train()
        b2_model.train()

        if fix_bn:
            for m in b1_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        correct = 0
        total = 0

        tg_lr_scheduler.step()
        fusion_lr_scheduler.step()

        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs, targets = inputs.to(device), targets.to(device)

            tg_optimizer.zero_grad()

            outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)

            if iteration == start_iteration+1:
                ref_outputs = ref_model(inputs)
            else:
                ref_outputs, ref_features_new = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, inputs)
            loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:,:num_old_classes]/T, dim=1), \
                F.softmax(ref_outputs.detach()/T, dim=1)) * T * T * beta * num_old_classes
            loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            loss = loss1 + loss2

            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Train set: {}, train loss1: {:.4f}, train loss2: {:.4f}, train loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_loss/(batch_idx+1), 100.*correct/total))
        
        b1_model.eval()
        b2_model.eval()
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            fusion_optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            loss.backward()
            fusion_optimizer.step()
        print('Fusion index:    mtl1, free1, mtl2, free2, mtl3, free3')
        print('Fusion vars:     {:.2f}, {:.2f},  {:.2f}, {:.2f},  {:.2f}, {:.2f}'.format(float(fusion_vars[0]), 1.0-float(fusion_vars[0]), float(fusion_vars[1]), 1.0-float(fusion_vars[1]), float(fusion_vars[2]), 1.0-float(fusion_vars[2])))
        print('Ref fusion vars: {:.2f}, {:.2f},  {:.2f}, {:.2f},  {:.2f}, {:.2f}'.format(float(ref_fusion_vars[0]), 1.0-float(ref_fusion_vars[0]), float(ref_fusion_vars[1]), 1.0-float(ref_fusion_vars[1]), float(ref_fusion_vars[2]), 1.0-float(ref_fusion_vars[2])))

        b1_model.eval()
        b2_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    print("Removing register forward hook")
    return b1_model, b2_model

def incremental_train_and_eval_first_phase(the_args, epochs, b1_model, ref_model, \
    tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iteration, \
    lamda, dist, K, lw_mr, fix_bn=False, weight_per_class=None, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        b1_model.train()

        if fix_bn:
            for m in b1_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        correct = 0
        total = 0

        tg_lr_scheduler.step()

        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            outputs = b1_model(inputs)
            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            loss.backward()
            tg_optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Train set: {}, train loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))

        b1_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = b1_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    return b1_model

