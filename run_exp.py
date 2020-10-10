import os
def run_exp(dataset='cifar100', nb_cl_fg=50, nb_cl=2, fusion_lr=0.0, branch_mode='dual', branch_1='ss', branch_2='free', baseline='lucir', imgnet_backbone='resnet18', gpu=0, label='Exp_01'):

    random_seed=1993
    nb_protos = 20
    the_command = 'python3.6 main.py' \
        + ' --nb_cl_fg=' + str(nb_cl_fg) \
        + ' --nb_cl=' + str(nb_cl) \
        + ' --gpu=' + str(gpu) \
        + ' --random_seed=' + str(random_seed)  \
        + ' --fusion_lr=' + str(fusion_lr) \
        + ' --baseline=' + baseline \
        + ' --imgnet_backbone=' + imgnet_backbone \
        + ' --branch_mode=' + branch_mode \
        + ' --branch_1=' + branch_1 \
        + ' --branch_2=' + branch_2
        
    if dataset=='cifar100':
        the_command += ' --dataset=cifar100'
        the_command += ' --ckpt_dir_fg=' + ckpt_fg_cifar
    elif dataset=='imagenet_sub':
        the_command += ' --dataset=imagenet_sub'
        the_command += ' --data_dir=./data/seed_1993_subset_100_imagenet/data'
        the_command += ' --test_batch_size=50' 
        the_command += ' --epochs=90' 
        the_command +=  ' --num_workers=16' 
        the_command +=  ' --custom_weight_decay=1e-4' 
        the_command +=  ' --test_batch_size=50' 
    elif dataset=='imagenet':
        the_command += ' --dataset=imagenet'
        the_command += ' --data_dir=./data/imagenet/data'
        the_command += ' --test_batch_size=50' 
        the_command += ' --ckpt_dir_fg=' + ckpt_fg_imagenet_full
        the_command += ' --epochs=90'
        the_command += ' --num_classes=1000'
        the_command +=  ' --num_workers=16' 
        the_command +=  ' --custom_weight_decay=1e-4' 
        the_command +=  ' --test_batch_size=50' 
    else:
        raise ValueError('Please set correct dataset.')

    the_command += ' --ckpt_label=' + label
    os.system(the_command)
    import pdb
    pdb.set_trace()

run_exp(dataset='imagenet', nb_cl_fg=500, nb_cl=100, fusion_lr=0.00001, \
    branch_mode='dual', branch_1='ss', branch_2='fixed', baseline='icarl', \
    imgnet_backbone='resnet18', gpu=0, label='Exp_01')
