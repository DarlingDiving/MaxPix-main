class DefaultConfig(object):

    '# fill in your own directory'
    data_dir = './Daset/' 
    train_data_root = '/home/aiuser1/ssd_data/xiaodai/person'  
    val_data_root = '/home/aiuser1/ssd_data/xiaodai/val/biggan1'  
    test_data_root_cnn = '/home/aiuser1/ssd_data/xiaodai/Faces-HQ' 


    load_model = True
    load_model_path_2 = './Daset/checkpoints/'
    save_model = True
    save_model_path = data_dir+"checkpoints/"

    seed = 43
    batch_size =  4
    use_gpu = True
    gpu_id = '0'
    trainer = 'combine'
    num_workers = 4

    img_size = 299
    use_sam = False

    max_epoch = 100
    lr = 0.00005
    lr_decay = 0.96
    weight_decay = 0
    nlabels = 2
    dropout     = 0.0#0.3
    mid_loss_weight = 0.5
    train_noise = None
    test_noise  = None
    noise_scale = 0

opt = DefaultConfig()
