#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
import wandb
from options import args_parser
from datetime import datetime

from trainers import Trainer
from models import ResNet_model
from utils import get_dataset, average_weights, exp_details, CheckpointManager

# import atexit

# def exit_handler():
#     from notify_run import Notify
#     notify = Notify()
#     notify.send("Done")

# atexit.register(exit_handler)

torch.multiprocessing.set_start_method('spawn', force=True)
mp.set_start_method('spawn', force=True)
if __name__ == '__main__':
    now = datetime.now().strftime('%Y%m%d_%H%M%S')

    args = args_parser()

    wandb_writer = wandb.init(
        reinit = True,
        name = now,
        project = "Fed", 
        save_code = True, 
        resume = "allow",
        id = now if args.wandb_tag == "" else args.wandb_tag
    )

    config = {
        "model": args.model, 
        "group_norm": args.group_norm,
        "exp": args.exp,
        "aug_strength": args.strength, 
        "out_dim": args.out_dim, 
        "freeze": args.freeze, 
        "pred_dim": args.pred_dim, 
        "temp": args.temperature,
        "num_users": int(args.num_users * args.frac), 
        "num_items": args.num_items, 
        "local_epoch": args.local_ep, 
        "local_bs": args.local_bs, 
        "lr": args.lr, 
        "dataset": args.dataset, 
        "optimizer": args.optimizer, 
    }
    wandb_writer.config.update(config)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    device = torch.device(f'cuda:{args.train_device}') if torch.cuda.is_available() else torch.device('cpu')
    # BUILD MODEL
    global_model = ResNet_model(args)
    
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.set_mode("train")
    
    # load dataset and user train indices
    train_dataset, test_dataset, user_train_idxs = get_dataset(args)

    # number of participating clients
    num_clients_part = int(args.frac * args.num_users)
    assert num_clients_part > 0
    
    # Save checkpoint of best model so far
    ckpt_manager = CheckpointManager(args.ckpt_criterion)

    # Training 
    for epoch in range(args.epochs):
        local_weights, local_losses, local_top1s, local_top5s = {}, {}, {}, {}
        local_train_losses = {}
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        # Select clients for training in this round
        part_users_ids = np.random.choice(range(args.num_users), num_clients_part, replace=False)
        
        processes = []
        q = mp.Queue()
        for i, client_id in enumerate(part_users_ids):
            trainset = Subset(
                train_dataset, 
                user_train_idxs[client_id]
            ) 

            client_model = copy.deepcopy(global_model)
       
            train_loader = DataLoader(
                trainset, 
                batch_size=args.local_bs, 
                shuffle=True, 
                num_workers=8, 
                pin_memory=True
            )

            test_loader  = DataLoader(
                test_dataset, 
                batch_size=args.local_bs, 
                shuffle=False, 
                num_workers=8, 
                pin_memory=True
            )
            
            gpu_count = torch.cuda.device_count()
            curr_device = torch.device(f"cuda:{i%gpu_count}")
            #print(curr_device)
            trainer = Trainer(
                args = args,
                model = client_model, 
                train_loader = train_loader,
                test_loader = test_loader,
                device = curr_device, #device, 
                client_id = client_id
            )

            if args.parallel:
                #mp.set_start_method('spawn')
                p = mp.Process(target = trainer.train, args=(q,))
                processes.append(p)

            else:
                summary = trainer.train()
                test_loss, test_top1, test_top5, model_state_dict, train_loss = summary["loss"], summary["top1"], summary["top5"], summary["model"], summary["train_loss"]
            
                local_weights[i] = model_state_dict
                local_losses[i] = test_loss
                local_top1s[i] = test_top1
                local_top5s[i] = test_top5
                local_train_losses[i] = train_loss

        if args.parallel:
            for proc in processes:
                proc.start()
            
            #for proc in processes:
            #    proc.join()
            #    print("process joined")
            #for proc in processes:
            #    proc.terminate()

            for i in range(num_clients_part):
                summary = q.get()
                print(f"{i} summary")
                test_loss, test_top1, test_top5, model_state_dict, train_loss = summary["loss"], summary["top1"], summary["top5"], summary["model"], summary["train_loss"]
            
                local_weights[i] = model_state_dict
                local_losses[i] = test_loss
                local_top1s[i] = test_top1
                local_top5s[i] = test_top5
                local_train_losses[i] = train_loss
        
        for i in range(num_clients_part):
            wandb_writer.log({
                f"train_loss_cli_{i}": local_train_losses[i], 
                f"test_loss_cli_{i}": local_losses[i], 
                f"top1_cli_{i}": local_top1s[i], 
                f"top5_cli_{i}": local_top5s[i] 
            })
        print(len(local_weights))
        # aggregate weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # test loader for linear eval 
        test_loader  = DataLoader(
            test_dataset, 
            batch_size=args.local_bs, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )

        server_model = Trainer(
            args = args,
            model = copy.deepcopy(global_model), 
            train_loader = None, 
            test_loader = test_loader,
            device = device, 
            client_id = -1
        )

        if args.exp != "FL":
            state_dict, loss_avg, top1_avg, top5_avg = server_model.test(
                finetune=args.finetune, 
                epochs=args.finetune_epoch
            )

            if args.finetune:
                missing_keys, unexpected_keys = global_model.load_state_dict(state_dict)
                print(f"missing keys {missing_keys}")
                print(f"unexp keys {unexpected_keys}")
                    
                    
        # FL일 경우 finetune 하지 않고 aggregate된 weight로만 성능 평가 
        else:
            state_dict, loss_avg, top1_avg, top5_avg = server_model.test(
                finetune=False, 
                epochs=1
            )
            
        print("#######################################################")
        print(f' \nAvg Validation Stats after {epoch+1} global rounds')
        print(f'Validation Loss     : {loss_avg:.2f}')
        print(f'Validation Accuracy : top1/top5 {top1_avg:.2f}%/{top5_avg:.2f}%\n')
        print("#######################################################")

        wandb_writer.log({
            "test_loss_server": loss_avg, 
            "top1_server": top1_avg,
            "top5_server": top5_avg 
        })
        
        ckpt_manager.save(loss_avg, top1_avg, state_dict, args.ckpt_path)

