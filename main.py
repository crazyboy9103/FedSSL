#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
import wandb
from datetime import datetime
from torch import nn
import matplotlib.pyplot as plt

from options import args_parser
from trainers import Trainer
from models import ResNet_model
from utils import get_dataset, average_weights, CheckpointManager, feed_noise_to_models, get_length_gradients, get_bn_stats, get_tsne

mp.set_start_method('spawn', force=True)
if __name__ == '__main__':
    now = datetime.now().strftime('%Y%m%d_%H%M%S')

    args = args_parser()

    wandb_writer = wandb.init(
        reinit = True,
        name = now if args.wandb_tag == "" else args.wandb_tag,
        project = "Fed", 
        resume = "never",
        id = now
    )

    # config = {
    #     "model": args.model, 
    #     "group_norm": args.group_norm,
    #     "exp": args.exp,
    #     "aug_strength": args.strength, 
    #     "out_dim": args.out_dim, 
    #     "freeze": args.freeze, 
    #     "pred_dim": args.pred_dim, 
    #     "temp": args.temperature,
    #     "num_users": int(args.num_users * args.frac), 
    #     "num_items": args.num_items, 
    #     "local_epoch": args.local_ep, 
    #     "local_bs": args.local_bs, 
    #     "lr": args.lr, 
    #     "dataset": args.dataset, 
    #     "optimizer": args.optimizer,
    #     "agg": args.agg, 
    #     "mu(fedprox)": args.mu
    # }
    # wandb_writer.config.update(config)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Set the model to train and send it to device.
    global_model = ResNet_model(args)
    global_model.to(device)
    
    
    # Get dataset and train indices of the dataset for each user
    train_dataset, test_dataset, user_train_idxs, server_data_idx = get_dataset(args)

    # number of participating clients
    num_clients_part = int(args.frac * args.num_users)
    assert num_clients_part > 0
    
    # Save checkpoint of best model based on (top1 / loss) so far
    ckpt_manager = CheckpointManager(args.ckpt_criterion)

    # all bn stats
    all_bn_stats = {}

    # Training 
    for epoch in range(args.epochs):
        # run training on clients if exp is not centralized
        if args.exp != "centralized":
            # train at server with iid data
            # start with linear mode, setting bn_momentum default
            # global_model.set_mode("linear")
            # print(f'\n | Global Training Round : {epoch+1} |\n')
            # iid_server_data_idx = []
            # for _class, idxs in server_data_idx.items():
            #     rand_idxs = np.random.choice(idxs, args.server_num_items, replace=False)
            #     iid_server_data_idx.extend(rand_idxs.tolist())

            # server_set = Subset(
            #     train_dataset, 
            #     iid_server_data_idx
            # )

            # server_loader = DataLoader(
            #     server_set, 
            #     batch_size=args.server_batch_size, 
            #     shuffle=True, 
            #     num_workers=8, 
            #     pin_memory=True,
            #     drop_last = True
            # )

            # test_loader  = DataLoader(
            #     test_dataset, 
            #     batch_size=args.server_batch_size, 
            #     shuffle=False, 
            #     num_workers=8, 
            #     pin_memory=True, 
            #     drop_last = True
            # )
                    
            # server_trainer = Trainer(
            #     args = args,
            #     model = copy.deepcopy(global_model), 
            #     train_loader = server_loader,
            #     test_loader = test_loader,
            #     device = torch.device("cuda:0"),
            #     client_id = -1 #server
            # ) 

            # server_summary = server_trainer.server_train(epochs=1)
            # global_model.load_state_dict(server_summary["model"])
            # wandb_writer.log({
            #     "epoch": epoch,
            #     "train_loss_server": server_summary["train_loss"], 
            #     "test_loss_server": server_summary["loss"], 
            #     "top1_server": server_summary["top1"], 
            #     "top5_server": server_summary["top5"]
            # })

            # print(f'\n | Server Trained : train_loss {server_summary["train_loss"]} test_loss {server_summary["loss"]} top1 {server_summary["top1"]} |\n')

            # convert train mode and bn_momentum = args.bn_stat_momentum
            global_model.set_mode("train")

            
            

            local_weights, local_losses, local_top1s, local_top5s, local_train_losses = {}, {}, {}, {}, {}
            
            
            # Select clients for training in this round
            part_users_ids = np.random.choice(range(args.num_users), num_clients_part, replace=False)
            
            # multiprocessing queue
            processes = []
            q = mp.Queue()
            for i, client_id in enumerate(part_users_ids):
                trainset = Subset(
                    train_dataset, 
                    user_train_idxs[client_id]
                ) 

                train_loader = DataLoader(
                    trainset, 
                    batch_size=args.local_bs, 
                    shuffle=True, 
                    num_workers=8, 
                    pin_memory=True,
                    drop_last = True
                )

                test_loader  = DataLoader(
                    test_dataset, 
                    batch_size=args.local_bs, 
                    shuffle=False, 
                    num_workers=8, 
                    pin_memory=True, 
                    drop_last = True
                )
                
                curr_device = torch.device(f"cuda:{i % torch.cuda.device_count()}")
    
                trainer = Trainer(
                    args = args,
                    model = copy.deepcopy(global_model), 
                    train_loader = train_loader,
                    test_loader = test_loader,
                    device = curr_device,
                    client_id = client_id
                )

                if args.parallel:
                    p = mp.Process(target = trainer.train, args=(q,))
                    processes.append(p)

                else:
                    summary = trainer.train()
                
                    local_weights[i] = summary["model"]
                    local_losses[i] = summary["loss"]
                    local_top1s[i] = summary["top1"]
                    local_top5s[i] = summary["top5"]
                    local_train_losses[i] = summary["train_loss"]

            if args.parallel:
                for proc in processes:
                    proc.start()

                # q: mp.Queue is blocking object, thus no need to join
                for i in range(num_clients_part):
                    summary = q.get()               
                    local_weights[i] = summary["model"]
                    local_losses[i] = summary["loss"]
                    local_top1s[i] = summary["top1"]
                    local_top5s[i] = summary["top5"]
                    local_train_losses[i] = summary["train_loss"]
            
            for i in range(num_clients_part):
                wandb_writer.log({
                    "epoch": epoch,
                    f"train_loss_cli_{i}": local_train_losses[i], 
                    f"test_loss_cli_{i}": local_losses[i], 
                    f"top1_cli_{i}": local_top1s[i], 
                    f"top5_cli_{i}": local_top5s[i] 
                })

            # # random noise through each client's parameter 
            # # output representation pairwise similarity
            # embed_vectors = feed_noise_to_models(local_weights_copy = copy.deepcopy(local_weights), global_model_copy = copy.deepcopy(global_model), batch_size = args.local_bs)
            # sims = []
            # for i in range(len(embed_vectors)):
            #     for j in range(i+1, len(embed_vectors)):
            #         vector_i, vector_j = embed_vectors[i].unsqueeze(0), embed_vectors[j].unsqueeze(0)
            #         sims.append(nn.CosineSimilarity()(vector_i, vector_j).cpu().item())

            # with torch.no_grad():
            #     _, S, _ = torch.linalg.svd(embed_vectors)
            #     # first principal axis' explaind variance ratio
            #     exp_var = (S[0] * S[0]) / S.square().sum()
            #     exp_var = exp_var.cpu().item()

            # # compute mean of cosine similarities of embedding vectors
            # mu, var = np.mean(sims), np.var(sims)
            # wandb_writer.log({
            #     "epoch": epoch,
            #     "embed_vector_mu": mu, 
            #     "embed_vector_var": var, 
            #     "1st_axis_exp_var": exp_var
            # })

            # Get bn stats
            get_bn_stats(copy.deepcopy(local_weights), all_bn_stats)
            
            # if (epoch + 1) % 5 == 0:
            # Bn params: [weight, bias, running_mean, running_var]
            types = ["weight", "bias", "mean", "var"]

            for layer_idx, (layer_name, stat) in enumerate(all_bn_stats.items()):
                # mean, var = stat["mean"], stat["var"] 
                cos_mean, cos_var = stat["cos_mean"], stat["cos_var"]
                # tsne_mean = get_tsne(mean)
                # tsne_var = get_tsne(var)
                
                layer_number, layer_type = layer_idx // 4, layer_idx % 4
                layer_type = types[layer_type]

                # fig1 = plt.figure()
                # fig1.suptitle(f"t-SNE mean vectors at layer {layer_number}")
                # plt.ylabel("comp2")
                # plt.xlabel("comp1")
                # plt.scatter(x=tsne_mean[:, 0], y=tsne_mean[:, 1], c = range(epoch+1), vmin = 0, vmax = epoch, cmap="summer")
                # plt.colorbar()
                # fig1.tight_layout()


                # fig2 = plt.figure()
                # fig2.suptitle(f"t-SNE var vectors at layer {layer_number}")
                # plt.ylabel("comp2")
                # plt.xlabel("comp1")
                # plt.scatter(x=tsne_var[:, 0], y=tsne_var[:, 1], c = range(epoch+1), vmin = 0, vmax = epoch, cmap="summer")
                # plt.colorbar()
                # fig2.tight_layout()


                wandb_writer.log({
                    "epoch": epoch,
                    f"cos_mean_layer{layer_number}_{layer_type}": cos_mean[epoch],
                    f"cos_var_layer{layer_number}_{layer_type}": cos_var[epoch],
                    # f"tsne_mean_layer{layer_number}_{layer_type}": wandb.Image(fig1),
                    # f"tsne_var_layer{layer_number}_{layer_type}": wandb.Image(fig2),
                })
                
                # fig3 = plt.figure()
                # fig3.suptitle("average l2 distance from mean vector")
                # plt.ylabel("l2 mean")
                # plt.xlabel("epoch")
                # plt.scatter(x=range(epoch+1), y=l2_mean)
                # fig3.tight_layout()

                
                # fig4 = plt.figure()
                # fig4.suptitle("variance of l2 distances from mean vector")
                # plt.ylabel("l2 var")
                # plt.xlabel("epoch")
                # plt.scatter(x=range(epoch+1), y=l2_var)
                # fig4.tight_layout()


            # if args.agg == "fedprox":
            #     # for fedprox (l2 regularization)
            #     # 1. gradient contributions from each layer in each client (ratio of l2 distance in each layer w.r.t. sum of l2 distances)
            #     # 2. average over multiple clients by layer
            #     ratio_grads = get_length_gradients(local_weights_copy = copy.deepcopy(local_weights), global_model_copy = copy.deepcopy(global_model))
            #     fig, ax = plt.subplots()
            #     ax.plot(ratio_grads)
            #     ax.set_title("L2 gradient by layer")
            #     ax.set_ylabel("grad prop")
            #     ax.set_xlabel("layer index")
            #     fig.tight_layout()

            #     wandb_writer.log({
            #         f"grad_{epoch}": wandb.Image(fig)
            #     })
            # aggregate weights
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)

        # if centralized 
        else:
            # use the same amount of training items as FLSL in each round
            part_users_ids = np.random.choice(range(args.num_users), num_clients_part, replace=False)
            train_idxs = []
            for client_id in part_users_ids:
                train_idxs.extend(user_train_idxs[client_id])
            
            trainset = Subset(
                train_dataset, 
                train_idxs
            )

            train_loader = DataLoader(
                trainset, 
                batch_size=args.local_bs, 
                shuffle=True, 
                num_workers=8, 
                pin_memory=True, 
                drop_last = True
            )

            test_loader  = DataLoader(
                test_dataset, 
                batch_size=args.local_bs, 
                shuffle=False, 
                num_workers=8, 
                pin_memory=True, 
                drop_last = True
            )

            trainer = Trainer(
                args = args,
                model = copy.deepcopy(global_model), 
                train_loader = train_loader,
                test_loader = test_loader,
                device = torch.device("cuda:0"),
                client_id = client_id
            )

            state_dict = trainer.train()
            global_model.load_state_dict(state_dict["model"])

        # test loader for linear eval 
        test_loader  = DataLoader(
            test_dataset, 
            batch_size=args.local_bs, 
            shuffle=False,  # important! 
            num_workers=4, 
            pin_memory=True, 
            drop_last = True
        )

        server_model = Trainer(
            args = args,
            model = copy.deepcopy(global_model), 
            train_loader = None, 
            test_loader = test_loader,
            device = device, 
            client_id = -1
        )

        if args.exp not in ["centralized", "FLSL"]:
            state_dict, loss_avg, top1_avg, top5_avg = server_model.test(
                finetune=True, 
                epochs=args.finetune_epoch
            )

            global_model.load_state_dict(state_dict)
                    
        # FL일 경우 finetune 하지 않고 aggregate된 weight로만 성능 평가 
        else:
            _, loss_avg, top1_avg, top5_avg = server_model.test(
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
            "top5_server": top5_avg,
            "epoch": epoch
        })
        
        ckpt_manager.save(loss_avg, top1_avg, global_model.state_dict(), args.ckpt_path)

