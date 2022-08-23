import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import copy 
import time
from utils import AverageMeter


class Trainer():
    def __init__(self, args, model, train_loader, test_loader, device, client_id):
        self.args = args
        self.exp = args.exp
        self.temperature = args.temperature
        self.local_epochs = args.local_ep
        self.device = device
        self.client_id = client_id
        self.model = model.to(device)
        
        self.optim_dict = {
            "sgd": torch.optim.SGD, 
            "adam": torch.optim.Adam
        }
        
        self.train_loader = train_loader
        self.test_loader = test_loader

        
        self.FL_criterion = nn.CrossEntropyLoss().to(self.device)
        self.SimCLR_criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.SimSiam_criterion = nn.CosineSimilarity(dim=-1).to(self.device)

        
    def simsiam_loss(self, p1, p2, z1, z2):
        loss = -(self.SimSiam_criterion(p1, z2).mean() + self.SimSiam_criterion(p2, z1).mean()) * 0.5
        return loss
    
    def nce_loss(self, features):
        # features = (local batch size * 2, out_dim) shape 
        feature1, feature2 = torch.tensor_split(features, 2, 0)
        # feature1, 2 = (local batch size, out_dim) shape
        feature1, feature2 = F.normalize(feature1, dim=1), F.normalize(feature2, dim=1)
        batch_size = feature1.shape[0]
        LARGE_NUM = 1e9
        
        # each example in feature1 (or 2) corresponds assigned to label in [0, batch_size) 
        labels = torch.arange(0, batch_size, device=self.device, dtype=torch.int64)
        masks = torch.eye(batch_size, device=self.device)
        
        
        logits_aa = torch.matmul(feature1, feature1.T) / self.temperature #similarity matrix 
        logits_aa = logits_aa - masks * LARGE_NUM
        
        logits_bb = torch.matmul(feature2, feature2.T) / self.temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        
        logits_ab = torch.matmul(feature1, feature2.T) / self.temperature
        logits_ba = torch.matmul(feature2, feature1.T) / self.temperature
        
        loss_a = self.SimCLR_criterion(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = self.SimCLR_criterion(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b
        return loss
    
    def train(self, q=None):
        # change to train mode (requires_grad = False for backbone if freeze=True)
        self.model.set_mode("train") 

        # copy original model for fedprox regularization
        global_model = copy.deepcopy(self.model)
        optimizer = self.optim_dict[self.args.optimizer](
            filter(lambda p: p.requires_grad, self.model.parameters()),
            self.args.lr
        )

        start = time.time()
        for epoch in range(self.local_epochs):
            # Metric
            running_loss = AverageMeter("loss")
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
            
                if self.exp == "FLSL" or self.exp == "centralized":
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    preds = self.model(images)
                    loss = self.FL_criterion(preds, labels)
                    
                elif self.exp == "simclr":
                    images = torch.cat(images, dim=0)
                    images = images.to(self.device, non_blocking=True)
                    features = self.model(images)
                    loss = self.nce_loss(features)
                
                elif self.exp == "simsiam":
                    images[0] = images[0].to(self.device, non_blocking=True)
                    images[1] = images[1].to(self.device, non_blocking=True)
                    p1, p2, z1, z2 = self.model(images[0], images[1]) 
                    loss = self.simsiam_loss(p1, p2, z1, z2)

                if self.args.agg == "fedavg":
                    pass
                
                elif self.args.agg == "fedprox":
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)

                    loss += (self.args.mu / 2) * proximal_term

                elif self.args.agg == "fedmatch":
                    pass

                loss.backward()
                optimizer.step()
                
                loss_value = loss.item()
                running_loss.update(loss_value)
            
            # Train metrics
            avg_loss = running_loss.get_result()
            running_loss.reset()

            
            lr = optimizer.param_groups[0]['lr']

            if (epoch + 1) % 5 == 0:
                # Finetune to test the client's performance
                _, test_loss, test_top1, test_top5 = self.test(finetune=True, epochs=1)
                
                end = time.time()
                time_taken = end-start
                start = end
                
                print(f"""Client {self.client_id} Epoch [{epoch+1}/{self.local_epochs}]:
                          learning rate : {lr:.6f}
                          test acc/top1 : {test_top1:.2f}%
                          test acc/top5 : {test_top5:.2f}%
                          test loss : {test_loss:.2f}
                          train loss : {avg_loss:.2f} 
                          time taken : {time_taken:.2f} """)

                
                # Return evaluation metrics and original model parameters
                state_dict = {
                    "loss": test_loss, 
                    "top1": test_top1, 
                    "top5": test_top5,
                    "model": copy.deepcopy(self.model.state_dict()), 
                    "train_loss" : avg_loss
                }
               

        print(f"Training complete best top1/top5: {test_top1:.2f}%/{test_top5:.2f}%")
        
        # Multiprocessing Queue
        if q != None:
            q.put(state_dict)
            
        return state_dict
        
    def test(self, finetune=False, epochs=1):
        print(f"Client {self.client_id} Linear evaluating {self.exp} model")
        eval_model = copy.deepcopy(self.model)
        eval_model.set_mode("linear")
        eval_model = eval_model.to(self.device)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, eval_model.parameters()),
            lr=0.001
        )
        
        N = len(self.test_loader)
        
        running_loss = AverageMeter("loss")
        running_top1 = AverageMeter("acc/top1")
        running_top5 = AverageMeter("acc/top5")
        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                if finetune:
                    if batch_idx < int(0.5 * N):
                        optimizer.zero_grad()

                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)

                        preds = eval_model(images)
                        loss = self.FL_criterion(preds, labels) # FL_criterion is standard CE

                        loss.backward()
                        optimizer.step()
                    # Testing
                    elif batch_idx >= int(0.5 * N):
                        with torch.no_grad():
                            images = images.to(self.device, non_blocking=True)
                            labels = labels.to(self.device, non_blocking=True)

                            preds = eval_model(images)
                            loss = self.FL_criterion(preds, labels)


                            loss_value = loss.item()

                            _, top1_preds = torch.max(preds.data, -1)
                            _, top5_preds = torch.topk(preds.data, k=5, dim=-1)

                            top1 = ((top1_preds == labels).sum().item() / labels.size(0)) * 100
                            top5 = 0
                            for label, pred in zip(labels, top5_preds):
                                if label in pred:
                                    top5 += 1

                            top5 /= labels.size(0)
                            top5 *= 100

                            running_loss.update(loss_value)
                            running_top1.update(top1)
                            running_top5.update(top5)
                else: # if no finetune, test only
                    with torch.no_grad():
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)

                        preds = eval_model(images)
                        loss = self.FL_criterion(preds, labels)


                        loss_value = loss.item()

                        _, top1_preds = torch.max(preds.data, -1)
                        _, top5_preds = torch.topk(preds.data, k=5, dim=-1)

                        top1 = ((top1_preds == labels).sum().item() / labels.size(0)) * 100
                        top5 = 0
                        for label, pred in zip(labels, top5_preds):
                            if label in pred:
                                top5 += 1

                        top5 /= labels.size(0)
                        top5 *= 100

                        running_loss.update(loss_value)
                        running_top1.update(top1)
                        running_top5.update(top5)
                        
        eval_model_state = copy.deepcopy(eval_model.state_dict())
        avg_loss = running_loss.get_result()
        avg_top1 = running_top1.get_result()
        avg_top5 = running_top5.get_result()
        return eval_model_state, avg_loss, avg_top1, avg_top5