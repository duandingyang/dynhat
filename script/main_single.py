import os
import sys
import time
from collections import OrderedDict
from math import isnan

import numpy as np
import torch
import torch.optim as optim
from models.Dynhat import Dynhat
from script.config import args
from script.inits import prepare
from script.loss import ReconLoss
from script.utils.data_util import loader, prepare_dir
from script.utils.runbuilder import RunBuilder
from script.utils.runManager import RunManager
from script.utils.util import disease_path, init_logger, logger, set_random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class Runner(object):
    def __init__(self):
        self.len = data['time_length']
        self.start_train = 0
        self.train_shots = list(range(0, self.len - args.testlength))
        self.test_shots = list(range(self.len - args.testlength, self.len))
        self.x = torch.eye(args.num_nodes).to(args.device) # one-hot encodding
        args.nfeat = self.x.size(1)
        self.model = Dynhat(args, len(self.train_shots)).to(args.device)
        self.loss = ReconLoss(args)

    def run(self, run_manager):
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        t_total0 = time.time()
        test_results, min_loss = [0] * 5, 100000
        best_epoch_auc = 0
        patient_auc = 0
        self.model.train()
        for epoch in range(1, args.max_epoch + 1):
            run_manager.begin_epoch()
            t0 = time.time()
            self.model.train()
            structural_out = []
            for t in self.train_shots:
                edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data, t)
                z = self.model(edge_index, self.x) # shape: [num_node, nout]
                structural_out.append(z)

            structural_outputs = [g[:,None,:] for g in structural_out]
            maximum_node_num = structural_outputs[-1].shape[0]  
            out_dim = structural_outputs[-1].shape[-1]  
            structural_outputs_padded = []
            for out in structural_outputs:  
                zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)  # padding
                padded = torch.cat((out, zero_padding), dim=0)  
                structural_outputs_padded.append(padded)
            structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]; 
            temporal_out = self.model.ddy_attention_layer(structural_outputs_padded)
            temporal_output = []
            for t in range(0, len(self.train_shots)):
                x = temporal_out[:, t, :]
                x = self.model.toHyperX(x, 1)
                temporal_output.append(x)
            temporal_outputs = [g[:, None, :] for g in temporal_output]
            final_outputs = torch.cat(temporal_outputs, dim=1)
            optimizer.zero_grad()
            epoch_losses = []
            epoch_loss = 0
            for t in self.train_shots:
                edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data, t)
                z = final_outputs[:, t, :].squeeze()
                epoch_loss += self.loss(z, edge_index)
                epoch_losses.append(epoch_loss.item())
            epoch_loss.backward()
            optimizer.step()
            self.model.eval()
            average_epoch_loss = np.mean(epoch_losses)
            if average_epoch_loss < min_loss:
                min_loss = average_epoch_loss
                test_results = self.test(epoch, z)
                patience = 0
            else:
                patience += 1
                if epoch > args.min_epoch and patience > args.patience:
                    print('loss early stopping')
                    break
            
            if test_results[1] > best_epoch_auc:
                best_epoch_auc = test_results[1]
                patient_auc = 0
            else:
                patient_auc += 1
                if patient_auc > args.patience:
                    print('auc early stopping')
                    break
         
            logger.info("Epoch:{:}, Test AUC: {:.4f}, AP: {:.4f}, New AUC: {:.4f}, New AP: {:.4f}".format(test_results[0], test_results[1], test_results[2], test_results[3],test_results[4]))

            if isnan(epoch_loss):
                print('nan loss')
                break
            run_manager.trace_loss('{:.4f}'.format(average_epoch_loss))
            run_manager.trace_auc('{:.4f}'.format(test_results[1]))
            run_manager.trace_ap('{:.4f}'.format(test_results[2]))
            run_manager.trace_new_auc('{:.4f}'.format(test_results[3]))
            run_manager.trace_new_ap('{:.4f}'.format(test_results[4]))

            run_manager.end_epoch()

    def test(self, epoch, embeddings=None):
        auc_list, ap_list = [], []
        auc_new_list, ap_new_list = [], []
        embeddings = embeddings.detach()
        for t in self.test_shots:
            edge_index, pos_edge, neg_edge = prepare(data, t)[:3]
            new_pos_edge, new_neg_edge = prepare(data, t)[-2:]
            auc, ap = self.loss.predict(embeddings, pos_edge, neg_edge)
            auc_new, ap_new = self.loss.predict(embeddings, new_pos_edge, new_neg_edge)
            auc_list.append(auc)
            ap_list.append(ap)
            auc_new_list.append(auc_new)
            ap_new_list.append(ap_new)
        if epoch % args.log_interval == 0:
            logger.info(
                'Epoch:{}, average AUC: {:.4f}; average AP: {:.4f}'.format(epoch, np.mean(auc_list), np.mean(ap_list)))
            logger.info('Epoch:{}, average AUC new: {:.4f}; average AP new: {:.4f}'.format(epoch, np.mean(auc_new_list),
                                                                                   np.mean(ap_new_list)))
        return epoch, np.mean(auc_list), np.mean(ap_list), np.mean(auc_new_list), np.mean(ap_new_list)


if __name__ == '__main__':
    params = OrderedDict(
        nhid=[args.nhid],
        nout=[args.nout],
        temporal_attention_layer_heads=[args.temporal_attention_layer_heads],
        heads=[args.heads],
        dataset=[args.dataset],
        split_count=[args.split_count]
    )

    run_manager = RunManager()
    for run in RunBuilder.get_runs(params):
        run_manager.begin_run(run)

        args.nhid = run.nhid
        args.nout = run.nout
        args.temporal_attention_layer_heads = run.temporal_attention_layer_heads
        args.heads = run.heads
        args.dataset = run.dataset

        args.split_count = f'{run.split_count}-split'  

        args.output_folder = '../data/output/log/{}/{}/{}/'.format(args.dataset, args.model, args.split_count)

        data = loader(dataset=args.dataset, split_count=args.split_count)
        args.num_nodes = data['num_nodes']
        set_random(args.seed)
        init_logger(prepare_dir(args.output_folder) + args.dataset + '.txt')
        try:
            runner = Runner()
            runner.run(run_manager)
            run_manager.save(args.output_folder, args.split_count, args.heads, args.temporal_attention_layer_heads, args.nout, args.nhid)
        except RuntimeError:
            logger.info(f'current args except runtime error')
        logger.info(f'current args: {args}')
        #run_manager.save('../data/output/log/{}/{}/'.format(args.dataset, args.model))