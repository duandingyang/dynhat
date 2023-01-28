import os
import sys
import time
import torch
import numpy as np
from models.Dynhat import Dynhat
from math import isnan

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class Runner(object):
    def __init__(self):
        self.len = data['time_length']
        self.start_train = 0
        self.train_shots = list(range(0, self.len - args.testlength))
        self.test_shots = list(range(self.len - args.testlength, self.len))
        self.load_feature()
        self.model = Dynhat(args, len(self.train_shots)).to(args.device)
        self.loss = ReconLoss(args)
        logger.info('total length: {}, test length: {}'.format(self.len, args.testlength))
        logger.info('nhid: {}, nout: {}'.format(args.nhid, args.nout))
        logger.info('gat heads: {}, ddy_attention_layer_heads: {}'.format(args.heads, args.ddy_attention_layer_heads))


    def load_feature(self):
        if args.trainable_feat:
            self.x = None
            logger.info("using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            self.x = torch.eye(args.num_nodes).to(args.device)
            logger.info('using one-hot feature')
            args.nfeat = self.x.size(1)

    def optimizer(self):
        import geoopt
        optimizer = geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr,
                                                        weight_decay=args.weight_decay)
        return optimizer

    def run(self, run_manager):
        optimizer = self.optimizer()
        t_total0 = time.time()
        test_results, min_loss = [0] * 5, 100000
        best_epoch_auc = 0
        patient_auc = 0
        self.model.train()
        
        for epoch in range(1, args.max_epoch + 1):

            run_manager.begin_epoch()

            t0 = time.time()
            # train
            self.model.train()
            structural_out = []
            for t in self.train_shots:
                edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data, t)
                z = self.model(edge_index, self.x) # shape: [num_node, nout]
                structural_out.append(z)

            # =======start self-attention=========
            structural_outputs = [g[:,None,:] for g in structural_out]
            maximum_node_num = structural_outputs[-1].shape[0]  # 节点数量
            out_dim = structural_outputs[-1].shape[-1]  # 输出特征的数量
            structural_outputs_padded = []
            for out in structural_outputs:  # 对节点进行补0，使其为同一个维度
                zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)  # padding节点的数量; 保持一定的维度;
                padded = torch.cat((out, zero_padding), dim=0)  # 节点特征拼接
                structural_outputs_padded.append(padded)
            structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]; 16个时刻拼接在一起; structural最终输出的节点特征
            temporal_out = self.model.ddy_attention_layer(structural_outputs_padded)
            # 映射到到双曲空间
            temporal_output = []
            for t in range(0, len(self.train_shots)):
                x = temporal_out[:, t, :]
                x = self.model.toHyperX(x, 1)
                temporal_output.append(x)
            temporal_outputs = [g[:, None, :] for g in temporal_output]
            final_outputs = torch.cat(temporal_outputs, dim=1)
            # =======end self-attention=========
            optimizer.zero_grad()
            epoch_losses = []
            epoch_loss = 0
            for t in self.train_shots:
                edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data, t)
                z = final_outputs[:, t, :].squeeze()
                # epoch_loss += self.loss(z, edge_index) + self.model.htc(z)
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

            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
         
            logger.info('==' * 27)
            logger.info("Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, average_epoch_loss, time.time() - t0, gpu_mem_alloc))
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

        logger.info('>> Total time : %6.2f' % (time.time() - t_total0))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |" % (args.lr, args.nhid))

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


# python ./main_single.py  --lr=0.001  --device_id=3 --max_epoch=100 --nhid=16 --nout=16 --ddy_attention_layer_heads=1 --heads=1 --dataset=movielen --split_count=13
if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.loss import ReconLoss
    from script.utils.data_util import loader, prepare_dir
    from script.inits import prepare
    from script.utils.runManager import RunManager
    from script.utils.runbuilder import RunBuilder
    from collections import OrderedDict

    '''
    单步运行, 没有runBuilder
    '''

    params = OrderedDict(
        nhid=[args.nhid],
        nout=[args.nout],
        ddy_attention_layer_heads=[args.ddy_attention_layer_heads],
        heads=[args.heads],
        dataset=[args.dataset],
        split_count=[args.split_count]
    )

    run_manager = RunManager()
    for run in RunBuilder.get_runs(params):
        run_manager.begin_run(run)

        args.nhid = run.nhid
        args.nout = run.nout
        args.ddy_attention_layer_heads = run.ddy_attention_layer_heads
        args.heads = run.heads
        args.dataset = run.dataset

        # 格式化切分目录
        args.split_count = f'{run.split_count}-split'  

        args.output_folder = '../data/output/log/{}/{}/{}/'.format(args.dataset, args.model, args.split_count)

        data = loader(dataset=args.dataset, split_count=args.split_count)
        args.num_nodes = data['num_nodes']
        set_random(args.seed)
        init_logger(prepare_dir(args.output_folder) + args.dataset + '.txt')
        try:
            runner = Runner()
            runner.run(run_manager)
            run_manager.save(args.output_folder, args.split_count, args.heads, args.ddy_attention_layer_heads, args.nout, args.nhid)
        except RuntimeError:
            logger.info(f'current args except runtime error')
        logger.info(f'current args: {run}')
        #run_manager.save('../data/output/log/{}/{}/'.format(args.dataset, args.model))