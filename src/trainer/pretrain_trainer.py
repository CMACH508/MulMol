import torch
import numpy as np
from sklearn.metrics import f1_score
class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, clf_loss_fn_cf, sl_loss_fn, reg_evaluator, clf_evaluator_fp, clf_evaluator_cf, result_tracker, summary_writer, device, ddp=False, local_rank=1):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.reg_loss_fn = reg_loss_fn
        self.clf_loss_fn = clf_loss_fn
        self.clf_loss_fn_cf = clf_loss_fn_cf
        self.sl_loss_fn = sl_loss_fn
        self.reg_evaluator = reg_evaluator
        self.clf_evaluator_fp = clf_evaluator_fp
        self.clf_evaluator_cf = clf_evaluator_cf
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.ddp = ddp
        self.local_rank = local_rank
        self.n_updates = 0
    def _forward_epoch(self, model, batched_data, idx):
        # here
        (smiles, batched_graph, fps, cfs, mds, sl_labels, disturbed_fps, disturbed_cfs, disturbed_mds) = batched_data
        batched_graph = batched_graph.to(self.device)
        fps = fps.to(self.device)
        # here
        cfs = cfs.to(self.device)
        mds = mds.to(self.device)
        sl_labels = sl_labels.to(self.device)
        disturbed_fps = disturbed_fps.to(self.device)
        # here
        disturbed_cfs = disturbed_cfs.to(self.device)
        disturbed_mds = disturbed_mds.to(self.device)
        # here
        triplet_emb, nd_pre, fp_pre, cf_pre, md_pre = model(batched_graph, disturbed_fps, disturbed_cfs, disturbed_mds, idx)
        mask_replace_keep = batched_graph.ndata['mask'][batched_graph.ndata['mask']>=1].cpu().numpy()
        # here
        return mask_replace_keep, triplet_emb, nd_pre, sl_labels, fp_pre, fps, disturbed_fps, cf_pre, cfs, md_pre, mds
    
    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            triplet = []
            nd_pre = []
            fp_pre = []
            cf_pre = []
            md_pre = []
            sl_loss = []
            fp_loss = []
            cf_loss = []
            md_loss = []
            try:
                print("start compute loss")
                self.optimizer.zero_grad()
                for idx in range(3):
                    # here
                    mask_replace_keep, triplet_emb, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, cf_predictions, cfs, md_predictions, mds = self._forward_epoch(model, batched_data, idx+1)
                    # mask_replace_keep, nd_pre_1, nd_pre_2, sl_labels, fp_pre_1, fp_pre_2, fp_irs, fps, md_pre_1, md_pre_2, md_irs, mds = self._forward_epoch(model, batched_data, idx+1)
                    batch_size = fp_predictions.shape[0]
                    triplet.append(triplet_emb)
                    nd_pre.append(sl_predictions)
                    fp_pre.append(fp_predictions)
                    # here
                    cf_pre.append(cf_predictions)
                    md_pre.append(md_predictions)
                    sl_loss.append(self.sl_loss_fn(sl_predictions, sl_labels).mean())
                    fp_loss.append(self.clf_loss_fn(fp_predictions, fps).mean())
                    # here
                    cf_loss.append(self.clf_loss_fn_cf(cf_predictions, cfs).mean())
                    md_loss.append(self.reg_loss_fn(md_predictions, mds).mean())
                
                # compute similarity
                cos_sim = torch.nn.CosineSimilarity(dim=1)
                
                # contrastive loss
                rs_loss = 0
                nce_loss = 0
                for i in range(3):
                    emb_similarity = torch.mean(cos_sim(triplet[i], triplet[(i+1)%3]))
                    fp_ma = self.compute_similarity_matrix(fp_pre[i], fp_pre[(i+1)%3])
                    md_ma = self.compute_similarity_matrix(md_pre[i], md_pre[(i+1)%3])
                    cf_ma = self.compute_similarity_matrix(cf_pre[i], cf_pre[(i+1)%3])

                    # pos-pos
                    fp_pos_similarity = torch.sum(torch.diag(fp_ma))/batch_size
                    md_pos_similarity = torch.sum(torch.diag(md_ma))/batch_size
                    cf_pos_similarity = torch.sum(torch.diag(cf_ma))/batch_size
                    fp_rs = fp_pos_similarity.view(1) - emb_similarity.view(1)
                    md_rs = md_pos_similarity.view(1) - emb_similarity.view(1)
                    cf_rs = cf_pos_similarity.view(1) - emb_similarity.view(1)
                    rs_loss = rs_loss + (max(0,(0.3-fp_rs)) + max(0,(0.3-md_rs)) + max(0,(0.3-cf_rs)))/3

                    # pos-neg
                    fp_ma_exp = fp_ma.exp()
                    fp_pos_exp = torch.sum(torch.diag(fp_ma_exp))
                    fp_neg_exp = torch.sum(torch.triu(fp_ma_exp, diagonal=1))
                    fp_nce_loss = self.contrastive_loss(fp_pos_exp, fp_neg_exp)

                    md_ma_exp = md_ma.exp()
                    md_pos_exp = torch.sum(torch.diag(md_ma_exp))
                    md_neg_exp = torch.sum(torch.triu(md_ma_exp, diagonal=1))
                    md_nce_loss = self.contrastive_loss(md_pos_exp, md_neg_exp)

                    cf_ma_exp = cf_ma.exp()
                    cf_pos_exp = torch.sum(torch.diag(cf_ma_exp))
                    cf_neg_exp = torch.sum(torch.triu(cf_ma_exp, diagonal=1))
                    cf_nce_loss = self.contrastive_loss(cf_pos_exp, cf_neg_exp)

                    nce_loss = nce_loss + (fp_nce_loss + md_nce_loss + cf_nce_loss)/3
                
                rs_loss = rs_loss / 3
                nce_loss = nce_loss / 3
                # similiarity_emb = 0
                # similiarity_fp = 0
                # # here
                # similiarity_cf = 0
                # similiarity_md = 0
                # for i in range(3):
                #     similiarity_emb += torch.mean(cos_sim(triplet[i], triplet[(i+1)%3]))
                #     similiarity_fp += torch.mean(cos_sim(fp_pre[i], fp_pre[(i+1)%3]))
                #     # here
                #     similiarity_cf += torch.mean(cos_sim(cf_pre[i], cf_pre[(i+1)%3]))
                #     similiarity_md += torch.mean(cos_sim(md_pre[i], md_pre[(i+1)%3]))

                # fp_irs = similiarity_fp - similiarity_emb
                # cf_irs = similiarity_cf - similiarity_emb
                # md_irs = similiarity_md - similiarity_emb

                # irs_loss = (max(0, 1.5-fp_irs) + max(0, 1.5-md_irs) + max(0, 1.5-cf_irs))/3

                loss = (sum(sl_loss)/3 + sum(fp_loss)/3 + sum(cf_loss)/3 + sum(md_loss)/3 + rs_loss + nce_loss)/6
                # loss = (sum(sl_loss)/3 + sum(fp_loss)/3 + sum(cf_loss)/3 + sum(md_loss)/3  + nce_loss)/5
                # loss = (sum(sl_loss)/3 + nce_loss) / 2
                print("loss =", loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                self.optimizer.step()
                self.n_updates += 1
                self.lr_scheduler.step()
                if self.summary_writer is not None:
                    loss_mask_1 = self.sl_loss_fn(nd_pre[0].detach().cpu()[mask_replace_keep==1],sl_labels.detach().cpu()[mask_replace_keep==1]).mean()
                    loss_mask_2 = self.sl_loss_fn(nd_pre[1].detach().cpu()[mask_replace_keep==2],sl_labels.detach().cpu()[mask_replace_keep==2]).mean()
                    loss_mask_3 = self.sl_loss_fn(nd_pre[2].detach().cpu()[mask_replace_keep==3],sl_labels.detach().cpu()[mask_replace_keep==3]).mean()
                    loss_mask = (loss_mask_1 + loss_mask_2 + loss_mask_3)/3
                    loss_replace_1 = self.sl_loss_fn(nd_pre[0].detach().cpu()[mask_replace_keep==4],sl_labels.detach().cpu()[mask_replace_keep==4]).mean()
                    loss_replace_2 = self.sl_loss_fn(nd_pre[1].detach().cpu()[mask_replace_keep==4],sl_labels.detach().cpu()[mask_replace_keep==4]).mean()
                    loss_replace_3 = self.sl_loss_fn(nd_pre[2].detach().cpu()[mask_replace_keep==4],sl_labels.detach().cpu()[mask_replace_keep==4]).mean()
                    loss_replace = (loss_replace_1 + loss_replace_2 + loss_replace_3)/3
                    loss_keep_1 = self.sl_loss_fn(nd_pre[0].detach().cpu()[mask_replace_keep==5],sl_labels.detach().cpu()[mask_replace_keep==5]).mean()
                    loss_keep_2 = self.sl_loss_fn(nd_pre[1].detach().cpu()[mask_replace_keep==5],sl_labels.detach().cpu()[mask_replace_keep==5]).mean()
                    loss_keep_3 = self.sl_loss_fn(nd_pre[2].detach().cpu()[mask_replace_keep==5],sl_labels.detach().cpu()[mask_replace_keep==5]).mean()
                    loss_keep = (loss_keep_1 + loss_keep_2 + loss_keep_3)/3
                    preds = np.argmax(nd_pre[0].detach().cpu().numpy(),axis=-1)
                    # preds_2 = np.argmax(nd_pre[1].detach().cpu().numpy(),axis=-1)
                    labels = sl_labels.detach().cpu().numpy()

                    self.summary_writer.add_scalar('Loss/loss_tot', loss, self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_bert', sum(sl_loss)/3, self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_mask', loss_mask, self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_replace', loss_replace, self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_keep', loss_keep, self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_clf_fp', sum(fp_loss)/3, self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_clf_cf', sum(cf_loss)/3, self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_reg', sum(md_loss)/3, self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_rise', rs_loss, self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_nce', nce_loss, self.n_updates)
                    
                    self.summary_writer.add_scalar('F1_micro/all', f1_score(preds, labels, average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/all', f1_score(preds, labels, average='macro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_micro/mask', f1_score(preds[mask_replace_keep==1], labels[mask_replace_keep==1], average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/mask', f1_score(preds[mask_replace_keep==1], labels[mask_replace_keep==1], average='macro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_micro/replace', f1_score(preds[mask_replace_keep==4], labels[mask_replace_keep==4], average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/replace', f1_score(preds[mask_replace_keep==4], labels[mask_replace_keep==4], average='macro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_micro/keep', f1_score(preds[mask_replace_keep==5], labels[mask_replace_keep==5], average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/keep', f1_score(preds[mask_replace_keep==5], labels[mask_replace_keep==5], average='macro'), self.n_updates)
                    self.summary_writer.add_scalar(f'Clf/{self.clf_evaluator_fp.eval_metric}_fp', np.mean(self.clf_evaluator_fp.eval(fps, fp_pre[0])), self.n_updates)
                    self.summary_writer.add_scalar(f'Clf/{self.clf_evaluator_cf.eval_metric}_cf', np.mean(self.clf_evaluator_cf.eval(cfs, cf_pre[0])), self.n_updates)
                if self.n_updates == self.args.n_steps:
                    if self.local_rank == 0:
                        self.save_model(model)
                    break

            except Exception as e:
                print(e)
            else:
                continue

    def fit(self, model, train_loader):
        for epoch in range(1, 1001):
            model.train()
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.n_updates >= self.args.n_steps:
                break

    def save_model(self, model):
        torch.save(model.state_dict(), self.args.save_path+f"/{self.args.config}.pth")
    
    def contrastive_loss(self, pos, neg):
        return (-torch.log(pos / (pos + neg))).mean()
    
    def compute_similarity_matrix(self, matrix1, matrix2):
        expanded_matrix1 = matrix1.unsqueeze(1)  
        expanded_matrix2 = matrix2.unsqueeze(0) 
        
        dot_products = torch.sum(expanded_matrix1 * expanded_matrix2, dim=-1)  
        norms1 = torch.norm(expanded_matrix1, dim=-1)  
        norms2 = torch.norm(expanded_matrix2, dim=-1)  
        similarity_matrix = dot_products / (norms1 * norms2) 
        
        return similarity_matrix

    