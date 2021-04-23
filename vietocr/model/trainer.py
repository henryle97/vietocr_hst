import math

import tqdm

from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from torch.optim import AdamW, Adam
from vietocr.tool.translate import build_model
from vietocr.tool.translate import translate, batch_translate_beam_search, translate_crnn
from vietocr.loader.aug import ImgAugTransform
import torch
from vietocr.loader.dataloader import OCRDataset, ClusterRandomSampler, Collator
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from vietocr.tool.utils import compute_accuracy, save_predictions
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
from vietocr.tool.config import Cfg
from vietocr.tool.logger import Logger
import pandas as pd



class Trainer():
    def __init__(self, config, pretrained=True, augmentor=ImgAugTransform()):

        self.config = config
        self.model, self.vocab = build_model(config)
        
        self.device = config['device']
        self.num_iters = config['trainer']['iters']
        self.beamsearch = config['predictor']['beamsearch']

        self.data_root = config['dataset']['data_root']
        self.train_annotation = config['dataset']['train_annotation']
        self.valid_annotation = config['dataset']['valid_annotation']
        self.train_lmdb = config['dataset']['train_lmdb']
        self.valid_lmdb = config['dataset']['valid_lmdb']
        self.dataset_name = config['dataset']['name']

        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']
        
        self.image_aug = config['aug']['image_aug']
        self.masked_language_model = config['aug']['masked_language_model']
        self.metrics = config['trainer']['metrics']
        self.is_padding = config['dataset']['is_padding']


        self.tensorboard_dir = config['monitor']['log_dir']
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tensorboard_dir)

        # LOGGER
        self.logger = Logger(config['monitor']['log_dir'])
        self.logger.info(config)

        self.iter = 0
        self.best_acc = 0
        self.scheduler = None
        self.is_finetuning = config['trainer']['is_finetuning']

        if self.is_finetuning:
            self.logger.info("Finetuning model ---->")
            if self.model.seq_modeling == 'crnn':
                self.optimizer = Adam(lr=0.0001, params=self.model.parameters(), betas=(0.5, 0.999))
            else:
                self.optimizer = AdamW(lr=0.0001, params=self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)

        else:

            self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
            self.scheduler = OneCycleLR(self.optimizer, total_steps=self.num_iters, **config['optimizer'])

        if self.model.seq_modeling == 'crnn':
            self.criterion = torch.nn.CTCLoss(self.vocab.pad, zero_infinity=True)
        else:
            self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)

        # Pretrained model
        if config['trainer']['pretrained']:
            self.load_weights(config['trainer']['pretrained'])
            self.logger.info("Loaded trained model from: {}".format(config['trainer']['pretrained']))

        # Resume
        elif config['trainer']['resume_from']:
            self.load_checkpoint(config['trainer']['resume_from'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(torch.device(self.device))

            self.logger.info("Resume training from {}".format(config['trainer']['resume_from']))


        # DATASET
        transforms = None
        if self.image_aug:
            transforms = augmentor

        train_lmdb_paths = [os.path.join(self.data_root, lmdb_path) for lmdb_path in self.train_lmdb]

        self.train_gen = self.data_gen(lmdb_paths=train_lmdb_paths,
                data_root=self.data_root, annotation=self.train_annotation, masked_language_model=self.masked_language_model, transform=transforms)

        if self.valid_annotation:
            self.valid_gen = self.data_gen([os.path.join(self.data_root, self.valid_lmdb)],
                    self.data_root, self.valid_annotation, masked_language_model=False)

        self.train_losses = []
        self.logger.info("Number batch samples of training: %d" % len(self.train_gen))
        self.logger.info("Number batch samples of valid: %d" % len(self.valid_gen))

        config_savepath = os.path.join(self.tensorboard_dir, "config.yml")
        if not os.path.exists(config_savepath):
            self.logger.info("Saving config file at: %s" % config_savepath)
            Cfg(config).save(config_savepath)


        
    def train(self):
        total_loss = 0
        
        total_loader_time = 0
        total_gpu_time = 0
        data_iter = iter(self.train_gen)
        for i in range(self.num_iters):
            self.iter += 1
            start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            total_loader_time += time.time() - start
            start = time.time()

            # LOSS
            loss = self.step(batch)
            total_loss += loss
            self.train_losses.append((self.iter, loss))

            total_gpu_time += time.time() - start

            if self.iter % self.print_every == 0:

                info = 'Iter: {:06d} - Train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(self.iter,
                        total_loss/self.print_every, self.optimizer.param_groups[0]['lr'], 
                        total_loader_time, total_gpu_time)
                lastest_loss = total_loss/self.print_every
                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                self.logger.info(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                val_time = time.time()
                val_loss = self.validate()
                acc_full_seq, acc_per_char, wer = self.precision(self.metrics)

                self.logger.info("Iter: {:06d}, start validating".format(self.iter))
                info = 'Iter: {:06d} - Valid loss: {:.3f} - Acc full seq: {:.4f} - Acc per char: {:.4f} - WER: {:.4f} - Time: {:.4f}'.format(self.iter, val_loss, acc_full_seq, acc_per_char, wer, time.time() - val_time)
                self.logger.info(info)

                if acc_full_seq > self.best_acc:
                    self.save_weights(self.tensorboard_dir + "/best.pt")
                    self.best_acc = acc_full_seq

                self.logger.info("Iter: {:06d} - Best acc: {:.4f}".format(self.iter, self.best_acc))

                filename = 'last.pt'
                filepath = os.path.join(self.tensorboard_dir, filename)
                self.logger.info("Save checkpoint %s" % filename)
                self.save_checkpoint(filepath)

                log_loss = {'train loss': lastest_loss,
                            'val loss': val_loss}
                self.writer.add_scalars('Loss', log_loss, self.iter)
                self.writer.add_scalar('WER', wer, self.iter)

    def validate(self):
        self.model.eval()

        total_loss = []
        
        with torch.no_grad():
            for step, batch in enumerate(self.valid_gen):
                batch = self.batch_to_device(batch)
                img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

                outputs = self.model(img, tgt_input, tgt_padding_mask)
#                loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))

                if self.model.seq_modeling == 'crnn':
                    length = batch['labels_len']
                    preds_size = torch.autograd.Variable(torch.IntTensor([outputs.size(0)] * self.batch_size))
                    loss = self.criterion(outputs, tgt_output, preds_size, length)
                else:
                    outputs = outputs.flatten(0, 1)
                    tgt_output = tgt_output.flatten()
                    loss = self.criterion(outputs, tgt_output)

                total_loss.append(loss.item())
                
                del outputs
                del loss

        total_loss = np.mean(total_loss)
        self.model.train()
        
        return total_loss
    
    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        img_files = []
        probs_sents = []
        imgs_sents = []

        for idx, batch in enumerate(tqdm.tqdm(self.valid_gen)):
            batch = self.batch_to_device(batch)

            if self.model.seq_modeling != 'crnn':
                if self.beamsearch:
                    translated_sentence = batch_translate_beam_search(batch['img'], self.model)
                    prob = None
                else:
                    translated_sentence, prob = translate(batch['img'], self.model)
                pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            else:
                translated_sentence, prob = translate_crnn(batch['img'], self.model)
                pred_sent = self.vocab.batch_decode(translated_sentence.tolist(), crnn=True)

            actual_sent = self.vocab.batch_decode(batch['tgt_output'].tolist())
            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)

            imgs_sents.extend(batch['img'])
            img_files.extend(batch['filenames'])
            probs_sents.extend(prob)


            # Visualize in tensorboard
            if idx == 0:
                try:
                    num_samples = self.config['monitor']['num_samples']
                    fig = plt.figure(figsize=(12, 15))
                    imgs_samples = imgs_sents[:num_samples]
                    preds_samples = pred_sents[:num_samples]
                    actuals_samples = actual_sents[:num_samples]
                    probs_samples = probs_sents[:num_samples]
                    for id_img in range(len(imgs_samples)):
                        img = imgs_samples[id_img]
                        img = img.permute(1, 2, 0)
                        img = img.cpu().detach().numpy()
                        ax = fig.add_subplot(num_samples, 1, id_img+1, xticks=[], yticks=[])
                        plt.imshow(img)
                        ax.set_title("LB: {} \n Pred: {:.4f}-{}".format(actuals_samples[id_img], probs_samples[id_img], preds_samples[id_img]),
                                     color=('green' if actuals_samples[id_img] == preds_samples[id_img] else 'red'),
                                     fontdict={'fontsize': 18, 'fontweight': 'medium'})

                    self.writer.add_figure('predictions vs. actuals',
                                      fig,
                                      global_step=self.iter)
                except Exception as error:
                    print(error)
                    continue

            if sample != None and len(pred_sents) > sample:
                break

        return pred_sents, actual_sents, img_files, probs_sents, imgs_sents

    def precision(self, sample=None, measure_time=True):
        t1 = time.time()
        pred_sents, actual_sents, _, _, _ = self.predict(sample=sample)
        time_predict = time.time() - t1

        sensitive_case = self.config['predictor']['sensitive_case']
        acc_full_seq = compute_accuracy(actual_sents, pred_sents,sensitive_case , mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents,sensitive_case, mode='per_char')
        wer = compute_accuracy(actual_sents, pred_sents,sensitive_case, mode='wer')


        if measure_time:
            print("Time: {:.4f}".format(time_predict / len(actual_sents)))
        return acc_full_seq, acc_per_char, wer

    def visualize_prediction(self, sample=16, errorcase=False, fontname='serif', fontsize=16, save_fig=False):

        pred_sents, actual_sents, img_files, probs, imgs = self.predict(sample)

        if errorcase:
            wrongs = []
            for i in range(len(img_files)):
                if pred_sents[i] != actual_sents[i]:
                    wrongs.append(i)

            pred_sents = [pred_sents[i] for i in wrongs]
            actual_sents = [actual_sents[i] for i in wrongs]
            img_files = [img_files[i] for i in wrongs]
            probs = [probs[i] for i in wrongs]
            imgs = [imgs[i] for i in wrongs]

        img_files = img_files[:sample]

        fontdict = {
            'family': fontname,
            'size': fontsize
        }
        ncols = 5
        nrows = int(math.ceil(len(img_files) / ncols))
        fig, ax = plt.subplots(nrows, ncols, figsize=(12, 15))

        for vis_idx in range(0, len(img_files)):
            row = vis_idx // ncols
            col = vis_idx % ncols

            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]
            prob = probs[vis_idx]
            img = imgs[vis_idx].permute(1, 2, 0).cpu().detach().numpy()

            ax[row, col].imshow(img)
            ax[row, col].set_title("Pred: {: <2} \n Actual: {} \n prob: {:.2f}".format(pred_sent, actual_sent, prob), fontname=fontname, color='r' if pred_sent != actual_sent else 'g')
            ax[row, col].get_xaxis().set_ticks([])
            ax[row, col].get_yaxis().set_ticks([])

        plt.subplots_adjust()
        if save_fig:
            fig.savefig('vis_prediction.png')
        plt.show()

    def log_prediction(self, sample=16, csv_file='model.csv'):
        pred_sents, actual_sents, img_files, probs, imgs = self.predict(sample)
        save_predictions(csv_file, pred_sents, actual_sents, img_files)

    def vis_data(self, sample=20):

        ncols = 5
        nrows = int(math.ceil(sample / ncols))
        fig, ax = plt.subplots(nrows, ncols, figsize=(12, 12))

        num_plots = 0
        for idx, batch in enumerate(self.train_gen):
            for vis_idx in range(self.batch_size):
                row = vis_idx // ncols
                col = vis_idx % ncols

                img = batch['img'][vis_idx].numpy().transpose(1, 2, 0)
                sent = self.vocab.decode(batch['tgt_input'].T[vis_idx].tolist())

                ax[row, col].imshow(img)
                ax[row, col].set_title("Label: {: <2}".format(sent), fontsize=16, color='g')

                ax[row, col].get_xaxis().set_ticks([])
                ax[row, col].get_yaxis().set_ticks([])

                num_plots += 1
                if num_plots >= sample:
                    plt.subplots_adjust()
                    fig.savefig('vis_dataset.png')
                    return



    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.iter = checkpoint['iter']
        self.train_losses = checkpoint['train_losses']
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.best_acc = checkpoint['best_acc']

    def save_checkpoint(self, filename):
        state = {'iter': self.iter,
                 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'train_losses': self.train_losses,
                 'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
                 'best_acc': self.best_acc
                 }
        
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))
        if self.is_checkpoint(state_dict):
            self.model.load_state_dict(state_dict['state_dict'])
        else:

            for name, param in self.model.named_parameters():
                if name not in state_dict:
                    print('{} not found'.format(name))
                elif state_dict[name].shape != param.shape:
                    print('{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                    del state_dict[name]
            self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
       
        torch.save(self.model.state_dict(), filename)

    def is_checkpoint(self, checkpoint):
        try:
            checkpoint['state_dict']
        except:
            return False
        else:
            return True

    def batch_to_device(self, batch):
        img = batch['img'].to(self.device, non_blocking=True)
        tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
        tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
        tgt_padding_mask = batch['tgt_padding_mask'].to(self.device, non_blocking=True)

        batch = {
                'img': img, 'tgt_input':tgt_input, 
                'tgt_output':tgt_output, 'tgt_padding_mask':tgt_padding_mask, 
                'filenames': batch['filenames'],
                'labels_len': batch['labels_len']
                }

        return batch

    def data_gen(self, lmdb_paths, data_root, annotation, masked_language_model=True, transform=None):
        datasets = []
        for lmdb_path in lmdb_paths:
            dataset = OCRDataset(lmdb_path=lmdb_path,
                    root_dir=data_root, annotation_path=annotation,
                    vocab=self.vocab, transform=transform,
                    image_height=self.config['dataset']['image_height'],
                    image_min_width=self.config['dataset']['image_min_width'],
                    image_max_width=self.config['dataset']['image_max_width'],
                                 separate=self.config['dataset']['separate'],
                                 batch_size=self.batch_size,
                                 is_padding=self.is_padding)
            datasets.append(dataset)
        if len(self.train_lmdb) > 1:
            dataset = torch.utils.data.ConcatDataset(datasets)

        if self.is_padding:
            sampler = None
        else:
            sampler = ClusterRandomSampler(dataset, self.batch_size, True)

        collate_fn = Collator(masked_language_model)

        gen = DataLoader(
                dataset,
                batch_size=self.batch_size, 
                sampler=sampler,
                collate_fn = collate_fn,
                shuffle= self.is_padding and 'train' in lmdb_path[0],
                drop_last=self.model.seq_modeling == 'crnn',
                **self.config['dataloader'])
       
        return gen


    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']    
        
        outputs = self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)
#        loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))

        if self.model.seq_modeling == 'crnn':
            length = batch['labels_len']
            preds_size = torch.autograd.Variable(torch.IntTensor([outputs.size(0)] * self.batch_size))
            loss = self.criterion(outputs, tgt_output, preds_size, length)
        else:
            outputs = outputs.view(-1, outputs.size(2))  # flatten(0, 1)    # B*S x N_class
            tgt_output = tgt_output.view(-1)  # flatten()    # B*S
            loss = self.criterion(outputs, tgt_output)

        self.optimizer.zero_grad()

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) 

        self.optimizer.step()

        if not self.is_finetuning:
            self.scheduler.step()

        loss_item = loss.item()

        return loss_item

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def gen_pseudo_labels(self, outfile=None):
        pred_sents = []
        img_files = []
        probs_sents = []

        for idx, batch in enumerate(tqdm.tqdm(self.valid_gen)):
            batch = self.batch_to_device(batch)

            if self.model.seq_modeling != 'crnn':
                if self.beamsearch:
                    translated_sentence = batch_translate_beam_search(batch['img'], self.model)
                    prob = None
                else:
                    translated_sentence, prob = translate(batch['img'], self.model)
                pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            else:
                translated_sentence, prob = translate_crnn(batch['img'], self.model)
                pred_sent = self.vocab.batch_decode(translated_sentence.tolist(), crnn=True)

            pred_sents.extend(pred_sent)
            img_files.extend(batch['filenames'])
            probs_sents.extend(prob)
        assert len(pred_sents) == len(img_files) and len(img_files) == len(probs_sents)
        with open(outfile, 'w', encoding='utf-8') as f:
            for anno in zip(img_files, pred_sents, probs_sents):
                f.write('||||'.join([anno[0], anno[1], str(float(anno[2]))]) + '\n')

