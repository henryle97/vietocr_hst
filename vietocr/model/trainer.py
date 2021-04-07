from vietocr.optim.optim import ScheduledOptim
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from torch.optim import Adam, SGD, AdamW
from vietocr.tool.translate import build_model
from vietocr.tool.translate import translate, batch_translate_beam_search, translate_crnn
from vietocr.tool.utils import download_weights
from vietocr.tool.logger import Logger
from vietocr.loader.aug import ImgAugTransform
import torch
from vietocr.loader.dataloader import OCRDataset, ClusterRandomSampler, Collator
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, OneCycleLR

from vietocr.tool.utils import compute_accuracy
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter


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
        self.dataset_name = config['dataset']['name']

        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']
        
        self.image_aug = config['aug']['image_aug']
        self.masked_language_model = config['aug']['masked_language_model']
        self.metrics = config['trainer']['metrics']

        # LOGGER
        # logger = config['trainer']['log']
        # if logger:
        #     self.logger = Logger(logger)

        self.tensorboard_dir = config['monitor']['log_dir']
        if not os.path.exists( self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tensorboard_dir)

        # if pretrained:
        #     print("Loading pretrained weight...")
        #     weight_file = download_weights(**config['pretrain'], quiet=config['quiet'])
        #     self.load_weights(weight_file)
        if config['trainer']['pretrained']:

            self.load_weights(config['trainer']['pretrained'])
            print("Loaded trained model from: {}".format(config['trainer']['pretrained']))

        self.iter = 0
        self.best_acc = 0

        self.scheduler = None
        self.is_finetuning = config['trainer']['is_finetuning']
        if self.is_finetuning:
            print("Finetuning model ---->")
            self.optimizer = AdamW(lr=0.0001, params=self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)

        else:

            self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
            self.scheduler = OneCycleLR(self.optimizer, total_steps=self.num_iters, **config['optimizer'])

        if self.model.seq_modeling == 'crnn':
            self.criterion = torch.nn.CTCLoss(self.vocab.pad)
        else:
            self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)



        # Resume
        if config['trainer']['resume_from']:
            self.load_checkpoint(config['trainer']['resume_from'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(torch.device(self.device))

            print("Resume from {}".format(config['trainer']['resume_from']))

        transforms = None
        if self.image_aug:
            transforms = augmentor

        self.train_gen = self.data_gen(self.data_root + "/" + 'train_{}'.format(self.dataset_name),
                self.data_root, self.train_annotation, self.masked_language_model, transform=transforms)
        if self.valid_annotation:
            self.valid_gen = self.data_gen(self.data_root + "/" + 'valid_{}'.format(self.dataset_name),
                    self.data_root, self.valid_annotation, masked_language_model=False)

        self.train_losses = []
        print("\nNumber batch samples of training: ", len(self.train_gen))
        print("Number batch samples of valid: ", len(self.valid_gen))
        
    def train(self):
        total_loss = 0
        
        total_loader_time = 0
        total_gpu_time = 0
        best_acc = 0

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
            loss = self.step(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:

                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(self.iter, 
                        total_loss/self.print_every, self.optimizer.param_groups[0]['lr'], 
                        total_loader_time, total_gpu_time)
                lastest_loss = total_loss/self.print_every
                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info) 
                # self.logger.log(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                val_time = time.time()
                val_loss = self.validate()
                acc_full_seq, acc_per_char, wer = self.precision(self.metrics)

                info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f} - WER: {:.4f} - Time: {:.4f}'.format(self.iter, val_loss, acc_full_seq, acc_per_char, wer, time.time() - val_time)

                print(info)
                # self.logger.log(info)

                if acc_full_seq > self.best_acc:
                    self.save_weights(self.tensorboard_dir + "/best.pt")
                    self.best_acc = acc_full_seq

                self.save_checkpoint(self.tensorboard_dir + "/last.pt")

                log_loss = {'train loss': lastest_loss,
                            'val loss': val_loss}
                self.writer.add_scalars('Loss', log_loss, self.iter)
                # self.writer.add_scalar('valid loss', val_loss, self.iter)
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
               
                outputs = outputs.flatten(0,1)
                tgt_output = tgt_output.flatten()
                loss = self.criterion(outputs, tgt_output)

                total_loss.append(loss.item())
                
                del outputs
                del loss

        total_loss = np.mean(total_loss)
        self.model.train()
        
        return total_loss
    
    def predict(self, sample=None, vis_tensorboard=False):
        pred_sents = []
        actual_sents = []
        img_files = []
        probs_sents = []
        imgs_sents = []

        for idx, batch in enumerate(self.valid_gen):
            batch = self.batch_to_device(batch)

            if self.model.seq_modeling != 'crnn':
                if self.beamsearch:
                    translated_sentence = batch_translate_beam_search(batch['img'], self.model)
                    prob = None
                else:
                    translated_sentence, prob = translate(batch['img'], self.model)
            else:
                translated_sentence, prob = translate_crnn(batch['img'], self.model)
            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch['tgt_output'].tolist())
            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)

            imgs_sents.extend(batch['img'])
            # img_files.extend(batch['filenames'])
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

    def precision(self, sample=None):

        pred_sents, actual_sents, _, _, _ = self.predict(sample=sample)

        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')
        wer = compute_accuracy(actual_sents, pred_sents, mode='wer')

        return acc_full_seq, acc_per_char, wer
    
    def visualize_prediction(self, sample=16, errorcase=False, fontname='serif', fontsize=16):
        
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

        img_files = img_files[:sample]

        fontdict = {
                'family':fontname,
                'size':fontsize
                } 

        for vis_idx in range(0, len(img_files)):
            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]
            prob = probs[vis_idx]
            img = imgs[vis_idx].permute(1, 2, 0).cpu().detach().numpy()
            plt.figure()
            plt.imshow(img)
            plt.title('pred: {} || prob: {:.3f} \n actual: {}'.format(pred_sent, prob, actual_sent), loc='left', fontdict=fontdict)
            plt.axis('off')

        plt.show()
    
    def visualize_dataset(self, sample=16, fontname='serif'):
        n = 0
        for batch in self.train_gen:
            for i in range(self.batch_size):
                img = batch['img'][i].numpy().transpose(1, 2, 0)
                sent = self.vocab.decode(batch['tgt_input'].T[i].tolist())
                
                plt.figure()
                plt.title('sent: {}'.format(sent), loc='center', fontname=fontname)
                plt.imshow(img)
                plt.axis('off')
                
                n += 1
                if n >= sample:
                    plt.show()
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

    def batch_to_device(self, batch):
        img = batch['img'].to(self.device, non_blocking=True)
        tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
        tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
        tgt_padding_mask = batch['tgt_padding_mask'].to(self.device, non_blocking=True)

        batch = {
                'img': img, 'tgt_input':tgt_input, 
                'tgt_output':tgt_output, 'tgt_padding_mask':tgt_padding_mask, 
                'filenames': batch['filenames']
                }

        return batch

    def data_gen(self, lmdb_path, data_root, annotation, masked_language_model=True, transform=None):
        dataset = OCRDataset(lmdb_path=lmdb_path, 
                root_dir=data_root, annotation_path=annotation, 
                vocab=self.vocab, transform=transform, 
                image_height=self.config['dataset']['image_height'], 
                image_min_width=self.config['dataset']['image_min_width'], 
                image_max_width=self.config['dataset']['image_max_width'],
                             separate=self.config['dataset']['separate'])

        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)

        gen = DataLoader(
                dataset,
                batch_size=self.batch_size, 
                sampler=sampler,
                collate_fn = collate_fn,
                shuffle=False,
                drop_last=False,
                **self.config['dataloader'])
       
        return gen


    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']    
        
        outputs = self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)
#        loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))
        outputs = outputs.view(-1, outputs.size(2))  #flatten(0, 1)    # B*S x N_class
        tgt_output = tgt_output.view(-1)#flatten()    # B*S
        
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

