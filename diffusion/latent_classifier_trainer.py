import math
import copy
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
import csv
import timeit

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from accelerate import Accelerator
import wandb

import diffusion.constant as constant
import diffusion.optimizer as optimizer
import dataset_utils.text_dataset as text_dataset
from utils.torch_utils import compute_grad_norm
import utils.file_utils as file_utils
from evaluation import evaluation

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class AverageMeter:
    def __init__(self, window_size = 10000):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.values = []
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.values.append(val)
        self.sum += val
        self.count += 1
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)

    @property
    def avg(self):
        return self.sum / self.count

class LatentClassifierTrainer(object):
    def __init__(
        self,
        args,
        diffusion,
        classifier,
        dataset_name,
        *,
        train_batch_size = 16,
        eval_batch_size = 64,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 50000,
        lr_schedule = 'inverse_sqrt',
        num_warmup_steps = 500,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        save_every = 1000,
        results_folder = './latent_classifier_results',
        amp = False,
        mixed_precision = 'no',
        split_batches = True,
    ):
        super().__init__()

        set_seeds(42)

        self.args = args

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision,
            log_with='wandb'
        )

        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        self.accelerator.native_amp = amp

        self.diffusion = diffusion
        self.classifier = classifier

        self.save_every = save_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.max_seq_len = diffusion.max_seq_len

        # Init Encoder-decoder model
        assert 'bart' in args.enc_dec_model
        self.bart_model = BartForConditionalGeneration.from_pretrained(args.enc_dec_model)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.enc_dec_model)

        # dataset and dataloader
        dataset = text_dataset.get_dataset(
            dataset_name,
        )

        self.dataset = dataset.shuffle(seed=42)
        self.num_samples = len(self.dataset['valid']['text'])
        # Subsample train and val splits for computing language generation during runtime

        self.dataloader = text_dataset.get_dataloader(args, dataset['train'], self.bart_model.config, self.tokenizer, self.max_seq_len)
        self.val_dataloader = text_dataset.get_dataloader(args, dataset['valid'], self.bart_model.config, self.tokenizer, self.max_seq_len)
        
        training_lengths = [min(sum(self.dataloader.dataset[idx]['attention_mask']), self.max_seq_len) for idx in range(self.dataloader.dataset.num_rows)]
        length_counts = Counter(training_lengths)
        probs = torch.tensor([length_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.max_seq_len+1)])
        assert probs[0] == 0, 'Can\'t have examples of length 0'
        self.length_categorical = torch.distributions.Categorical(probs=probs)

        if self.diffusion.diffusion_model.class_conditional:
            training_labels = [self.dataloader.dataset[idx]['label'] for idx in range(self.dataloader.dataset.num_rows)]
            label_counts = Counter(training_labels)
            probs = torch.tensor([label_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.diffusion.diffusion_model.num_classes)])
            self.class_categorical = torch.distributions.Categorical(probs=probs)

        # optimizer

        self.opt = optimizer.get_adamw_optimizer(classifier.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)

        # scheduler

        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=train_num_steps,
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(classifier, beta = ema_decay, update_every = ema_update_every, power=3/4)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.diffusion, self.classifier, self.bart_model, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(self.diffusion, self.classifier, self.bart_model, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)
        
        self.diffusion.eval()
        self.classifier.train()
        self.bart_model.eval()
        
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)
        self.reference_dict = {}

    def save(self):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.classifier),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model.pt'))

    def load_diffusion(self, diffusion_file_path=None):
        diffusion_file_path = Path(diffusion_file_path) if exists(diffusion_file_path) else self.results_folder
        # diffusion_file_path = Path('/home/ckfan/controlled_text_generation/saved_models/humor_speech/epochs_10000')
        diffusion_data = torch.load(str(diffusion_file_path / f'model.pt'), map_location=self.accelerator.device)
        # diffusion_data = torch.load(str(f'/home/ckfan/controlled_text_generation/saved_models/humor_speech/epochs_10000_both' / f'model.pt'), map_location=self.accelerator.device)
        diffusion = self.accelerator.unwrap_model(self.diffusion)
        diffusion.load_state_dict(diffusion_data['model'])
        print("load successfully")

    def load(self, file_path=None):

        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.classifier)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        # For backwards compatibility with earlier models
        self.ema.load_state_dict(data['ema'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def gen_synthetic_dataset(self, num_samples, seed=42):
        num_classes = self.diffusion.diffusion_model.num_classes
        num_samples_per_class = num_samples//num_classes
        assert num_samples % num_classes == 0, f'Dataset size ({num_samples}) must be divisible by the number of classes ({num_classes})'
        data = {'text': [], 'label': []}
        self.ema.ema_model.eval()
        torch.manual_seed(seed)
        device = self.accelerator.device
        for class_id in range(num_classes):
            text = []
            while len(text) < num_samples_per_class:
                batches = num_to_groups(num_samples_per_class-len(text), self.eval_batch_size)
                model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.diffusion.sample(batch_size=n, length=self.length_categorical.sample((n,)), class_id=torch.tensor([class_id]*n, dtype=torch.long, device=device), classifier=self.classifier)), batches))

                for (latents, mask) in model_outputs:
                    latents, mask = latents.to(device), mask.to(device)
                    if self.args.normalize_latent:
                        latents = self.diffusion.unnormalize_latent(latents)
                    encoder_output = BaseModelOutput(last_hidden_state=latents.clone())
                    sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask.clone(), **constant.generate_kwargs['beam'])
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                    text.extend(texts_list)
            data['text'].extend(text)
            data['label'].extend([class_id]*num_samples_per_class)

        save_path = os.path.join(self.results_folder, f'synth_sample{num_samples}_seed{seed}.csv')
        print(save_path)
        with open(save_path, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))
            
    @torch.no_grad()
    def sample(self, num_samples=None, class_id=None, seed=42, test=False):
        num_samples = default(num_samples, self.num_samples)
        accelerator = self.accelerator
        device = accelerator.device
        # self.diffusion.to('cpu')
        # self.classifier.to('cpu')
        torch.cuda.empty_cache() 

        self.ema.ema_model.eval()

        # Extract references
        reference_texts = {}
        if exists(class_id):
            for filter_class_id in range(2):
                filtered_dataset = self.dataset.filter(lambda example: example["label"]==filter_class_id)
                if test:
                    reference_texts[f'ref{filter_class_id}_test'] = filtered_dataset['test']['text']
                    continue
                reference_texts[f'ref{filter_class_id}_val'] = filtered_dataset['valid']['text']
                reference_texts[f'ref{filter_class_id}_train'] = filtered_dataset['train']['text']
            
            for key, reference_text in reference_texts.items():
                num_samples = min(num_samples, len(reference_text))
            reference_texts = {k: v[:num_samples] for k, v in reference_texts.items()}
        else:
            if test:
                reference_texts[f'test'] = self.dataset['test']['text'][:num_samples]
                reference_texts['train'] = self.dataset['train']['text'][:num_samples]
            else:
                reference_texts['val'] = self.dataset['valid']['text'][:num_samples]
                reference_texts['train'] = self.dataset['train']['text'][:num_samples]

        milestone = 20
        # Stores generation outputs for each strategy
        all_texts_lists = {k:[] for k,_ in constant.generate_kwargs.items()}

        torch.manual_seed(seed)
        def get_class_id(n):
            if exists(class_id):
                return torch.tensor([class_id]*n, dtype=torch.long, device=device)
            if self.diffusion.diffusion_model.class_conditional:
                if self.diffusion.diffusion_model.class_unconditional_prob > 0:
                    return torch.tensor([self.diffusion.diffusion_model.num_classes]*n, dtype=torch.long, device=device)
                return self.class_categorical.sample((n,)).to(device)
            return None
        # Loop until enough senetences have been generated across all strategies 
        while min([len(all_texts_lists[ele]) for ele in all_texts_lists]) < num_samples:
            batches = num_to_groups(num_samples-min([len(all_texts_lists[ele]) for ele in all_texts_lists]), max(self.eval_batch_size,self.train_batch_size))
            model_outputs = list(map(lambda n: tuple(x for x in self.diffusion.sample(batch_size=n, length=self.length_categorical.sample((n,)), class_id=get_class_id(n), classifier=self.classifier)), batches))
            
            for (latents, mask) in model_outputs:
                latents, mask = latents.to(device), mask.to(device)
                if self.args.normalize_latent:
                    latents = self.diffusion.unnormalize_latent(latents)
                for k, kwargs in constant.generate_kwargs.items():
                    encoder_output = BaseModelOutput(last_hidden_state=latents.clone())
                    sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask.clone(), **kwargs)
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                    all_texts_lists[k].extend(texts_list)
        
        assert min([len(all_texts_lists[ele]) for ele in all_texts_lists]) >= num_samples
        text_generations = {k:v[:num_samples] for k,v in all_texts_lists.items()} 
        save_path = os.path.join(self.results_folder, f'synth_sample{num_samples}_seed{seed}.csv')
        print(save_path)
        with open(save_path, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(text_generations.keys())
            writer.writerows(zip(*text_generations.values()))
        metrics = {}

        self.ema.to('cpu')
        torch.cuda.empty_cache() 
        for strategy, all_texts_list in text_generations.items():
            class_id_prefix = f'cond{class_id}_' if exists(class_id) else ''
            file_utils.save_text_samples(all_texts_list, os.path.join(self.results_folder, f'{"eval-" if self.args.eval else ""}{f"eval{seed}-" if self.args.eval_test else ""}{class_id_prefix}{strategy}-sample-{milestone}.txt'))
            metrics[f"model/{strategy}/{class_id_prefix}perplexity"] = evaluation.compute_perplexity(all_texts_list)
            # metrics[f"model/{strategy}/{class_id_prefix}unique_wordcount"] = evaluation.compute_wordcount(all_texts_list)
            ngram_metrics = evaluation.compute_diversity(all_texts_list)
            for k, v in ngram_metrics.items():
                metrics[f"model/{strategy}/{class_id_prefix}{k}"] = v
            metrics[f"model/{strategy}/{class_id_prefix}memorization"] = evaluation.compute_memorization(all_texts_list, self.dataset['train']['text'])
            table = wandb.Table( 
                columns=['Samples'], data=[[text] for text in all_texts_list])
            accelerator.log({f"model/{strategy}/{class_id_prefix}samples": table}, self.step)

            # Only evaluate MAUVE if generations are reasonable
            if metrics[f"model/{strategy}/{class_id_prefix}perplexity"] > 5000:
                continue

            for mauve_model_id in ["gpt2-large", "all-mpnet-base-v2"]:
                for key, reference_text in reference_texts.items():
                    metrics[f"model/{strategy}/{mauve_model_id}_{class_id_prefix}{key}_mauve"], _ = evaluation.compute_mauve(all_texts_list, reference_text, mauve_model_id)

        # if len(self.reference_dict) == 0 or test:
        #     self.log_reference_metrics(test)
        if test:
            metrics_dict = {**metrics,**self.reference_dict}
            metrics_dict = {f'{k}_seed{seed}':v for k,v in metrics_dict.items()}
            accelerator.log(metrics_dict, self.step)
        else:
            accelerator.log({**metrics,**self.reference_dict}, self.step)
        torch.cuda.empty_cache() 
        self.diffusion.to(device)
        self.classifier.to(device)
        self.ema.to(device)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        train_acc_meter = AverageMeter(window_size=250)
        val_acc_meter = AverageMeter()

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                #TODO center and normalize BART latent space with empirical est. of mean/var.

                total_loss = 0.

                for grad_accum_step in range(self.gradient_accumulate_every):
                    data = next(self.data_iter).to(device)
                    with torch.no_grad():
                        latent = self.bart_model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask']).last_hidden_state
                        if self.args.normalize_latent:
                            if self.step==0 and grad_accum_step==0:
                                latent_vecs = torch.cat([latent[i][:torch.sum(data['attention_mask'][i])] for i in range(latent.shape[0])], dim=0)
                                # Add mean stats to diffusion
                                self.diffusion.latent_mean = torch.mean(latent_vecs, dim=0)
                                # Add var stats to diffusion
                                self.diffusion.latent_scale = torch.std(latent_vecs-self.diffusion.latent_mean, unbiased=False)
                            latent = self.diffusion.normalize_latent(latent)

                    mask = data['attention_mask'].bool()
                    with self.accelerator.autocast():

                        t = torch.randint(0, self.diffusion.num_timesteps, (1,), device=device).long()
                        noise = torch.randn_like(latent)
                        x = self.diffusion.q_sample(latent, t, noise=noise)
                        label = data['label']

                        logits = self.classifier(x, mask, t)
                        acc, loss = self.classifier.compute_metrics(logits, label)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        train_acc_meter.update(acc.item())

                    self.accelerator.backward(loss)

                accelerator.wait_for_everyone()

                grad_norm = compute_grad_norm(self.classifier.parameters())

                accelerator.clip_grad_norm_(self.classifier.parameters(), 1.0)
                if not self.args.resume_training:
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    # Log to WandB
                    if self.step % 50 == 0:
                        self.classifier.eval()
                        self.ema.ema_model.eval()
                        with torch.no_grad():
                            total_val_loss = 0.
                            total_val_ema_loss = 0.
                            val_acc_meter.reset()
                            for grad_accum_step in range(self.gradient_accumulate_every):
                                data = next(self.val_iter).to(device)
                                latent = self.bart_model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask']).last_hidden_state
                                if self.args.normalize_latent or self.args.scale_latent:
                                    latent = self.diffusion.normalize_latent(latent)
                                    
                                mask = data['attention_mask'].bool()
                                with self.accelerator.autocast():
                                    
                                    t = torch.randint(0, self.diffusion.num_timesteps, (1,), device=device).long()
                                    noise = torch.randn_like(latent)
                                    x = self.diffusion.q_sample(latent, t, noise=noise)
                                    label = data['label']
                                    logits = self.classifier(x, mask, t)
                                    acc, loss = self.classifier.compute_metrics(logits, label)

                                    loss = loss / self.gradient_accumulate_every
                                    total_val_loss += loss.item()
                                    logits = self.ema.ema_model(x, mask, t)
                                    acc, loss = self.ema.ema_model.compute_metrics(logits, label)
                                    loss = loss / self.gradient_accumulate_every
                                    total_val_ema_loss += loss.item()

                                    val_acc_meter.update(acc.item())

                            logs = {
                                "loss": total_loss, 
                                "val_loss": total_val_loss, 
                                "val_ema_loss": total_val_ema_loss,
                                "train_acc": train_acc_meter.avg,
                                "val_acc": val_acc_meter.avg,
                                "grad_norm": grad_norm, 
                                "lr": self.lr_scheduler.get_last_lr()[0], 
                                "step": self.step, 
                                "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader), 
                                "samples": self.step*self.train_batch_size*self.gradient_accumulate_every,
                            }

                            pbar.set_postfix(**logs)
                            accelerator.log(logs, step=self.step)     

                        self.classifier.train()

                    if self.step % self.args.save_every == 0:
                        self.save()

                pbar.update(1)

        accelerator.print('training complete')