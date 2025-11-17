import argparse
import re

from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, default='LD')
parser.add_argument('--name', type=str, default='HASS')
parser.add_argument('--basepath', type=str, default=None)
parser.add_argument('--configpath', type=str, default=None)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default=None)
parser.add_argument('--cpdir', type=str, default=None)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--topk_w', type=float, default=1.0)
parser.add_argument('--forward_num_total', type=int, default=3)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--data_num', type=int, default=68000)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--use_adapter', action='store_true')
parser.add_argument('--train_LD', action='store_true')
parser.add_argument('--hass_path', type=str, default=None)
parser.add_argument('--p_w', type=float, default=0.1)
parser.add_argument('--v_w', type=float, default=1.0)

args = parser.parse_args()

total_steps = int(args.data_num * 0.95 * (args.epoch + 1) / (args.bs * args.gradient_accumulation_steps))
warm_steps = total_steps // 100

train_config = {
    "data_num": args.data_num,
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": args.epoch,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": warm_steps,
    "total_steps": total_steps,
    "p_w": args.p_w,
    "v_w": args.v_w,
    "topk_w": args.topk_w,
    "head_w": 0.1,
    "num_workers": 8,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 1
}
import json
import safetensors
from safetensors import safe_open
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(
    mixed_precision='bf16',
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"]
)
from LD.large_drafter import LargeDrafter

from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig

if accelerator.is_main_process:
    import wandb

    wandb.init(entity='ciai-llm', project=args.project, name=args.name, mode='offline', config=train_config)

baseconfig = AutoConfig.from_pretrained(args.basepath)

head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def compute_loss(target, target_p, predict, loss_mask):
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)

    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)

    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)

    topk_mask = torch.topk(target_p, k=args.topk, dim=2).indices
    topk_loss = -torch.sum(torch.sum(loss_mask * plogp.gather(dim=2, index=topk_mask), 2)) / (loss_mask.sum() + 1e-5)

    return vloss, ploss, topk_loss, out_head


@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    def generate(hidden_states, input_ids, head, max_length=4, use_cache=True):
        if use_cache:
            past_key_values = None
            unwarped_model = accelerator.unwrap_model(model)
            unwarped_model.reset_step()
            for i in range(max_length):
                if i < args.forward_num_total:
                    assert i == unwarped_model.current_step, i
                if past_key_values != None:
                    out_hidden, past_key_values, sample_hidden = model(last_hidden, input_ids=token,
                                                                       past_key_values=past_key_values,
                                                                       use_cache=True)
                else:
                    out_hidden, past_key_values, sample_hidden = model(hidden_states, input_ids=input_ids,
                                                                       use_cache=True)
                last_hidden = out_hidden[:, -1:]
                if not args.use_adapter:
                    last_headout = head(last_hidden)
                else:
                    last_headout = head(sample_hidden[:, -1:])
                token = torch.argmax(last_headout, dim=-1)
                input_ids = torch.cat((input_ids, token), dim=1)
        else:
            raise NotImplementedError

        return input_ids

    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    target_ids = target_headout.argmax(dim=2)

    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0:
            continue
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_input_ids = input_ids[:, :pre_len]
        outs = generate(pre_hidden_states, pre_input_ids, head, max_length=max_length)
        generate_ids = outs[:, pre_len:]
        for bid in range(bs):
            for k in range(max_length):
                if pre_len + k >= seq_len:
                    break
                if loss_mask[bid, pre_len + k] == 0:
                    break
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc


# =================
#      DATASET
# =================
SEED = 42

dataset_info = {
    "ShareGPT": {
        "conv_key": "conversations",
        "role_key": "from",
        "content_key": "value"
    },
    "UltraChat": {
        "conv_key": "messages",
        "role_key": "role",
        "content_key": "content"
    },
    "OpenThoughts2": {
        "conv_key": "conversations",
        "role_key": "from",
        "content_key": "value"
    }
}


def build_dataset_rank(tokenizer, dataset, name: str, num_proc: int = 16):
    info = dataset_info[name]

    def preprocess_function(examples):
        new_examples = {
            "input_ids": [],
            "loss_mask": [],
            "attention_mask": []
        }
        for i in range(len(examples[info["conv_key"]])):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
                        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                        "If you don't know the answer to a question, please don't share false information."
                    )
                }
            ]

            convroles = ["user", "assistant"]
            if name == "ShareGPT":
                roles = {"human": convroles[0], "gpt": convroles[1]}
            else:
                roles = {"user": convroles[0], "assistant": convroles[1]}

            source = examples[info["conv_key"]][i]
            if roles[source[0][info["role_key"]]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]

            for j, sentence in enumerate(source):
                role = roles[sentence[info["role_key"]]]
                assert role == convroles[j % 2], f"{i}"
                content = sentence[info["content_key"]]

                if name == "OpenThoughts2" and role == "assistant":
                    content = re.sub(
                        r"<think>.*?</think>", "", content, flags=re.DOTALL
                    ).strip()

                messages.append(
                    {"role": role, "content": content}
                )

            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                truncation=True,
                max_length=train_config["max_len"],
                add_special_tokens=False
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)
            # print(i)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            total_len = len(input_ids)

            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions
                if i == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3

            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )

    dataset.set_format(type="torch")
    return dataset


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


tokenizer = AutoTokenizer.from_pretrained(args.basepath)
datapath = train_config["datapath"]

# ShareGPT
dataset_0 = load_dataset('json', data_files=f'{datapath}/ShareGPT_V4.3_unfiltered_cleaned_split.json')
dataset_0 = dataset_0['train'].shuffle(seed=SEED).select(range(68000))
dataset_0 = build_dataset_rank(tokenizer, dataset_0, name='ShareGPT')

# UltraChat
dataset_1 = load_dataset(f'{datapath}/ultrachat_200k')
dataset_1 = concatenate_datasets(
    [dataset_1['train_sft'], dataset_1['train_gen']]).shuffle(seed=SEED).select(range(463000))
dataset_1 = build_dataset_rank(tokenizer, dataset_1, name='UltraChat')

# OpenThoughts2
dataset_2 = load_dataset(f'{datapath}/OpenThoughts2-1M')
dataset_2 = dataset_2['train'].shuffle(seed=SEED).select(range(269000))
dataset_2 = build_dataset_rank(tokenizer, dataset_2, name='OpenThoughts2')

dataset = concatenate_datasets([dataset_0, dataset_1, dataset_2]).shuffle(seed=SEED).select(range(args.data_num))
dataset = dataset.train_test_split(train_size=0.95, seed=SEED)

train_dataset = dataset['train']
test_dataset = dataset['test']

train_loader = DataLoader(
    train_dataset, batch_size=train_config["bs"], shuffle=True,
    num_workers=train_config["num_workers"], pin_memory=True,
    collate_fn=DataCollatorWithPadding()
)
test_loader = DataLoader(
    test_dataset, batch_size=train_config["bs"], shuffle=False,
    num_workers=train_config["num_workers"], pin_memory=True,
    collate_fn=DataCollatorWithPadding()
)

target_model = AutoModelForCausalLM.from_pretrained(args.basepath, torch_dtype=torch.float16)
target_model.eval()
for param in target_model.parameters():
    param.requires_grad = False

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

with open(train_config["config_path"]) as f:
    config = json.load(f)
assert config.get("use_adapter", False) == args.use_adapter

model = LargeDrafter(config, load_emb=True, path=args.basepath, hass_path=args.hass_path)
if accelerator.is_main_process:
    logger.info(model)

if args.ckpt_path is not None:
    ea_model_path = args.ckpt_path
    load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
    if os.path.exists(load_model_path):
        ea_layer_state_dict = torch.load(load_model_path, map_location="cuda")
    else:
        load_model_path = os.path.join(ea_model_path, "model.safetensors")
        ea_layer_state_dict = safetensors.torch.load_file(load_model_path)
    model.load_state_dict(ea_layer_state_dict, strict=True)
    print(f"load model from {load_model_path}")

criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    model, head, target_model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, target_model, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, target_model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, target_model, optimizer, train_loader, test_loader
    )

# work-around: DDP 不支持 嵌套地gradient checkpoint计算
model._set_static_graph()
unwarped_model = accelerator.unwrap_model(model)


@torch.no_grad()
def data_prepare(input_ids, attention_mask, loss_mask):
    def padding(tensor, left=True):
        zeropadding = torch.zeros_like(tensor[:, -1:])
        if left:
            tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
        else:
            tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
        return tensor

    outs = target_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outs.hidden_states[-1]

    input_ids = padding(input_ids, left=False)

    target = hidden_states
    target = padding(target, left=False)

    return {
        "hidden_states": hidden_states,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target": target,
        "loss_mask": loss_mask
    }


for epoch in range(num_epochs + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):

        if args.debug and batch_idx > 10:
            break

        with accelerator.accumulate(model):
            optimizer.zero_grad()

            data = data_prepare(data["input_ids"], data["attention_mask"], data["loss_mask"])
            hidden_states, input_ids, attention_mask, target, loss_mask = data["hidden_states"], data["input_ids"], \
                data["attention_mask"], data["target"], data["loss_mask"][..., None]

            loss = 0
            with torch.no_grad():
                target_head = head(target)
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()

            q_hidden_states = None  ### q hidden states is used to store past step's hidden states
            unwarped_model.reset_step()
            for forward_idx in range(args.forward_num_total):  ### forward for multiple times
                assert forward_idx == unwarped_model.current_step
                predict, sample_hidden = model(hidden_states, input_ids, attention_mask,
                                               q_hidden_states=q_hidden_states)  ### for me. just enable the model to switch paratmers is enough

                if q_hidden_states is None:
                    q_hidden_states = torch.cat([hidden_states[:, :1, :], predict[:, :-1, :]], dim=1)[None, :, :, :]
                    ### see here, q_hidden states is built by using the first in the sequence and add the predict into the sequence(discard the last), then it's the same size as the input hidden_states
                    ### then an additional dimension is added
                else:
                    new_q_hidden_states = torch.cat([q_hidden_states[-1][:, :1, :], predict[:, :-1, :]], dim=1)[None, :,
                    :, :]
                    q_hidden_states = torch.cat([q_hidden_states, new_q_hidden_states], dim=0)
                    ### q_hidden_states always maintains the hidden states of different generation steps

                if not args.train_LD:
                    q_hidden_states = q_hidden_states.detach()
                ### see here, the gradient is detached

                if not args.use_adapter:
                    vloss, ploss, topk_loss, out_head = compute_loss(target, target_p, predict, loss_mask)
                else:
                    vloss, ploss, topk_loss, out_head = compute_loss(target, target_p, sample_hidden, loss_mask)
                total_loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss + train_config[
                    "topk_w"] * topk_loss
                loss += total_loss

                if not args.train_LD:
                    accelerator.backward(total_loss)

            # in LD train, loss is backwarded after all forward steps is done
            if args.train_LD:
                accelerator.backward(loss)

            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            optimizer.zero_grad()
            loss /= args.forward_num_total
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/topkloss": topk_loss.item(), "train/loss": loss.item(),
                       "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)
            # for id,i in enumerate(top_3acc):
            #     wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    if (epoch + 1) % train_config["save_freq"] == 0:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()

        k_acc = [[] for i in range(5)]
        for batch_idx, data in enumerate(tqdm(test_loader)):

            if args.debug and batch_idx > 10:
                break

            data = data_prepare(data["input_ids"], data["attention_mask"], data["loss_mask"])

            with torch.no_grad():
                if batch_idx < 1:
                    acces = getkacc(model, data, head, max_length=5)
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])

                q_hidden_states = None
                unwarped_model.reset_step()
                for forward_idx in range(args.forward_num_total):
                    assert forward_idx == unwarped_model.current_step
                    predict, sample_hidden = model(data["hidden_states"], input_ids=data["input_ids"],
                                                   attention_mask=data["attention_mask"],
                                                   q_hidden_states=q_hidden_states)
                    if q_hidden_states is None:
                        q_hidden_states = torch.cat([data["hidden_states"][:, :1, :], predict[:, :-1, :]], dim=1)[None,
                        :, :, :]
                    else:
                        new_q_hidden_states = torch.cat([q_hidden_states[-1][:, :1, :], predict[:, :-1, :]], dim=1)[
                            None, :, :, :]
                        q_hidden_states = torch.cat([q_hidden_states, new_q_hidden_states], dim=0)

                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                loss_mask = data["loss_mask"][:, :, None]

                if not args.use_adapter:
                    vloss, ploss, topk_loss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
                else:
                    vloss, ploss, topk_loss, out_head = compute_loss(data["target"], target_p, sample_hidden, loss_mask)

                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss + train_config["topk_w"] * topk_loss

                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1

        mean_acces = []
        for id, i in enumerate(k_acc):
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
            mean_acces.append(mean_acc)

        mean_acces = accelerator.gather_for_metrics(mean_acces)
        if accelerator.is_local_main_process:
            for id, i in enumerate(mean_acces):
                mean_acc = i.mean().item()
                wandb.log({f"test/{id}_acc": mean_acc})

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
