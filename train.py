import pandas as pd
from network import *
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import *
import argparse
from transformers.modeling_utils import *
from fairseq.data import Dictionary
from vncorenlp import VnCoreNLP
from utils import *
import torch
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('--train_path', type=str, default='./data/train.csv')
parser.add_argument('--rdrsegmenter_path', type=str, default='/mnt/data/nobita/vncorenlp/VnCoreNLP-1.1.1.jar')
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--accumulation_steps', type=int, default=5)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--ckpt_path', type=str, default='./models')
parser.add_argument('-no_cuda', action='store_true')

args = parser.parse_args()

args.device = 'cuda' if args.no_cuda is False else 'cpu'
if args.device == 'cuda':
    assert torch.cuda.is_available()

tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base')
rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m', port=9000)

seed_everything(2021)

# Load model
model_bert = RobertaForAIViVN.from_pretrained('vinai/phobert-base',
                                              output_hidden_states=True,
                                              num_labels=1)
model_bert.to(args.device)

if torch.cuda.device_count():
    print(f"Training using {torch.cuda.device_count()} gpus")
    model_bert = nn.DataParallel(model_bert)
    tsfm = model_bert.module.roberta
else:
    tsfm = model_bert.roberta

# Load the dictionary  
vocab = Dictionary()
vocab.add_from_file(args.dict_path)

# Load training data
train_df = pd.read_csv(args.train_path,sep='\t').fillna("###")
print('Tokenize training data')
train_df.text = train_df.text.progress_apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
y_train = train_df.label.values
X_train = convert_lines(train_df, tokenizer, args.max_sequence_length)

# Creating optimizer and lr schedulers
param_optimizer = list(model_bert.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = int(args.epochs*len(train_df)/args.batch_size/args.accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
# Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
scheduler0 = get_constant_schedule(optimizer)  # PyTorch scheduler. Create a schedule with a constant learning rate.

if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X_train, y_train))
for fold, (train_idx, val_idx) in enumerate(splits):
    print("Train fold {}".format(fold))
    best_score = 0

    if fold != args.fold:
        continue

    train_dataset = TensorDataset(torch.tensor(X_train[train_idx], dtype=torch.long), torch.tensor(y_train[train_idx], dtype=torch.long))
    valid_dataset = TensorDataset(torch.tensor(X_train[val_idx], dtype=torch.long), torch.tensor(y_train[val_idx], dtype=torch.long))

    # https://stackoverflow.com/questions/52465723/what-is-the-difference-between-parameters-and-children
    for child in tsfm.children(): # init frozen BERT
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = False
    frozen = True

    for epoch in range(args.epochs+1):

        if epoch > 0 and frozen:
            for child in tsfm.children():
                for param in child.parameters():
                    param.requires_grad = True
            frozen = False
            del scheduler0
            if args.device == 'cuda':
                torch.cuda.empty_cache()

        val_preds = []
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        num_training_batches = len(train_loader)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        avg_loss = 0.
        avg_accuracy = 0.

        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=num_training_batches)
        for i, (x_batch, y_batch) in pbar:
            model_bert.train() # enable Bert training mode
            y_pred = model_bert(x_batch.to(args.device), attention_mask=(x_batch != 1).to(args.device))
            # https://medium.com/@zhang_yang/how-is-pytorchs-binary-cross-entropy-with-logits-function-related-to-sigmoid-and-d3bd8fb080e7
            loss = F.binary_cross_entropy(y_pred.to(args.device), y_batch.float().to(args.device))
            loss.backward()
            # only update gradient after args.accumulation_steps batches or last batch
            if i % args.accumulation_steps == 0 or i == num_training_batches - 1:
                optimizer.step()
                optimizer.zero_grad()
                if not frozen:
                    scheduler.step()
                else:
                    scheduler0.step()
            avg_loss += loss.item()
            # y_pred = torch.sigmoid(y_pred).view(-1)
            avg_accuracy += get_accuracy(y_batch, y_pred)
            pbar.set_description('[epoch %d] loss = %.4f - acc = %.4f' % (epoch, avg_loss / (i+1), avg_accuracy / (i+1)))

        model_bert.eval() # enable Bert evaluation mode
        avg_val_loss = 0
        avg_val_accuracy = 0
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for i, (x_batch, y_batch) in pbar:
            y_pred = model_bert(x_batch.to(args.device), attention_mask=(x_batch != 1).to(args.device))
            loss = F.binary_cross_entropy(y_pred.to(args.device), y_batch.float().to(args.device))
            avg_val_loss += loss.item()
            avg_val_accuracy += get_accuracy(y_batch, y_pred)
            pbar.set_description('[epoch %d] val_loss = %.4f - val_acc = %.4f' % (epoch, avg_val_loss / (i + 1), avg_val_accuracy / (i + 1)))

        best_th = 0
        score = avg_val_accuracy / len(valid_loader)
        if score >= best_score:
            torch.save(model_bert.state_dict(),os.path.join(args.ckpt_path, f"model_{fold}.bin"))
            best_score = score
    break