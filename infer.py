import pandas as pd
from network import *
from tqdm import tqdm
tqdm.pandas()
from transformers import *
import torch.utils.data
import argparse
from transformers.modeling_utils import *
from vncorenlp import VnCoreNLP
from utils import * 

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('--test_path', type=str, default='./data/test.csv')
parser.add_argument('--rdrsegmenter_path', type=str, default='/home/nobita/vncorenlp/VnCoreNLP-1.1.1.jar')
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--ckpt_path', type=str, default='./models')
parser.add_argument('-no_cuda', action='store_true')

args = parser.parse_args()

args.device = 'cuda' if args.no_cuda is False else 'cpu'
if args.device == 'cuda':
    assert torch.cuda.is_available()

tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base')
rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m') 

# Load model
model_bert = RobertaForAIViVN.from_pretrained('vinai/phobert-base',
                                              output_hidden_states=True,
                                              num_labels=1)
model_bert.to(args.device)

if torch.cuda.device_count():
    print(f"Testing using {torch.cuda.device_count()} gpus")
    model_bert = nn.DataParallel(model_bert)
    tsfm = model_bert.module.roberta
else:
    tsfm = model_bert.roberta

test_df = pd.read_csv(args.test_path,sep='\t').fillna("###")
test_df.text = test_df.text.progress_apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
X_test = convert_lines(test_df, tokenizer, args.max_sequence_length)

preds_en = []
for fold in range(5):
    print(f"Predicting for fold {fold}")
    preds_fold = []
    # load model
    model_bert.load_state_dict(torch.load(os.path.join(args.ckpt_path, f"model_{fold}.bin")))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test,dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model_bert.eval()
    pbar = tqdm(enumerate(test_loader),total=len(test_loader),leave=False)
    for i, (x_batch,) in pbar:
        y_pred = model_bert(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
        y_pred = y_pred.view(-1).detach().cpu().numpy()
        preds_fold = np.concatenate([preds_fold, y_pred])
    preds_fold = sigmoid(preds_fold)
    preds_en.append(preds_fold)
preds_en = np.mean(preds_en,axis=0)
test_df["label"] = (preds_en > 0.5).astype(np.int)
test_df[["id","label"]].to_csv("submission.csv")
