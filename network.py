import torch
from torch import nn
from transformers import *

class AlbertPreTrainedModel(PreTrainedModel):
   config_class = AlbertConfig
   pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
   base_model_prefix = "albert"

   def _init_weights(self, module):
       if isinstance(module, (nn.Linear, nn.Embedding)):
           module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
           if isinstance(module, (nn.Linear)) and module.bias is not None:
               module.bias.data.zero_()
       elif isinstance(module, nn.LayerNorm):
           module.bias.data.zero_()
           module.weight.data.fill_(1.0)

class AlbertForAIViVN(AlbertPreTrainedModel):
   def __init__(self, config):
       super(AlbertForAIViVN, self).__init__(config)
       self.num_labels = config.num_labels
       self.albert = AlbertModel(config)
       self.qa_outputs = nn.Linear(4*config.hidden_size, self.num_labels)

       self.init_weights()

   def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

       outputs = self.albert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
       cls_output = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
       logits = self.qa_outputs(cls_output)
       return logits

class BertForAIViVN(BertPreTrainedModel):
   def __init__(self, config):
       super(BertForAIViVN, self).__init__(config)
       self.num_labels = config.num_labels
       self.bert = BertModel(config)
       self.qa_outputs = nn.Linear(4*config.hidden_size, self.num_labels)

       self.init_weights()

   def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

       outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
       cls_output = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
       logits = self.qa_outputs(cls_output)
       return logits

class RobertaForAIViVN(BertPreTrainedModel):
   config_class = RobertaConfig
   pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
   base_model_prefix = "roberta"
   def __init__(self, config):
       super(RobertaForAIViVN, self).__init__(config)
       self.num_labels = config.num_labels
       self.roberta = RobertaModel(config)
       self.qa_outputs = nn.Linear(4*config.hidden_size, self.num_labels)
       self.norm = Norm(4*config.hidden_size)

       self.init_weights()

   '''
   https://huggingface.co/transformers/glossary.html#attention-mask
   https://huggingface.co/transformers/glossary.html#position-ids
   https://huggingface.co/transformers/glossary.html#token-type-ids
   '''
   def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
       # phoBERT's input is 2D tensor (batch_size, max_seq_len)
       outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
#                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
       # ouputs[0]: last hidden layer
       # outputs[1]: unknow :(
       # ouputs[2]: all hidden layers
       # get [CLS] of 4 last hidden layer
       # [:,0,:] = [batch_size, timestep_0, hidden_size]
       cls_output = torch.cat((outputs[2][-1][:,0,:],outputs[2][-2][:,0,:], outputs[2][-3][:,0,:], outputs[2][-4][:,0,:]), -1)
       cls_output_norm = self.norm(cls_output)
       logits = self.qa_outputs(cls_output_norm)
       return torch.sigmoid(logits).view(-1)


# batch normalized layer
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisationa
        # model will return alpha and bias when call parameter() method
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


if __name__ == '__main__':
    x = Norm(128)
    xx = list(x.named_parameters())
    print(xx)
    xxx = list(x.parameters())
    print(xxx)