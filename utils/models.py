import torch
import torch.nn as nn

# ─────────────────────────────
# 훈련,평가용 유틸
# ─────────────────────────────
from torch.utils.data import DataLoader

def get_optimizer(model):
    decay, no_decay = [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        (no_decay if p.ndim==1 or "bias" in n or "norm" in n.lower() else decay).append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-3
    )
    return optimizer

import copy
class Call_back:
    def __call__(self, context:dict)->bool:
        pass
class Save_best_model(Call_back):
    def __init__(self, save_dir=None):
        super().__init__()
        self.best_score = None
        self.best_model_state = None
        self.save_dir = save_dir
    def __call__(self, context):
        if self.best_score == None or context['val_loss'] < self.best_score:
            print('saving best model...')
            self.best_score = context['val_loss']
            self.best_model_state = copy.deepcopy(context['model'].state_dict())
            if self.save_dir != None:
                torch.save(self.best_model_state, self.save_dir)
            return True
    def get_best_model_state(self):
        return self.best_model_state
class Early_stop(Call_back):
    def __init__(self, max_cnt = 3):
        super().__init__()
        self.max_cnt = max_cnt
        self.prev_loss = None
        self.cur_cnt = None
    def __call__(self, context):
        if self.prev_loss == None:
            self.prev_loss = context['val_loss']
            self.cur_cnt = 0
            return True
        if context['val_loss'] >= self.prev_loss:
            self.cur_cnt += 1
            if self.cur_cnt >= self.max_cnt:
                print('Early stopped...')
                return False
        else:
            self.cur_cnt = 0
            self.prev_loss = context['val_loss']
            return True
        



def train_model(model, train_data, validation_data, optimizer, collator, criterion, batch=16, num_epochs=5, device = 'cuda', call_backs=[]):
    train_loader = DataLoader(
        train_data,       
        batch_size=batch,#16,32,128              
        shuffle=True,               
        collate_fn=collator        
    )

    for epoch in range(num_epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["generated"].float().to(device)      # (N,) float 0/1

            optimizer.zero_grad(set_to_none=True)
            out = model(ids, attn)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            cur_batch_size = ids.size(0)
            total += loss.item()*cur_batch_size
            n += cur_batch_size
            train_loss = total / max(n,1)

        print(f'epoch {epoch+1} end')
        print(f'train_loss: {train_loss}')
        if validation_data != None:
            val_loss = evaluate(model, validation_data, criterion, collator, device)
            print(f'val_loss: {val_loss}')
            for call_back in call_backs:
                if not call_back({'model':model, 'train_loss':train_loss, 'val_loss':val_loss}):
                    break
        print('============================================================')

from torch import sigmoid
def get_preds(model, data, collator, postprocess=sigmoid, device='cuda'):
    loader = DataLoader(
        data,
        batch_size = 64,
        shuffle=False,
        collate_fn=collator
    )
    pred = []
    target = []
    if postprocess == None:
        postprocess = lambda x: x

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets        = batch["generated"].float().to(device)  

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            out = postprocess(out)
            for o in out:
                pred.append(o.item())
            for t in targets:
                target.append(t.item())
    return pred, target
            

def evaluate(model, validation_data, criterion, collator, device='cuda'):
    model.eval()
    preds, label = get_preds(model, validation_data, collator, None, device)
    if isinstance(preds, list):
        preds = torch.Tensor(preds)
    if isinstance(label, list):
        label = torch.Tensor(label)
    loss = criterion(preds, label)
    return loss.item()
def get_accuracy(model, validation_data, thres, collator, device='cuda'):
    model.eval()
    preds, label = get_preds(model, validation_data, collator, device)
    preds = [p >= thres for p in preds]
    rets = [p==l for p, l in zip(preds,label)]
    return sum(rets)/len(rets)
from sklearn.metrics import accuracy_score, f1_score
def get_metric(y_pred, y_true, thres, metric='accuracy'):
    y_pred_label = [float(y >= thres) for y in y_pred]
    if metric == 'accuracy':
        return accuracy_score(y_true=y_true, y_pred=y_pred_label)
    if metric == 'f1_score':
        return f1_score(y_true=y_true, y_pred=y_pred_label)






# ─────────────────────────────
# 특수토큰을 마스킹함
# ─────────────────────────────
def mask_special_bert_token(input_ids, attention_mask, tokenizer):
    # input_ids: (N, L), attention_mask: (N, L)
    # PAD은 이미 attention_mask가 0
    special_ids = set([
        tokenizer.cls_token_id, tokenizer.sep_token_id
    ]) - {None}
    # 특수토큰 제외
    is_special = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        is_special |= (input_ids == sid)
    mask = attention_mask.bool() & (~is_special)
    return mask  # (N,L) in {0,1}
# ─────────────────────────────
# llm의 임베딩벡터를 Conv1dBlock의 입력에 맞게 변환함: (배치, 길이, 차원) -> (배치, 차원, 길이)
# ─────────────────────────────
def llm_embed_to_conv1d(x:torch.Tensor):
    return x.transpose(1,2)

#################################################################후처리(after_embed)
# ─────────────────────────────
# 패딩에 해당하는 부분을 0으로 함
# ─────────────────────────────
def zeroing_mask(x, mask):
    # x: (N, C, L), mask: (N, L) in {0,1}
    m = mask.unsqueeze(1).float()               # (N,1,L)
    return x * m                      

# ─────────────────────────────
# 길이 축(L)에 대해 마스크 가중 전역 평균 풀링 (Conv 출력: N,C,L -> N,C)
# ─────────────────────────────
def masked_global_avgpool(x, mask):
    # x: (N, C, L), mask: (N, L) in {0,1}
    m = mask.unsqueeze(1).float()               # (N,1,L)
    s = (x * m).sum(dim=2)                      # (N, C)
    d = m.sum(dim=2).clamp_min(1.0)             # (N,1)
    return s / d
def normalize_to_length1(x, mask):
    """
    x: (batch, channels, length)
    벡터의 방향만 남김
    """
    norm = (x.pow(2).sum(dim=1, keepdim=True).sqrt()).clamp_min(1e-12)
    return x / norm
###########################################################################
# ─────────────────────────────
# 마스크 가중 MSE, masked된 부분은 MSE계산에서 제외함
# pred/target: (N, C, L), mask: (N, L)
# 반환: per_sample(mse) (N,)
# ─────────────────────────────
def masked_mse(pred, target, mask):
    m = mask.unsqueeze(1).float()             # (N, 1, L)
    diff2 = (pred - target).pow(2) * m        # (N, C, L)
    num = diff2.sum(dim=(1, 2))               # (N,)
    den = m.sum(dim=(1, 2)).clamp_min(1.0)    # (N,)
    return num / den                          # (N,)

# ─────────────────────────────
# 
# ─────────────────────────────
class LayerNormOverC(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c, bias=False)
    def forward(self, x):            # x: (N,C,L)
        return self.ln(x.transpose(1,2)).transpose(1,2)
#########################################################################conv1d블록
class Normalize_base(nn.Module):
    def __init__(self, cin, cout, normalize_type='instance'):
        super().__init__()

        self.normalize_type = normalize_type

        if normalize_type == 'batch':
            self.norm = nn.BatchNorm1d(num_features=cout)
        elif normalize_type == 'layer':
            self.norm = LayerNormOverC(cout)
        elif normalize_type == 'instance':
            self.norm = nn.InstanceNorm1d(cout)
        elif normalize_type == None:
            self.norm = lambda x: x
        self.act = nn.ReLU(inplace=True)
# ─────────────────────────────
# 작은 Conv1d 블록 (정규화 + ReLU, 길이 보존)
# ─────────────────────────────
class Conv1dBlock(Normalize_base):
    """
    cin: 입력의 채널수
    cout: 출력의 채널수
    seq_length: 입출력의 길이
    k=커널크기
    dilation: 확장합성곱
    normalize_type: batch, layer
    """
    def __init__(self, cin, cout, k=5, dilation=1, normalize_type='instance'):
        super().__init__(cin,cout,normalize_type)
        pad = ((k - 1) // 2) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel_size=k, padding=pad,
                              dilation=dilation, bias=False)

    def forward(self, x:torch.Tensor):
        #입력은 (배치,임배딩차원,길이)!!!
        x_conv = self.conv(x)
        x_normalized = self.norm(x_conv)
        y = self.act(x_normalized)
        return y


from transformers import AutoModel

# ─────────────────────────────
# 다양한 크기의 커널을 적용후 concat하는 모듈
# 생성자: cin(임베딩차원), cout(각 conv의 출력차원->모두 concat됨)
# 입력: (배치, 임베딩차원, 길이) 데이터
# 출력: concat된거
# ─────────────────────────────
class MultiKernelConv1d(Normalize_base):
    def __init__(self, cin, cout, kernel_sizes=[3,5,7], normalize_type='instance', mode='concat'):
        super().__init__(cin, cout*3, normalize_type)
        self.branches = nn.ModuleList([
            Conv1dBlock(cin, cout, k=k, normalize_type=normalize_type)
            for k in kernel_sizes
        ])
        self.mode = mode
    def forward(self, x:torch.Tensor):            
        outs = [b(x) for b in self.branches]
        if self.mode == 'concat':
            y = torch.cat(outs, dim=1)
        elif self.mode == 'add':
            y = torch.stack(outs, dim=0).sum(dim=0)
        y = self.act(self.norm(y))
        return y

# ─────────────────────────────
# skipconnection 해주는거: concat, add
# 생성자: layers(결과들을 skip연결해줌), skip_type
# 입력: (배치, 임베딩차원, 길이) 데이터
# 출력: concat된거
# ─────────────────────────────
class SkipConnectBlock(nn.Module):
    def __init__(self, layers:nn.ModuleList, skip_type='concat', before=None, after=None):
        super().__init__()
        self.layers = layers
        self.skip_type = skip_type
        self.before = before
        if before == None:
            self.before = lambda x: x
        if after == None:
            self.after = lambda x: x
    def forward(self, x:torch.Tensor):
        ret = x
        for layer in self.layers:
            temp = layer(self.before(ret))
            if self.skip_type == 'concat':
                ret = torch.cat((ret, temp), dim=1)
            elif self.skip_type == 'add':
                ret = ret + temp
            ret = self.after(ret)
        return ret
            
# ─────────────────────────────
# Conv1D+분류기
# 입력: input_ids, attention_mask
# 출력: 로짓(모델의 출력에 시그모이드를 적용해 줘야 됩니다)
# ─────────────────────────────
class BertConv1dBinaryClassifier(nn.Module):
    def __init__(self, base_model_name, freeze_bert, after_conv, after_embbed, tokenizer=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.bert.config.hidden_size
        if freeze_bert:
            for p in self.bert.parameters(): p.requires_grad = False
        self.conv = None
        self.classifier = None
        self.after_conv = after_conv
        self.after_embbed = after_embbed
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        x_llm_embed = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (N,L,H)
        x = x_llm_embed.transpose(1, 2)
        if self.tokenizer != None:
            attention_mask = mask_special_bert_token(input_ids, attention_mask)
        for p in self.after_embbed:
            x = p(x,attention_mask)

        x_conv = x
        for block in self.conv:
            x_conv = block(x_conv)
            for p in self.after_conv:
                x_conv=p(x_conv,attention_mask)


        x_final = masked_global_avgpool(x_conv, attention_mask)            # (N,C)
        logit = self.classifier(x_final).squeeze(-1)                        # (N,)
        return logit

# 초기모델, 각 레이어에서 conv1d 하나만 씀
class BertConv1dBinaryClassifier_Default(BertConv1dBinaryClassifier):
    def __init__(self, base_model_name, freeze_bert=True, 
                 couts=[256,128,64], filter_sizes=[5,5,3], dilations=[1,1,1], 
                 normalize_type='instance', after_conv = [zeroing_mask], after_embbed = [zeroing_mask, normalize_to_length1],
                 tokenizer=None):
        super().__init__(base_model_name, freeze_bert, after_conv, after_embbed, tokenizer)
        self.conv = nn.ModuleList([Conv1dBlock(
                        cin=_cin,
                        cout=_cout,
                        k=_k, 
                        dilation=_dilation, 
                        normalize_type=normalize_type) 
                        for _cin, _cout, _dilation, _k in zip([self.hidden_size]+couts[:-1], couts, dilations, filter_sizes)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(couts[-1]),
            nn.Linear(couts[-1], couts[-1]*2),
            nn.GELU(),
            nn.Linear(couts[-1]*2, 1),  # 한 개 로그릿
        )

# 최종모델, 각 레이어에서 conv1d를 3개씩 씀
class BertConv1dBinaryClassifier_MultiK(BertConv1dBinaryClassifier):
    def __init__(self, base_model_name, freeze_bert=True, couts=[256,128,64], normalize_type='instance', after_conv = [zeroing_mask], after_embbed = [zeroing_mask, normalize_to_length1], tokenizer=None):
        super().__init__(base_model_name, freeze_bert, after_conv, after_embbed, tokenizer)
        cins = [self.hidden_size]+[i*3 for i in couts[:-1]]
        self.conv = nn.ModuleList([MultiKernelConv1d(
                        cin=_cin,
                        cout = _cout,
                        normalize_type= normalize_type)
                        for _cin, _cout in zip(cins, couts)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(couts[-1]*3),
            nn.Linear(couts[-1]*3, couts[-1]*6),
            nn.GELU(),
            nn.Linear(couts[-1]*6, 1),  # 한 개 로그릿
        )