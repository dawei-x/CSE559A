import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from tqdm import tqdm
from scipy.special import softmax 
import torch.backends.cudnn as cudnn
import random
import pickle
import os


def create_model(model_name):
    if model_name == 'ResNet18':
        model = torchvision.models.resnet18(weights = "IMAGENET1K_V1")
    elif model_name == 'ResNet50':
        model = torchvision.models.resnet50(weights = "IMAGENET1K_V1")
    elif model_name == 'ResNet101':
        model = torchvision.models.resnet101(weights = "IMAGENET1K_V1")
    elif model_name == 'ResNet152':
        model = torchvision.models.resnet152(weights = "IMAGENET1K_V1")
    elif model_name == 'ResNeXt101':
        model = torchvision.models.resnext101_32x8d(weights = "IMAGENET1K_V1")
    elif model_name == 'VGG16':
        model = torchvision.models.vgg16(weights = "IMAGENET1K_V1")
    elif model_name == 'ShuffleNet':
        model = torchvision.models.shufflenet_v2_x1_0(weights = "IMAGENET1K_V1")
    elif model_name == 'Inception':
        model = torchvision.models.inception_v3(weights = "IMAGENET1K_V1")
    elif model_name == 'DenseNet161':
        model = torchvision.models.densenet161(weights = "IMAGENET1K_V1")
    
    model.eval()
    model = torch.nn.DataParallel(model).cuda()
    return model



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def validate(val_loader, model, print_bool):
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            output, S = model(x.cuda())
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S, target)

            # Update meters
            top1.update(prec1.item()/100.0, n=x.shape[0])
            top5.update(prec5.item()/100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
    if print_bool:
        print('') #Endline

    return top1.avg, top5.avg, coverage.avg, size.avg 

def coverage_size(S,targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            covered += 1
        size = size + S[i].shape[0]
    return float(covered)/targets.shape[0], size/targets.shape[0]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Computes logits and targets from a model and loader
def get_logits_targets(model, loader):
    logits = torch.zeros((len(loader.dataset), 1000)) # 1000 classes in Imagenet.
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.cuda()).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]
    
    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return dataset_logits


def get_logits_dataset(modelname, datasetname, datasetpath, cache='/content/experiments/.cache/'):
    fname = cache + datasetname + '/' + modelname + '.pkl' 

    # If the file exists, load and return it.
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    # Else we will load our model, run it on the dataset, and save/return the output.
    model = create_model(modelname)

    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std =[0.229, 0.224, 0.225])
                    ])
    
    dataset = torchvision.datasets.ImageFolder(datasetpath, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=False, pin_memory=True)

    # Get the logits and targets
    dataset_logits = get_logits_targets(model, loader)

    # Save the dataset 
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as handle:
        pickle.dump(dataset_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_logits

def sort_sum(scores):
    I = scores.argsort(axis=1)[:,::-1]
    ordered = np.sort(scores,axis=1)[:,::-1]
    cumsum = np.cumsum(ordered,axis=1) 
    return I, ordered, cumsum


def split2(dataset, n1, n2):
    data1, temp = torch.utils.data.random_split(dataset, [n1, dataset.tensors[0].shape[0]-n1])
    data2, _ = torch.utils.data.random_split(temp, [n2, dataset.tensors[0].shape[0]-n1-n2])
    return data1, data2


### MAIN MODEL AND PIPELINE

class ConformalModelLogits(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, naive=False, batch_size=32):
        super(ConformalModelLogits, self).__init__()
        self.model = model 
        self.alpha = alpha
        self.randomized = randomized
        self.T = platt_logits(self, calib_loader)

        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        self.Qhat = 1-alpha
        if not naive:
          self.penalties[:, kreg:] += lamda
          self.Qhat = conformal_calibration_logits(self, calib_loader)
        

    def forward(self, logits, randomized=None):
        if randomized == None:
            randomized = self.randomized
        
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy/self.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties, randomized=randomized)

        return logits, S

def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 


def conformal_calibration_logits(cmodel, calib_loader):
    with torch.no_grad():
        E = np.array([])
        for logits, targets in calib_loader:
            logits = logits.detach().cpu().numpy()

            scores = softmax(logits/cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum,penalties=cmodel.penalties,randomized=True)))
            
        Qhat = np.quantile(E,1-cmodel.alpha, method = 'higher')

        return Qhat 


### CORE CONFORMAL INFERENCE FUNCTIONS

# Generalized conditional quantile function.
def gcq(scores, tau, I, ordered, cumsum, penalties, randomized):
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1]) # 1-1000

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1/ordered[i,sizes_base[i]-1] * \
                    (tau-(cumsum[i,sizes_base[i]-1]-ordered[i,sizes_base[i]-1])-penalties_cumsum[0,sizes_base[i]-1]) # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1] # always predict max size if alpha==0. (Avoids numerical error.)

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i,0:sizes[i]],]

    return S

# Get the 'p-value'
def get_tau(score, target, I, ordered, cumsum, penalty, randomized): # For one example
    idx = np.where(I==target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]
    
    U = np.random.random()

    if idx == (0,0):
        return U * tau_nonrandom + penalty[0] 
    else:
        return U * ordered[idx] + cumsum[(idx[0],idx[1]-1)] + (penalty[0:(idx[1][0]+1)]).sum()

# Gets the histogram of Taus. 
def giq(scores, targets, I, ordered, cumsum, penalties, randomized):
    """
        Generalized inverse quantile conformity score function.
        E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(scores[i:i+1,:],targets[i].item(),I[i:i+1,:],ordered[i:i+1,:],cumsum[i:i+1,:],penalties[0,:],randomized=randomized)

    return E


