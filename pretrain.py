import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from samplers import LinearSampler, CategoriesSampler
from mini_imagenet import MiniImageNet
from models import ResNet
from layers import Bottleneck
from utils import *

def save_model(save_path, name):
    torch.save(model.state_dict(), osp.join(save_path, name + '.pth'))

def pretrain(args) :

    set_gpu(args.gpu)
    ensure_path(args.save_path)

    trainset = MiniImageNet(["train", "val"])
    train_sampler = LinearSampler(args.batch_size, len(trainset))
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)
    print("No of classes: ", len(trainset)/600)
    model = ResNet(Bottleneck, [2, 2, 2, 2], 3, int(len(trainset)/600)).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1) :

        lr_scheduler.step()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1) :

            imgs, labels = batch
            # imgs = [_.cuda() for _ in imgs]; labels = labels.cuda();
            imgs = imgs.cuda(); labels = labels.cuda();
            feats = model(imgs)
            logits = F.softmax(feats, dim = 1)
            loss = F.cross_entropy(logits, labels)
            acc = count_acc(logits, labels)
            tl.add(loss.item())
            ta.add(acc)
            if i % 20 == 0 :
                print("logits", torch.argmax(logits, dim=1))
                print("labels", labels)
                print("")
                print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                      .format(epoch, i, len(train_loader), loss.item(), acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(epoch, tl.item(), ta.item()))
        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    save_model('pretrain')



if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-path', default='./save/proto-1')
    args = parser.parse_args()
    pprint(vars(args))

    pretrain(args)
