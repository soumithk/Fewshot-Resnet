import torch
import math
import torch.nn as nn

from torch.nn import functional as F

from layers import *


# code from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResNet(nn.Module):
    def __init__(self, block, layers, in_c, num_classes, return_embedding = False):
        super(ResNet, self).__init__()

        self.return_embedding = return_embedding
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def set_return_embedding(val) :
        self.return_embedding = val

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.return_embedding :
            x = torch.flatten(x, 1)
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# code from https://github.com/BoyuanJiang/matching-networks-pytorch/blob/master/matching_networks.py
class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set embeddings and the target image embeddings.
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, query_set):
        """
        forward implement
        :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
        :param input_image: the embedding of the target image,shape[batch_size,64]
        :return:shape[batch_size,sequence_length]
        """
        eps = 1e-10
        sum_support = torch.sum(torch.pow(support_set, 2), 1)
        support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
        # dot_product = support_set.unsqueeze(1).bmm(input_images.unsqueeze(2)).squeeze()
        dot_product = input_images.unsqueeze(0).bmm(support_set.t().unsqueeze(0)).squeeze(0)
        similatities = dot_product * support_manitude
        logits = F.softmax(similarites, 1)
        return logits

        # similarities = []
        # for support_image in support_set:
        #     sum_support = torch.sum(torch.pow(support_image, 2), 1)
        #     support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
        #     dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
        #     cosine_similarity = dot_product * support_manitude
        #     similarities.append(cosine_similarity)
        # similarities = torch.stack(similarities)
        # return similarities.t()

# code from https://github.com/BoyuanJiang/matching-networks-pytorch/blob/master/matching_networks.py
# class AttentionalClassify(nn.Module):
#     def __init__(self):
#         super(AttentionalClassify, self).__init__()
#
#     def forward(self, similarities, support_set_y):
#         """
#         Products pdfs over the support set classes for the target set image.
#         :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
#         :param support_set_y:[batch_size,sequence_length,classes_num]
#         :return: Softmax pdf shape[batch_size,classes_num]
#         """
#         softmax = nn.Softmax()
#         softmax_similarities = softmax(similarities)
#         preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
#         return preds
#
class MatchingNet(nn.Module):

    def __init__(self, base_model, args):

        self.base_model = base_model
        self.distance_network = DistanceNetwork()
        self.attention_classify = AttentionalClassify()
        self.args = args

    def forward(data):
        x = self.base_model(data)
        support_set = x[ : self.args.train_way * self.args.shot]
        query_set = x[self.args.train_way * self.args.shot : ]
        similarites = self.DistanceNetwork()
