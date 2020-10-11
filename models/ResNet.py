import torch
from torch import nn as nn
from torch.nn import functional as F
"""
Collected and Adapted from PyTorch Official Implementation of ResNet

"""

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""

	return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()

		norm_layer = nn.BatchNorm3d
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=256,groups=1, width_per_group=64):

		super(ResNet, self).__init__()

		self._norm_layer = nn.BatchNorm3d

		self.inplanes = 64
		self.dilation = 1
		norm_layer=self._norm_layer

		self.groups=groups
		self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
		self.fc = nn.Linear(256, num_classes)



	def _make_layer(self, block, planes, blocks, stride=1,downsample=None):

		norm_layer = self._norm_layer
		previous_dilation = self.dilation

		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))

		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		xa = self.conv1(x)
		x = self.bn1(xa)
		x = self.relu(x)
		xb = self.maxpool(x)

		xc = self.layer1(xb)
		xd = self.layer2(xc)

		xe = self.layer3(xd)
		xf = self.layer4(xe)

		x = self.avgpool(xf)
		x = torch.flatten(x, 1)
		#x = self.fc(x)
		# store={'xa':xa,'xb':xb,'xc':xc,'xd':xd,'xe':xe,'xf':xf}
		return x

def resnet18(**kwargs):

	return ResNet(BasicBlock, [2, 2, 2, 2],**kwargs)


if __name__=='__main__':
	print("Loading 3D ResNet18")
	print("-------------------")
	model=resnet18()
	t=torch.rand(1,1,32,32,32)
	feature_embed=model(t)[1]['xa'].shape
	print(feature_embed)
