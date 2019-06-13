import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as models


# modulo para medir la diferencia de contenido
class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # El detach es importante porque la imagen la pasamos por la red
        # sin el detach intenta backpropagar la imagen
        self.target = target.detach()
        self.loss = 0

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    batch_size, channels, a, b = input.size()
    # (a,b) = dimensiones

    features = input.view(batch_size * channels, a * b)

    G = torch.mm(features, features.t())  # producto de gram

    return G.div(batch_size * channels * a * b)


# modulo para medir la diferencia de estilo
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# modulo que normaliza la imagen de entrada
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view para que el número de dimensiones permita operar con el tensor de la imagen
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):

    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # empezamos con una capa que normaliza con los valores a los que vgg está acostumbrada
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # para iterar facilmente sobre la pérdida
    content_losses = []
    style_losses = []

    # aquí vamos a ir poniendo los modulos en orden
    model = nn.Sequential(normalization)

    i = 0  # contador de capas de convolución
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # Si tiene inplace eso puede hacer que nuestras capas no funcionen
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # se quita toda la parte de vgg posterior al último layer de pérdida
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer




