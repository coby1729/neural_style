import torch
import os.path

from util import show_images, convert_to_image, save, image_loader
from model import get_style_model_and_losses, get_input_optimizer

# sustituir con rutas a imágenes
content_path = ""
style_path = ""
out_path = "\\{}_{}".format(
    os.path.basename(content_path).split(".")[0],
    os.path.basename(style_path).split(".")[0]
)


style_layers = ['conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12']
content_layers = ['conv_5']

if not os.path.isdir(out_path):
    os.mkdir(out_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tamaño de la imagen, si no hay gpu tiene que ser pequeñita para no tirarse años
image_size = 512 if torch.cuda.is_available() else 128

content_image = image_loader(content_path, image_size).to(device, torch.float)
style_image = image_loader(style_path, image_size).to(device, torch.float)

input_image = content_image.clone()


def run_style_transfer(content_image, style_image, input_image,
                       num_steps=300,
                       style_weight=1000000, content_weight=1,
                       path=""):

    vgg_model, style_losses, content_losses = get_style_model_and_losses(style_image, content_image,
                                                                         style_layers=style_layers,
                                                                         content_layers=content_layers)
    optimizer = get_input_optimizer(input_image)

    i = [0]
    input_image.data.clamp_(0, 1)

    if path != "":
        path += "\\{}.jpg"
        save(input_image, path.format(i[0]))

    while i[0] <= num_steps:

        def closure():
            optimizer.zero_grad()
            vgg_model(input_image)
            style_score = 0
            content_score = 0

            for s in style_losses:
                style_score += s.loss
            for c in content_losses:
                content_score += c.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            # puede darse que la imagen haya acabado con píxeles fuera de [0,1], esto lo arregla
            input_image.data.clamp_(0, 1)
            i[0] += 1

            if i[0] % 50 == 0:
                print("iteraciones: {}:".format(i[0]))
                print('Diferencia de estilo : {:4f} Diferencia de contenido: {:4f}'.format(
                    style_score.item(), content_score.item()))
                if path != "":
                    save(input_image, path.format(i[0]))
                print()

            return style_score + content_score

        optimizer.step(closure)

    return input_image


output = run_style_transfer(content_image, style_image, input_image,
                            num_steps=500, style_weight=1000000,
                            path=out_path)

show_images([output], [])
