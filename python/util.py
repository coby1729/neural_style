import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


def convert_to_image(tensor):
    image = tensor.clone().detach().cpu()  # clonar pa no cargarse el tensor original
    image = image.squeeze(0)      # quitar la dimensión extra
    unloader = transforms.ToPILImage()  # convertir a algo que se pueda mostrar
    image = unloader(image)
    return image


def show_images(tensors, titles=['Style Image', 'Content Image']):
    plt.figure()
    for i in range(len(tensors)):
        plt.subplot(1, len(tensors), i + 1)
        image = convert_to_image(tensors[i])
        plt.imshow(image)
        if len(titles) > i:
            plt.title(titles[i])
        plt.subplots_adjust(wspace=0.2)
    plt.show()


def save(tensor, path):
    image = convert_to_image(tensor)
    image.save(path)


def image_loader(image_path, image_size):
    image = Image.open(image_path).convert('RGB')

    # transformación para imágenes de entrada
    trans = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()])

    # la nn está hecha para recibir un batch de imágenes
    # con el unsqueeze se convierte en un batch de una (1) imagen
    image = trans(image).unsqueeze(0)
    return image
