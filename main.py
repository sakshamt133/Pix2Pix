import torch
from generator import Generator
from discriminator import PatchGAN


def save_model(save):
    if save is False:
        return

    torch.save({
        'GENERATOR_WEIGHTS': gan.state_dict(),
        'OPTIMIZER_DICT': gan_opt.state_dict()
    }, 'model.pth')


img_channels = 3
epochs = 5
dis = PatchGAN(img_channels)
gan = Generator(img_channels)
dis_opt = torch.optim.Adam(dis.parameters(), lr=0.001)
gan_opt = torch.optim.Adam(gan.parameters(), lr=0.001)
loss1 = torch.nn.L1Loss()
adv_loss = torch.nn.BCELoss()
saving = True


# Dummmy Data
real_input = torch.randn((1, 3, 26, 26))
real_output = torch.randn((1, 3, 26, 26))


# Training
for epoch in range(epochs):
    fake = gan(real_input)
    print(f"fake shape is {fake.shape}")

    for _ in range(3):

        dis_real = dis(real_input, real_output)
        dis_fake = dis(real_input, fake)

        dis_real_loss = adv_loss(dis_real, torch.ones_like(dis_real))
        dis_fake_loss = adv_loss(dis_fake, torch.zeros_like(dis_fake))

        dis_loss = torch.mean(dis_fake_loss) + torch.mean(dis_real_loss)
        print(f'loss is {dis_loss}')
        dis_loss.backward(retain_graph=True)

        dis_opt.step()
        dis_opt.zero_grad()

    dis_gan = dis(real_input, fake)
    gan_main_loss = adv_loss(dis_gan, torch.ones_like(dis_gan))
    gan_pixel_loss = loss1(fake, real_output)
    gan_loss = gan_main_loss + gan_pixel_loss
    gan_loss.backward()
    print(f"gan loss is {gan_loss}")

    gan_opt.step()
    gan_opt.zero_grad()

save_model(saving)

