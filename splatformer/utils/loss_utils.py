class lpips_loss_fn():
    def __init__(self):
        import lpips
        self.lpips = lpips.LPIPS(net='vgg').cuda()
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

    def __call__(self, x, y):


        loss = self.lpips(x.permute(0,3,1,2), y.permute(0,3,1,2), normalize=True)
        return loss