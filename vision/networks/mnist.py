from networks.net_28 import EncoderVqResnet28, DecoderVqResnet28


class EncoderVq_resnet(EncoderVqResnet28):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_resnet, self).__init__(dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "MNIST"


class DecoderVq_resnet(DecoderVqResnet28):
    def __init__(self, dim_z, cfgs, flg_bn):
        super(DecoderVq_resnet, self).__init__(dim_z, cfgs, flg_bn)
        self.dataset = "MNIST"

