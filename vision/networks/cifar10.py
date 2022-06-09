from networks.net_32 import EncoderVqResnet32, DecoderVqResnet32

 
class EncoderVq_resnet(EncoderVqResnet32):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_resnet, self).__init__(dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "Cifar10"


class DecoderVq_resnet(DecoderVqResnet32):
    def __init__(self, dim_z, cfgs, flg_bn):
        super(DecoderVq_resnet, self).__init__(dim_z, cfgs, flg_bn)
        self.dataset = "Cifar10"

