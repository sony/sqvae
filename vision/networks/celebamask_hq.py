from networks.net_64 import EncoderVqResnet64Label, DecoderVqResnet64Label


class EncoderVq_resnet_label(EncoderVqResnet64Label):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_resnet_label, self).__init__(dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "CelebAMask_HQ"


class DecoderVq_resnet_label(DecoderVqResnet64Label):
    def __init__(self, dim_z, cfgs, flg_bn):
        super(DecoderVq_resnet_label, self).__init__(dim_z, cfgs, cfgs.act_decoder, flg_bn)
        self.dataset = "CelebAMask_HQ"

