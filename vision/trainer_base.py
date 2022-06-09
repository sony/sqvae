import shutil
import json
import datetime
from torch import nn

from model import GaussianSQVAE, VmfSQVAE
from util import *

class TrainerBase(nn.Module):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(TrainerBase, self).__init__()
        self.cfgs = cfgs
        self.flgs = flgs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = eval(
            "nn.DataParallel({}(cfgs, flgs).cuda())".format(cfgs.model.name))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfgs.train.lr, amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3,
            verbose=True, threshold=0.0001, threshold_mode="rel",
            cooldown=0, min_lr=0, eps=1e-08)
    
    def load(self, timestamp=""):
        if timestamp != "":
            self.path = os.path.join(self.cfgs.path, timestamp)
        self.model.load_state_dict(
            torch.load(os.path.join(self.path, "best.pt")))
        self.plots = np.load(
            os.path.join(self.path, "plots.npy"), allow_pickle=True).item()
        print(self.path)
        self.model.eval()
    

    ## Methods for main loop

    def main_loop(self, max_iter=None, timestamp=None):
        if timestamp == None:
            self._make_path()
        else:
            self.path = os.path.join(self.cfgs.path, timestamp)
        BEST_LOSS = 1e+20
        LAST_SAVED = -1

        if max_iter == None:
            max_iter = self.cfgs.train.epoch_max
        for epoch in range(1, max_iter+1):
            myprint("[Epoch={}]".format(epoch), self.flgs.noprint)
            res_train = self._train(epoch)
            if self.flgs.save:
                self._writer_train(res_train, epoch)
            res_test = self._test()
            if self.flgs.save:
                self._writer_val(res_test, epoch)
            
            if self.flgs.save:
                if res_test["loss"] <= BEST_LOSS:
                    BEST_LOSS = res_test["loss"]
                    LAST_SAVED = epoch
                    myprint("----Saving model!", self.flgs.noprint)
                    torch.save(
                        self.model.state_dict(), os.path.join(self.path, "best.pt"))
                    self.generate_reconstructions(
                        os.path.join(self.path, "reconstrucitons_best"))
                else:
                    myprint("----Not saving model! Last saved: {}"
                        .format(LAST_SAVED), self.flgs.noprint)
                torch.save(
                    self.model.state_dict(), os.path.join(self.path, "current.pt"))
                self.generate_reconstructions(
                    os.path.join(self.path, "reconstructions_current"))
    
    def preprocess(self, x, y):
        if self.cfgs.dataset.name == "CelebAMask_HQ":
            y[:, 0, :, :] = y[:, 0, :, :] * 255.0
            y = torch.round(y[:, 0, :, :]).cuda()
        return y
    
    def test(self, mode="test"):
        result = self._test(mode)
        if mode == "test":
            self._writer_test(result)
        return result
    
    def _set_temperature(self, step, param):
        temperature = np.max([param.init * np.exp(-param.decay*step), param.min])
        return temperature

    
    def _save_config(self):
        tf = open(self.path + "/configs.json", "w")
        json.dump(self.cfgs, tf)
        tf.close()
    
    def _train(self):
        raise NotImplementedError()
    
    def _test(self):
        raise NotImplementedError()
    
    def print_loss(self):
        raise NotImplementedError()
    

    ## Visualization    

    def generate_reconstructions(self):
        raise NotImplementedError()
    
    def generate_reconstructions_paper(self, nrows=1, ncols=10, off_set=0):
        self.model.eval()
        x = self.test_loader.__iter__().next()[0]
        x = x[off_set:off_set+nrows*ncols].cuda()
        output = self.model(x, flg_train=False, flg_quant_det=True)
        x_tilde = output[0]
        images_original = x.cpu().data.numpy()
        images_reconst = x_tilde.cpu().data.numpy()
        plot_images_paper(images_original,
            os.path.join(self.path, "paper_original"), nrows=nrows, ncols=ncols)
        plot_images_paper(images_reconst,
            os.path.join(self.path, "paper_reconst"), nrows=nrows, ncols=ncols)

    def _generate_reconstructions_continuous(self, filename, nrows=4, ncols=8):
        self.model.eval()
        x = self.test_loader.__iter__().next()[0]
        x = x[:nrows*ncols].cuda()
        output = self.model(x, flg_train=False, flg_quant_det=True)
        x_tilde = output[0]
        x_cat = torch.cat([x, x_tilde], 0)
        images = x_cat.cpu().data.numpy()
        plot_images(images, filename+".png", nrows=nrows, ncols=ncols)
    
    def _generate_reconstructions_discrete(self, filename, nrows=4, ncols=8):
        self.model.eval()
        x, y = self.test_loader.__iter__().next()
        x = x[:nrows*ncols].cuda()
        y = y[:nrows*ncols].cuda()
        y[:, 0, :, :] = y[:, 0, :, :] * 255.0
        y_long = y
        y = y[:, 0, :, :]
        output = self.model(y, flg_train=False, flg_quant_det=True)
        label_tilde = output[0]
        label_real = idx_to_onehot(y_long)
        label_batch_predict = generate_label(label_tilde[:,:19,:,:], x.shape[-1])
        label_batch_real = generate_label(label_real, x.shape[-1])
        x_cat = torch.cat([label_batch_real, label_batch_predict], 0)
        images = x_cat.cpu().data.numpy()
        plot_images(images, filename+".png", nrows=nrows, ncols=ncols)


    ## Saving

    def _make_path(self):
        import glob
        dt_now = datetime.datetime.now()
        timestamp = dt_now.strftime("%m%d_%H%M")
        self.path = os.path.join(self.cfgs.path, "{}_seed{}_{}".format(
            self.cfgs.network.name, self.cfgs.train.seed,timestamp))
        print(self.path)
        if self.flgs.save:
            self._makedir(self.path)
            list_dir = self.cfgs.list_dir_for_copy
            files = []
            for dirname in list_dir:
                files.append(glob.glob(dirname+"*.py"))
            target = os.path.join(self.path, "codes")
            for i, dirname in enumerate(list_dir):
                if not os.path.exists(os.path.join(target, dirname)):
                    os.makedirs(os.path.join(target, dirname))
                for file in  files[i]:
                    shutil.copyfile(file, os.path.join(target, file))

    def _makedir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            i = 1
            while True:
                path += "_{}".format(i)
                if not os.path.exists(path):
                    os.makedirs(path)
                    break
                print(i)
                i += 1
        self._save_config()
        self.path = path
        
    def _writer_train(self, result, epoch):
        self._append_writer_train(result)
        np.save(os.path.join(self.path, "plots.npy"), self.plots)
    
    def _writer_val(self, result, epoch):
        self._append_writer_val(result)
        np.save(os.path.join(self.path, "plots.npy"), self.plots)
    
    def _writer_test(self, result):
        self._append_writer_test(result)
        np.save(os.path.join(self.path, "plots.npy"), self.plots)
    
    def _append_writer_train(self, result):
        for metric in result:
            self.plots[metric+"_train"].append(result[metric])
        
    def _append_writer_val(self, result):
        for metric in result:
            self.plots[metric+"_val"].append(result[metric])
    
    def _append_writer_test(self, result):
        for metric in result:
            self.plots[metric+"_test"].append(result[metric])

