from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing


class PrintDot(Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


class DASDataset:
    def __init__(self, metadata, loader_fn, loader_kw, targets,
                 train_metadata=None, val_metadata=None, test_metadata=None,
                 normalize_by=None, normalization_type='std'):
        self.metadata = metadata.copy()
        if train_metadata is not None:
            self.train_metadata = train_metadata.copy()
        else:
            self.train_metadata = metadata.copy()
        if val_metadata is not None:
            self.val_metadata = val_metadata.copy()
        else:
            self.val_metadata = metadata.copy()
        if test_metadata is not None:
            self.test_metadata = test_metadata.copy()
        else:
            self.test_metadata = metadata.copy()
        self._loader = loader_fn
        self._loader_kw = loader_kw
        self._targets = targets
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self._im_size = None
        self.normalization_is_enabled = False
        self.normalize_by = normalize_by
        if self.normalize_by is not None:
            self.metadata[normalize_by] = self.metadata[normalize_by].astype(str)
            self.train_metadata[normalize_by] = self.train_metadata[normalize_by].astype(str)
            self.val_metadata[normalize_by] = self.val_metadata[normalize_by].astype(str)
            self.test_metadata[normalize_by] = self.test_metadata[normalize_by].astype(str)
            self.normalization_is_enabled = True
            if str(normalization_type).lower() == 'std':
                def new_channels_scalers(): # 4 is the n of channels
                    return [preprocessing.StandardScaler() for _ in range(4)]
            else:
                if str(normalization_type).lower() != 'minmax':
                    print("Unrecognized normalization type: "
                          f"{str(normalization_type)}. "
                          "Using Min Max as default.")
                def new_channels_scalers():
                    return [preprocessing.MinMaxScaler() for _ in range(4)]
            self._column_to_use_for_scale = normalize_by
            self._scalers_categories = self.metadata[normalize_by].unique()
            self._scalers = { cat : new_channels_scalers() for cat in self._scalers_categories }
            # self._scalers[ITEM IN SCALE COLUMN, FOR EXAMPLE, 1 OR 2 OF REP, OR A DATE]
                                 
            for scaler_cat in self._scalers_categories:
                features_of_cat = self._load_sample_for_fit(scaler_cat)
                self._im_size = features_of_cat.shape[1:-1]
                for i, channel_scaler in enumerate(self._scalers[scaler_cat]):
                    channel_scaler.fit([self._reshape_before_fit(x)
                                        for x in features_of_cat[:, :, :, i]])

    def _reshape_before_fit(self, single_ch):
        if self._im_size is None:
            self._im_size = single_ch.shape[0:-1]
        return single_ch.reshape(-1)

    def _reshape_after_fit(self, single_ch):
        return single_ch.reshape(self._im_size)

    def scale(self, x, scaler_cat):
        x = np.asarray(x)
        for c in range(4):  # iterate over channels  
            m_of_ch = x[:, :, c].reshape(1, -1)
            
            m_of_ch = self._scalers[scaler_cat][c].transform(m_of_ch)
            x[:, :, c] = m_of_ch.reshape(self._im_size)
        return x + 0.000001

    def _load_sample_for_fit(self, scaler_cat):
        features_of_cat = []
        metadata_of_cat = self.metadata[
            self.metadata[self._column_to_use_for_scale] == str(scaler_cat)
        ]

        for index, row in metadata_of_cat.iterrows():
            index, data = self._loader(row, **self._loader_kw)
            features_of_cat.append(data[0])
        return (np.asarray(features_of_cat))

    def load_train_val_test_data(self):
        for dataset in zip([self.train_metadata, self.val_metadata, self.test_metadata],
                           ['train', 'val', 'test']):
            if dataset is None:
                continue
            features = []
            das = []
            y = []
            for index, row in dataset[0].iterrows():
                index, data = self._loader(row, **self._loader_kw)
                data = data[0]
                if self.normalization_is_enabled:
                    data = self.scale(data, row[self._column_to_use_for_scale])
                    # data = self.scale(data, 26/02/20)

                features.append(data)
                das.append([row.DAS])
                if isinstance(self._targets, list):
                    targets = []
                    for target in self._targets:
                        targets.append(row[target])
                    y.append(targets)
                else:
                    y.append(row[self._targets])
            if dataset[1] == 'train':
                self.train_data = np.asarray(
                    features), np.asarray(das), np.asarray(y)
            elif dataset[1] == 'val':
                self.val_data = np.asarray(
                    features), np.asarray(das), np.asarray(y)
            else:
                self.test_data = np.asarray(
                    features), np.asarray(das), np.asarray(y)


class RegressorDASGenerator(Sequence, DASDataset):
    def __init__(self, metadata_flow, new_shape=None, batch_size: int = 32,
                 augmentation=1, shuffle: bool = True, temp_dir=None,
                 shape=None, use_cache=False, **dataset):
        DASDataset.__init__(self, **dataset)
        self.metadata_flow = metadata_flow.copy()
        if self.normalize_by is not None:
            self.metadata_flow[self.normalize_by] = self.metadata_flow[self.normalize_by].astype(str)
        self.load_train_val_test_data()
        self._batch_size = batch_size
        self._new_shape = new_shape
        self._augmentation = augmentation
        if shape is not None:
            self._shape = shape
        else:
            self._set_shape()
        self._temp_dir = temp_dir
        self._use_cache = use_cache
        self._cache = {}
        if temp_dir is not None:
            os.makedirs(temp_dir, exist_ok=True)
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))

        if shuffle:
            self.metadata_flow = self.metadata_flow.sample(frac=1)

    @property
    def no_augmented_dataset(self):
        """
        Loads all dataset without augmentation (train, val and test)
        """
        if self.train_data is None:
            self.load_train_val_test_data()
        return self.train_data, self.val_data, self.test_data

    def _set_shape(self):
        loaderkws = self._loader_kw.copy()
        loaderkws['augment_type'] = None
        sample = self.metadata.iloc[0, ]
        _, data = self._loader(sample, **loaderkws)
        if len(data) > 1 and self._augmentation == 1:
            print(
                f"Warning: augmented data detected. Setting augmentation to {len(data)}")
            self._augmentation = len(data)
        # data must be in shape (augmented, data shape ...)
        # where augmented must be at least equal to 1, when no augmentation
        # is performed
        self._shape = (np.asarray(data).shape)

    def _is_saved(self, item):
        saved = f'{item.Instance}_{self._shape}_{self._augmentation}.npy'
        if not os.path.isfile(os.path.join(self._temp_dir, saved)):
            return False
        return True

    def _load_from_disk(self, item):
        saved = f'{item.Instance}_{self._shape}_{self._augmentation}.npy'
        return np.load(os.path.join(self._temp_dir, saved))

    def _save_to_disk(self, item, data):
        saved = f'{item.Instance}_{self._shape}_{self._augmentation}.npy'
        np.save(os.path.join(self._temp_dir, saved), arr=data)

    def __getitem__(self, index):
        items = self.metadata_flow.iloc[index*(self._batch_size // self._augmentation):
                                   (index+1)*(self._batch_size // self._augmentation), ]
        # Fill batches
        x = []
        y = []
        das = []
        for i, item in items.iterrows():
            data = None
            if self._use_cache and item.Instance in self._cache:
                data = self._cache[item.Instance]

            if data is None and self._temp_dir is not None and self._is_saved(item):
                data = self._load_from_disk(item)

            if data is None:
                _, data = self._loader(item, **self._loader_kw)

                if self._temp_dir is not None:
                    self._save_to_disk(item, data)
                if self._use_cache:
                    self._cache[item.Instance] = data
            for d in data:
                if self.normalization_is_enabled:
                    d = self.scale(d, item[self._column_to_use_for_scale])
                # if self.standardize_by_rep or self.normalize_by_rep:
                #     d = self.scale(d, item.REP)
                if self._new_shape is not None:
                    x.append(d.reshape(self._new_shape))
                else:
                    x.append(d)
                das.append([item.DAS])
                 
                if isinstance(self._targets, list):
                    targets = []
                    for target in self._targets:
                        targets.append(item[target])
                    y.append(targets)
                else:
                    y.append(item[self._targets]) 
        return [np.asarray(x), np.asarray(das)], np.asarray(y)

    def __len__(self):
        return int((np.floor(self.metadata_flow.shape[0])*self._augmentation) / self._batch_size)



class ShowLR(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 100 == 0:
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("\nLearning rate:", K.eval(lr_with_decay))


class PlotMetricsRegressor(Callback):

    def __init__(self, val_data, val_y, interval=1,
                 suavize_data='interval', auto_limits=True,
                 savefig=None, savefig_epoch=None):
        super().__init__()
        os.makedirs('graficos/TRAINNING', exist_ok=True)
        self.val_data = val_data
        self.val_y = val_y
        self.interval = interval
        self.savefig = savefig
        self.savefig_epoch = savefig_epoch
        self.suavize_data = interval if suavize_data == 'interval' else suavize_data
        self.auto_limits = auto_limits  # Não implementado

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.mses = []
        self.vmses = []
        self.lrs = []
#         self.maes = []
#         self.vmaes = []
        self.r2s = []
        self.vr2s = []
        self.pearsons = []
        self.vpearsons = []

        self.logs = []

    @staticmethod
    def smooth_curve(points, factor=0.75):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(
                    previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    def plot(self, epoch, logs={}):
        clear_output(wait=True)
        self.fig, (self.ax1, self.ax2, self.ax3,
                   self.ax4) = plt.subplots(1, 4, figsize=(20, 5))

        if self.suavize_data is not None:
            losses = PlotMetricsRegressor.smooth_curve(self.losses)
            val_losses = PlotMetricsRegressor.smooth_curve(self.val_losses)
#             maes = PlotMetricsRegressor.smooth_curve(self.maes)
#             vmaes = PlotMetricsRegressor.smooth_curve(self.vmaes)
            mses = PlotMetricsRegressor.smooth_curve(self.mses)
            vmses = PlotMetricsRegressor.smooth_curve(self.vmses)
            r2s = PlotMetricsRegressor.smooth_curve(self.r2s)
            pearsons = PlotMetricsRegressor.smooth_curve(self.pearsons)
            vr2s = PlotMetricsRegressor.smooth_curve(self.vr2s)
            vpearsons = PlotMetricsRegressor.smooth_curve(self.vpearsons)
        else:
            losses = self.losses
            val_losses = self.val_losses
#             maes = self.maes
#             vmaes = self.vmaes
            vmses = self.vmses
            mses = self.mses
            r2s = self.r2s
            pearsons = self.pearsons
            vr2s = self.vr2s
            vpearsons = self.vpearsons
        lrs = self.lrs

        self.ax1.plot(self.x, losses, label="loss")
        self.ax1.plot(self.x, val_losses, label="val_loss")
#         self.ax1.set_xlim([max(0, self.x[-1]-30), self.x[-1]])
#         self.ax1.set_ylim([0, 10])
        self.ax1.set_xscale("log")
        self.ax1.set_yscale("log")
        self.ax1.legend()

#         self.ax2.plot(self.x, mses, label="mse")
#         self.ax2.plot(self.x, vmses, label="val_mse")
        self.ax2.plot(self.x, lrs, label="Learning Rate")
        self.ax2.legend()

        self.ax3.plot(self.x, r2s, label="R2")
        self.ax3.plot(self.x, pearsons, label="Pearson")
        self.ax3.plot(self.x, vr2s, label="Val R2")
        self.ax3.plot(self.x, vpearsons, label="Val Pearson")
        self.ax3.set_ylim([0, 1])

        self.ax3.legend()

        predictions = self.model.predict(self.val_data)
        self.ax4.scatter(self.val_y, predictions, marker='.')
        self.ax4.set_xlabel('True Values')
        self.ax4.set_ylabel('Predictions')
        self.ax4.axis('equal')
        self.ax4.axis('square')
        self.ax4.set_xlim([0, plt.xlim()[1]])
        self.ax4.set_ylim([0, plt.ylim()[1]])
        if epoch >= self.savefig_epoch:
            plt.savefig(f'graficos{os.sep}TRAINNING{os.sep}{self.savefig}.png')
        plt.show()

    def on_epoch_end(self, epoch, logs={}):
    
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.lrs.append(lr_with_decay)
        self.mses.append(logs.get('mse'))
        self.vmses.append(logs.get('val_mse'))
        self.val_losses.append(logs.get('val_loss'))
        self.r2s.append(logs.get('r2_keras'))
        self.vr2s.append(logs.get('val_r2_keras'))
        self.pearsons.append(logs.get('pearson_r'))
        self.vpearsons.append(logs.get('val_pearson_r'))
        self.i += 1
        if epoch % self.interval == 0:
            self.plot(epoch, logs)
