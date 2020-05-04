import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from DynAE import DynAE
from datasets import load_data
from datasets import load_specific_handwriting  # Experiment
from convert import label_pruning
import metrics

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

dataset ='handwriting'
loss_weight_lambda=0.5
save_dir='/home/yeongjoon/models/image_cluster/DynAE/tmp_results'
visualisation_dir='/home/yeongjoon/models/image_cluster/DynAE/visualization'
data_path ='/home/yeongjoon/data/image_cluster/data/hanza_ocr/' + dataset

# hyperparameter
# batch_size=256
# maxiter_pretraining=130e3
# maxiter_clustering=1e5

# Temporal hyperparamter
batch_size=16
maxiter_pretraining=130e2
maxiter_clustering=1e3

# Temporal hyperparameter
# maxiter_pretraining=130e1
# maxiter_clustering=1e1
tol=0.01
optimizer1=SGD(0.001, 0.9)
optimizer2=tf.train.AdamOptimizer(0.0001)
kappa = 3
ws=0.1
hs=0.1
rot=10
scale=0.
gamma=100

# x, y = load_data(dataset, data_path)

# Experiment
syllable_list = ['가', '나', '다', '라', '마', '바', '사', '아', '자', '차']
x, y = load_specific_handwriting(data_path=data_path, syallble_list=syllable_list, width=100)
x = x.reshape([x.shape[0], -1])

n_clusters = len(np.unique(y))
# n_clusters = 100   # 강제로 지정
# y = None
model = DynAE(batch_size=batch_size, dataset=dataset, dims=[x.shape[-1], 5000, 5000, 10000, n_clusters], loss_weight=loss_weight_lambda, gamma=gamma, n_clusters=n_clusters, visualisation_dir=visualisation_dir, ws=ws, hs=hs, rot=rot, scale=scale)
# model = DynAE(batch_size=batch_size, dataset=dataset, dims=[x.shape[-1], 500, 500, 2000, 10], loss_weight=loss_weight_lambda, gamma=gamma, n_clusters=n_clusters, visualisation_dir=visualisation_dir, ws=ws, hs=hs, rot=rot, scale=scale)
model.compile_dynAE(optimizer=optimizer2)
model.compile_disc(optimizer=optimizer2)
model.compile_aci_ae(optimizer=optimizer2)

#Load the pretraining weights if you have already pretrained your network

#TODO 이 부분 항상 확인

model.ae.load_weights(save_dir + '/' + dataset + '/pretrain/ae_weights.h5')
model.critic.load_weights(save_dir + '/' + dataset + '/pretrain/critic_weights.h5')

#Pretraining phase

# model.train_aci_ae(x, y=None, maxiter=maxiter_pretraining, batch_size=batch_size, validate_interval=2800, save_interval=2800, save_dir=save_dir, verbose=1, aug_train=True)
#
# #Save the pretraining weights if you do not want to pretrain your model again
#
# model.ae.save_weights(save_dir + '/' + dataset + '/pretrain/ae_weights.h5')
# model.critic.save_weights(save_dir + '/' + dataset + '/pretrain/critic_weights.h5')

#Clustering phase

# y_pred = model.train_dynAE(x=x, y=y, kappa=kappa, n_clusters=n_clusters, maxiter=maxiter_clustering, batch_size=batch_size, tol=tol, validate_interval=140, show_interval=None, save_interval=2800, save_dir=save_dir, aug_train=True)
#
# #Save the clustering weights
#
# model.ae.save_weights(save_dir + '/' + dataset + '/cluster/ae_weights.h5')

#Load the clustering weights

model.ae.load_weights(save_dir + '/' + dataset + '/cluster/ae_weights.h5')

#Print ACC and NMI

model.compute_acc_and_nmi(x, y)

#Print only y_pred
y_pred = model.predict_y(x, n_clusters)
dic = {'x':x, 'y_pred':y_pred}
import pickle
with open(save_dir + '/' + dataset + '/x_and_y_pred.pkl', 'wb') as f:
    pickle.dump(dic, f)
