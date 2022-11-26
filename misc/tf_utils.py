import tensorflow as tf


class ConvergenceEarlyStopping(tf.keras.callbacks.Callback):
  def __init__(self, eps=1e-8, patience=1):
    super(ConvergenceEarlyStopping, self).__init__()
    self.eps = eps
    self.loss_history = []
    self.patience = patience
    self.countdown = self.patience

  def on_epoch_end(self, epoch, logs=None):
    self.loss_history.append(logs.get('loss'))
    if len(self.loss_history) > 1 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.eps:
      if self.countdown == 0:
        self.model.stop_training = True
      else:
        self.countdown -= 1
    else:
      self.countdown = self.patience