
Run 1:
commit: 3bb0a3a
batch size: 128
lr: 1e-3
patch size: 4
embed dim: 8
epochs: 10

train accuracy: 0.49
val accuracy: 0.51
Comment: Single head attention, no positional embedding. 7k large model not fully trained


Run 2:
commit: 65e0e68
batch size: 128
lr: 1e-3
patch size: 4
embed dim: 8
epochs: 20

train accuracy: 0.87
val accuracy: 0.87
Comment: Added global positional embedding. Seems close to fully trained. Already at ten epochs the train and val accuracy are above 80 %



Run 3:
commit: 43be92f
batch size: 128
lr: 1e-3
patch size: 4
embed dim: 8
epochs: 20

train accuracy: 0.88
val accuracy: 0.89
Comment: Added a CLS token and used its output as the embedding to send to the classifier. 
