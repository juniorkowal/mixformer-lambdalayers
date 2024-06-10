from models.mixformer_pytorch import MixFormer, MixFormer_B0, MixFormer_B1, \
      MixFormer_B2, MixFormer_B3, MixFormer_B4, MixFormer_B5, MixFormer_B6

from models.ablation1 import MixFormer_B0_AB1, MixFormer_B1_AB1, \
      MixFormer_B2_AB1, MixFormer_B3_AB1, MixFormer_B4_AB1, MixFormer_B5_AB1, MixFormer_B6_AB1

from models.ablation2 import MixFormer_B0_AB2, MixFormer_B1_AB2, \
      MixFormer_B2_AB2, MixFormer_B3_AB2, MixFormer_B4_AB2, MixFormer_B5_AB2, MixFormer_B6_AB2

from models.ablation3 import MixFormer_B0_AB3, MixFormer_B1_AB3, \
      MixFormer_B2_AB3, MixFormer_B3_AB3, MixFormer_B4_AB3, MixFormer_B5_AB3, MixFormer_B6_AB3

from models.ablation4 import MixFormer_B0_AB4, MixFormer_B1_AB4, \
      MixFormer_B2_AB4, MixFormer_B3_AB4, MixFormer_B4_AB4, MixFormer_B5_AB4, MixFormer_B6_AB4

from models.hybrid_test import LambdaMixFormer

from models.lambda_networks import LambdaLayer
from models.lambda_networks.data_preprocessing import data_preprocessing
from models.lambda_networks.user_input import load_user_input
import torch
from tqdm import tqdm
import wandb
from datetime import datetime
from copy import deepcopy
import os
from plot_scheduler import plot_scheduler


def check_outputs(outputs, labels):
      predicted = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)
      correct = torch.argwhere(predicted==labels).size()[0]
      incorrect = torch.argwhere(predicted!=labels).size()[0]

      return correct, incorrect


def train(ab, c):
      try:
            BEST_ACC = 0
            BEST_WEIGHTS = None

            # MODEL = 'B1'
            # DATASET = dataset
            # CLASSNUM = 10 if DATASET == 'CIFAR10' else 100
            # DEBUG = False
            # if MODEL in ['B0', 'B1', 'B2', 'B3']:
            #       LEARNING_RATE = 8e-4 #(B0-B3)
            #       # LEARNING_RATE = 0.001 #experimental
            #       W_DECAY = 0.04 #(B0-B3)
            # elif MODEL in ['B4', 'B5', 'B6']:
            #       LEARNING_RATE = 1e-3 #(B4-B6)
            #       W_DECAY = 0.05 #(B4-B6)
            # else:
            #       raise NotImplementedError(f'Incorrect model name: {MODEL}')

            # if MODEL in ['B0', 'B1', 'B2', 'B3', 'B4']:
            #       WARMUP_EPOCHS = 20 #(B0-B4)
            #       # WARMUP_EPOCHS = 10 #Experimental
            # elif MODEL in ['B5', 'B6']:
            #       WARMUP_EPOCHS = 40 #(B5-B6)
            # else:
            #       raise NotImplementedError(f'Incorrect model name: {MODEL}')
            # WARMUP_LR = 1e-5
            # # WARMUP_LR = 0.001

            # B1 = 0.9
            # B2 = 0.999

            # MIN_LR = 1e-6
            # BATCH_SIZE=256
            # LOSS = 'Crossentropy'
            # OPTIMIZER = 'AdamW'
            # SCHEDULER = 'cosine'
            # EPOCHS = 200

            MODEL = c['MODEL']
            DATASET = c['DATASET']
            CLASSNUM = 10 if DATASET == 'CIFAR10' else 100
            DEBUG = c['DEBUG']
            LEARNING_RATE = c['LEARNING_RATE']
            W_DECAY = c['W_DECAY']
            WARMUP_EPOCHS = c['WARMUP_EPOCHS']
            WARMUP_LR = c['WARMUP_LR']
            B1 = c['B1']
            B2 = c['B2']
            MIN_LR = c['MIN_LR']
            BATCH_SIZE = c['BATCH_SIZE']
            LOSS = c['LOSS']
            OPTIMIZER = c['OPTIMIZER']
            SCHEDULER = c['SCHEDULER']
            EPOCHS = c['EPOCHS']
            VALID_SPLIT = c['VAL_SPLIT']

            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            inp = load_user_input(dataset=DATASET, batch_size=BATCH_SIZE)
            if VALID_SPLIT:
                  print(f'{VALID_SPLIT=} : Hyperparameters tuning mode')
                  data_loader_train, data_loader_valid, _ = data_preprocessing(inp, val_split=VALID_SPLIT)
            else:
                  print(f'{VALID_SPLIT=} : Evaluating mode (test set used as validation set)')
                  data_loader_train, _, data_loader_valid = data_preprocessing(inp, val_split=VALID_SPLIT)
            
            now = datetime.now()
            RES_PATH_FOLDER = f'./weights/{MODEL}/{now.strftime("%Y-%m-%d %H_%M_%S")} {LOSS} {DATASET} {LEARNING_RATE:.0e} sample'
            RES_PATH = f'{RES_PATH_FOLDER}/BEST.pt'

            print(f'TRAINING MODEL {MODEL}, ablation study: #{ab}')
            if MODEL == 'B0':
                  m = MixFormer_B0(
                  img_size=32,
                  class_num=CLASSNUM
                  )
            elif MODEL == 'B1':
                  if ab == 0:
                        m = MixFormer_B1(
                        img_size=32,
                        class_num=CLASSNUM,
                        # win_size=8
                        )
                  elif ab == 1:
                        m = MixFormer_B1_AB1(
                        img_size=32,
                        class_num=CLASSNUM,
                        # win_size=8
                        )
                  elif ab==2:
                        m = MixFormer_B1_AB2(
                        img_size=32,
                        class_num=CLASSNUM,
                        # win_size=8
                        )
                  elif ab==3:
                        m = MixFormer_B1_AB3(
                        img_size=32,
                        class_num=CLASSNUM,
                        # win_size=8
                        )
                  elif ab==4:
                        m = MixFormer_B1_AB4(
                        img_size=32,
                        class_num=CLASSNUM,
                        # win_size=8
                        )
                  elif ab==5:
                        m = MixFormer_B1(
                        img_size=32,
                        class_num=CLASSNUM,
                        # win_size=8
                        dwconv_kernel_size=1
                        )
                  elif ab==6:
                        m = MixFormer_B1(
                        img_size=32,
                        class_num=CLASSNUM,
                        # win_size=8
                        dwconv_kernel_size=5
                        )
            elif MODEL == 'B2':
                  m = MixFormer_B2(
                  img_size=32,
                  class_num=CLASSNUM
                  )
            elif MODEL == 'B2':
                  m = MixFormer_B2(
                  img_size=32,
                  class_num=CLASSNUM
                  )
            elif MODEL == 'B3':
                  m = MixFormer_B3(
                  img_size=32,
                  class_num=CLASSNUM
                  )
            elif MODEL == 'B4':
                  m = MixFormer_B4(
                  img_size=32,
                  class_num=CLASSNUM
                  )
            elif MODEL == 'B5':
                  m = MixFormer_B5(
                  img_size=32,
                  class_num=CLASSNUM
                  )
            elif MODEL == 'B6':
                  m = MixFormer_B6(
                  img_size=32,
                  class_num=CLASSNUM
                  )
            elif MODEL == 'HYBRID':
                  m = LambdaMixFormer(
                        img_size=32,
                        # embed_dim=24,
                        # depths=[1, 2, 6, 6],
                        # num_heads=[3, 6, 12, 24],
                        # drop_path_rate=0.,
                        class_num=CLASSNUM,
                  )
            else:
                  raise NotImplementedError(f'Incorrect model name: {MODEL}')

            m.to(device)

            if not DEBUG: 
                  wandb.init(
                        # set the wandb project where this run will be logged
                        project="ZASN",
                        
                        # track hyperparameters and run metadata
                        config={
                        "learning_rate": LEARNING_RATE,
                        "architecture": MODEL,
                        "dataset": DATASET,
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "loss": LOSS,
                        "scheduler": SCHEDULER,
                        "optimizer": OPTIMIZER,
                        "b1": B1,
                        "b2": B2,
                        "warmup_epochs": WARMUP_EPOCHS,
                        "warmup_lr": WARMUP_LR,
                        "w_decay": W_DECAY,
                        "min_lr": MIN_LR,
                        "note": now
                        }
                  )



            if LOSS == 'Crossentropy':
                  loss_fn = torch.nn.CrossEntropyLoss()

            ##Training loop
            if OPTIMIZER == 'Adam':
                  optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)
            elif OPTIMIZER == 'AdamW':
                  optimizer = torch.optim.AdamW(m.parameters(), weight_decay=W_DECAY, betas=[B1,B2], lr=LEARNING_RATE)
            if SCHEDULER == 'cosine':
                  scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=MIN_LR, T_max=200-WARMUP_EPOCHS)
                  # scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=WARMUP_LR/LEARNING_RATE, total_iters=WARMUP_EPOCHS)
                  scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=WARMUP_LR/LEARNING_RATE, end_factor=1.0, total_iters=WARMUP_EPOCHS)
                  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[WARMUP_EPOCHS])
            if SCHEDULER == 'const+cosine':
                  scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=MIN_LR, T_max=200-WARMUP_EPOCHS)
                  scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=WARMUP_EPOCHS)
                  # scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=WARMUP_LR/LEARNING_RATE, end_factor=1.0, total_iters=WARMUP_EPOCHS)
                  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[WARMUP_EPOCHS])

            # plot_scheduler(300, optimizer, scheduler)

            main_progress_bar = tqdm(
                  range(EPOCHS), desc="Training progress", position=0
            )

            for epoch in main_progress_bar:
                  main_progress_bar.set_postfix(Epoch=f"{epoch} / {EPOCHS}")
                  train_epoch_progress = tqdm(
                        data_loader_train, f"Epoch {epoch} (Train)", leave=False
                  )
                  m.train()
                  train_loss = 0
                  train_correct = 0
                  train_incorrect = 0
                  for i, data in enumerate(train_epoch_progress):
                        # Every data instance is an input + label pair
                        inputs, labels = data
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                        # Zero your gradients for every batch!
                        optimizer.zero_grad()

                        # Make predictions for this batch
                        outputs = m(inputs)
                        # Compute the loss and its gradients
                        loss = loss_fn(outputs, labels)
                        loss.backward()

                        # Adjust learning weights
                        optimizer.step()
                        correct, incorrect = check_outputs(outputs, labels)
                        train_correct += correct
                        train_incorrect += incorrect

                        # Gather data and report
                        train_loss += loss.item()
                        train_epoch_progress.set_postfix(
                              Loss=f"{train_loss / (1+i):.4f}",
                        )
                  scheduler.step()
                  train_loss = train_loss / len(train_epoch_progress)
                  train_accuracy = train_correct / (train_correct+train_incorrect)
                  if DEBUG:
                        print(f"{train_accuracy=} \n {train_loss=}")

                  train_epoch_progress.close()
                  m.eval()
                  with torch.no_grad():
                        # train_loss = 0
                        # train_correct = 0
                        # train_incorrect = 0
                        # train_epoch_progress = tqdm(
                        #       data_loader_train, f"[EVAL] Epoch {epoch} (Train)", leave=False
                        # )
                        # for i, data in enumerate(train_epoch_progress):
                        #       inputs, labels = data
                        #       inputs = inputs.to(device, non_blocking=True)
                        #       labels = labels.to(device, non_blocking=True)

                        #       outputs = m(inputs)
                        #       loss = loss_fn(outputs, labels)
                        #       train_loss += loss.item()
                        #       correct, incorrect = check_outputs(outputs, labels)
                        #       train_correct += correct
                        #       train_incorrect += incorrect
                        #       train_epoch_progress.set_postfix(
                        #       Loss=f"{train_loss / (1+i):.4f}",
                        #       )
                        # train_loss = train_loss / len(train_epoch_progress)
                        # train_accuracy = train_correct / (train_correct+train_incorrect)
                        # print(f"{train_accuracy=} \n {train_loss=}")


                        val_loss = 0
                        val_correct = 0
                        val_incorrect = 0
                        val_epoch_progress = tqdm(
                              data_loader_valid, f"[EVAL] Epoch {epoch} (Valid)", leave=False
                        )
                        for i, data in enumerate(val_epoch_progress):
                              inputs, labels = data
                              inputs = inputs.to(device, non_blocking=True)
                              labels = labels.to(device, non_blocking=True)

                              outputs = m(inputs)
                              loss = loss_fn(outputs, labels)
                              val_loss += loss.item()
                              correct, incorrect = check_outputs(outputs, labels)
                              val_correct += correct
                              val_incorrect += incorrect
                              val_epoch_progress.set_postfix(
                              Loss=f"{val_loss / (1+i):.4f}",
                              )

                        val_loss = val_loss / len(val_epoch_progress)
                        val_accuracy = val_correct / (val_correct+val_incorrect)
                        if BEST_ACC < val_accuracy:
                              BEST_ACC = val_accuracy
                              BEST_WEIGHTS = deepcopy(m.state_dict())
                        if DEBUG:
                              print(f"{val_accuracy=} \n {val_loss=}")
                        if not DEBUG: 
                              wandb.log(
                                    {
                                          "[VALID] Loss": val_loss,
                                          "[TRAIN] Loss": train_loss,

                                          "[VALID] Accuracy": val_accuracy,
                                          "[TRAIN] Accuracy": train_accuracy,

                                          "Learning_Rate": scheduler.get_last_lr()[0]

                                    }
                                    )
            os.makedirs(RES_PATH_FOLDER, exist_ok=True)
            torch.save(m.state_dict(), rf'{RES_PATH_FOLDER}\LAST.pt')
            torch.save(BEST_WEIGHTS, rf'{RES_PATH_FOLDER}\BEST_ACC_{BEST_ACC}.pt')
            wandb.finish()
      except KeyboardInterrupt:
            os.makedirs(RES_PATH_FOLDER, exist_ok=True)
            torch.save(m.state_dict(), rf'{RES_PATH_FOLDER}\LAST.pt')
            torch.save(BEST_WEIGHTS, rf'{RES_PATH_FOLDER}\BEST_ACC_{BEST_ACC}.pt')
            wandb.finish()
                        
if __name__ == '__main__':
      # base_config = {
      # 'MODEL': 'B1',
      # 'DATASET': 'CIFAR10',
      # 'DEBUG': False,
      # 'LEARNING_RATE': 0.001,
      # 'W_DECAY': 0.04,
      # 'WARMUP_EPOCHS': 20,
      # 'WARMUP_LR': 1e-5,
      # 'B1': 0.9,
      # 'B2': 0.999,
      # 'MIN_LR': 1e-6,
      # 'BATCH_SIZE': 256,
      # 'LOSS': 'Crossentropy',
      # 'OPTIMIZER': 'AdamW',
      # 'SCHEDULER': 'cosine',
      # 'EPOCHS': 200
      # }
      configs = [
            # {
            # 'MODEL': 'B1',
            # 'DATASET': 'CIFAR10',
            # 'DEBUG': False,
            # 'LEARNING_RATE': 0.001,
            # 'W_DECAY': 0.04,
            # 'WARMUP_EPOCHS': 10, ######
            # 'WARMUP_LR': 1e-5,
            # 'B1': 0.9,
            # 'B2': 0.999,
            # 'MIN_LR': 1e-6,
            # 'BATCH_SIZE': 256,
            # 'LOSS': 'Crossentropy',
            # 'OPTIMIZER': 'AdamW',
            # 'SCHEDULER': 'cosine',
            # 'EPOCHS': 200
            # },
            # {
            # 'MODEL': 'B1',
            # 'DATASET': 'CIFAR10',
            # 'DEBUG': False,
            # 'LEARNING_RATE': 0.001,
            # 'W_DECAY': 0.04,
            # 'WARMUP_EPOCHS': 20,
            # 'WARMUP_LR': 1e-5,
            # 'B1': 0.9,
            # 'B2': 0.999,
            # 'MIN_LR': 1e-6,
            # 'BATCH_SIZE': 256,
            # 'LOSS': 'Crossentropy',
            # 'OPTIMIZER': 'AdamW',
            # 'SCHEDULER': 'const+cosine', ######
            # 'EPOCHS': 200
            # },
            # {
            # 'MODEL': 'B1',
            # 'DATASET': 'CIFAR10',
            # 'DEBUG': False,
            # 'LEARNING_RATE': 0.006,  ######
            # 'W_DECAY': 0.04,
            # 'WARMUP_EPOCHS': 20,
            # 'WARMUP_LR': 1e-5,
            # 'B1': 0.9,
            # 'B2': 0.999,
            # 'MIN_LR': 1e-6,
            # 'BATCH_SIZE': 256,
            # 'LOSS': 'Crossentropy',
            # 'OPTIMIZER': 'AdamW',
            # 'SCHEDULER': 'cosine',
            # 'EPOCHS': 200
            # },
            # {
            # 'MODEL': 'B1',
            # 'DATASET': 'CIFAR10',
            # 'DEBUG': False,
            # 'LEARNING_RATE': 0.006,  ######
            # 'W_DECAY': 0.04,
            # 'WARMUP_EPOCHS': 20,
            # 'WARMUP_LR': 1e-5,
            # 'B1': 0.9,
            # 'B2': 0.999,
            # 'MIN_LR': 1e-6,
            # 'BATCH_SIZE': 256,
            # 'LOSS': 'Crossentropy',
            # 'OPTIMIZER': 'AdamW',
            # 'SCHEDULER': 'const+cosine',  ######
            # 'EPOCHS': 200
            # },
            # {
            # 'MODEL': 'B1',
            # 'DATASET': 'CIFAR10',
            # 'DEBUG': False,
            # 'LEARNING_RATE': 0.006,  ######
            # 'W_DECAY': 0.04,
            # 'WARMUP_EPOCHS': 50,
            # 'WARMUP_LR': 1e-5,
            # 'B1': 0.9,
            # 'B2': 0.999,
            # 'MIN_LR': 1e-6,
            # 'BATCH_SIZE': 256,
            # 'LOSS': 'Crossentropy',
            # 'OPTIMIZER': 'AdamW',
            # 'SCHEDULER': 'const+cosine',  ######
            # 'EPOCHS': 200,
            # 'VAL_SPLIT': False
            # },
            ######################## HYBRID
            {
            'MODEL': 'HYBRID',
            'DATASET': 'CIFAR10',
            'DEBUG': False,
            'LEARNING_RATE': 0.001,
            'W_DECAY': 0.04,
            'WARMUP_EPOCHS': 20,
            'WARMUP_LR': 0.001,
            'B1': 0.9,
            'B2': 0.999,
            'MIN_LR': 1e-6,
            'BATCH_SIZE': 128,
            'LOSS': 'Crossentropy',
            'OPTIMIZER': 'AdamW',
            'SCHEDULER': 'const+cosine',
            'EPOCHS': 200,
            'VAL_SPLIT': True
            },
            {
            'MODEL': 'HYBRID',
            'DATASET': 'CIFAR10',
            'DEBUG': False,
            'LEARNING_RATE': 0.01,  ######
            'W_DECAY': 0.04,
            'WARMUP_EPOCHS': 20,
            'WARMUP_LR': 0.01,
            'B1': 0.9,
            'B2': 0.999,
            'MIN_LR': 1e-6,
            'BATCH_SIZE': 128,
            'LOSS': 'Crossentropy',
            'OPTIMIZER': 'AdamW',
            'SCHEDULER': 'const+cosine',
            'EPOCHS': 200,
            'VAL_SPLIT': True
            },
            {
            'MODEL': 'HYBRID',
            'DATASET': 'CIFAR10',
            'DEBUG': False,
            'LEARNING_RATE': 0.001,
            'W_DECAY': 0.04,
            'WARMUP_EPOCHS': 10, #####
            'WARMUP_LR': 0.001,
            'B1': 0.9,
            'B2': 0.999,
            'MIN_LR': 1e-6,
            'BATCH_SIZE': 128,
            'LOSS': 'Crossentropy',
            'OPTIMIZER': 'AdamW',
            'SCHEDULER': 'const+cosine',
            'EPOCHS': 200,
            'VAL_SPLIT': True
            },
############################################################# SCHEDULER
            {
            'MODEL': 'HYBRID',
            'DATASET': 'CIFAR10',
            'DEBUG': False,
            'LEARNING_RATE': 0.001,
            'W_DECAY': 0.04,
            'WARMUP_EPOCHS': 20,
            'WARMUP_LR': 1e-5,
            'B1': 0.9,
            'B2': 0.999,
            'MIN_LR': 1e-6,
            'BATCH_SIZE': 128,
            'LOSS': 'Crossentropy',
            'OPTIMIZER': 'AdamW',
            'SCHEDULER': 'cosine',
            'EPOCHS': 200,
            'VAL_SPLIT': True
            },
            {
            'MODEL': 'HYBRID',
            'DATASET': 'CIFAR10',
            'DEBUG': False,
            'LEARNING_RATE': 0.01,  ######
            'W_DECAY': 0.04,
            'WARMUP_EPOCHS': 20,
            'WARMUP_LR': 1e-5,
            'B1': 0.9,
            'B2': 0.999,
            'MIN_LR': 1e-6,
            'BATCH_SIZE': 128,
            'LOSS': 'Crossentropy',
            'OPTIMIZER': 'AdamW',
            'SCHEDULER': 'cosine',
            'EPOCHS': 200,
            'VAL_SPLIT': True
            },
            {
            'MODEL': 'HYBRID',
            'DATASET': 'CIFAR10',
            'DEBUG': False,
            'LEARNING_RATE': 0.001,
            'W_DECAY': 0.04,
            'WARMUP_EPOCHS': 10, #####
            'WARMUP_LR': 1e-5,
            'B1': 0.9,
            'B2': 0.999,
            'MIN_LR': 1e-6,
            'BATCH_SIZE': 128,
            'LOSS': 'Crossentropy',
            'OPTIMIZER': 'AdamW',
            'SCHEDULER': 'cosine',
            'EPOCHS': 200,
            'VAL_SPLIT': True
            },
            {
            'MODEL': 'HYBRID',
            'DATASET': 'CIFAR10',
            'DEBUG': False,
            'LEARNING_RATE': 0.01, #####
            'W_DECAY': 0.04,
            'WARMUP_EPOCHS': 10, #####
            'WARMUP_LR': 1e-5,
            'B1': 0.9,
            'B2': 0.999,
            'MIN_LR': 1e-6,
            'BATCH_SIZE': 128,
            'LOSS': 'Crossentropy',
            'OPTIMIZER': 'AdamW',
            'SCHEDULER': 'cosine',
            'EPOCHS': 200,
            'VAL_SPLIT': True
            },
            {
            'MODEL': 'HYBRID',
            'DATASET': 'CIFAR10',
            'DEBUG': False,
            'LEARNING_RATE': 0.01, #####
            'W_DECAY': 0.04,
            'WARMUP_EPOCHS': 10, #####
            'WARMUP_LR': 0.01,
            'B1': 0.9,
            'B2': 0.999,
            'MIN_LR': 1e-6,
            'BATCH_SIZE': 128,
            'LOSS': 'Crossentropy',
            'OPTIMIZER': 'AdamW',
            'SCHEDULER': 'const+cosine',
            'EPOCHS': 200,
            'VAL_SPLIT': True
            },
      ]
      for config in configs:
            config['DEBUG'] = False
            config['EPOCHS'] = 150
            config['BATCH_SIZE'] = 128
            train(0, config)
      # for ablation in range(7):
      #       # configs[0]['DEBUG'] = True
      #       # configs[0]['EPOCHS'] = 1
      #       train(ablation,configs[0])
