import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# from model import *
from model_attention import *
from util_serv import *
import torch.optim as optim
from torch.utils.data import DataLoader

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

VAL_FAKE_ROOT1 = '00-Datasets/00-FF++/Deepfakes/raw/val'
VAL_REAL_ROOT1 = '00-Datasets/00-FF++/Real/raw/val'

VAL_FAKE_ROOT2 = '00-Datasets/00-FF++/FaceShifter/raw/val'
VAL_REAL_ROOT2 = '00-Datasets/00-FF++/Real/raw/val'

VAL_FAKE_ROOT3 = '00-Datasets/00-FF++/FaceSwap/raw/val'
VAL_REAL_ROOT3 = '00-Datasets/00-FF++/Real/raw/val'

VAL_FAKE_ROOT4 = '00-Datasets/00-FF++/Face2Face/raw/val'
VAL_REAL_ROOT4 = '00-Datasets/00-FF++/Real/raw/val'

TYPE = 'c40'
EPOCH = 50
BATCH_SIZE = 32
LENGTH = BATCH_SIZE * 200

# net = resnet18().to(device)
net = aspp_a_net().to(device)
pretext_model = torch.load(r'/data2/Jianwei-Fei/01-Projects/02-Similarity/00-final/resnet18-5c106cde.pth')
model2_dict = net.state_dict()
state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
state_dict.pop('fc.weight')
state_dict.pop('fc.bias')
model2_dict.update(state_dict)
net.load_state_dict(model2_dict)

net.to(device)
dealDataset = DealDataset(LENGTH=LENGTH, TYPE=TYPE)
train_loader = DataLoader(dataset=dealDataset, batch_size=BATCH_SIZE, shuffle=True)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)


if __name__ == '__main__':
    print('Attention train on DF-c23')
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        print('-'*80)
        net.train()
        step = 0
        for i, data in enumerate(train_loader, 0):  # 从零开始
            inputs, label = data
            inputs, label = inputs.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()

            data = '[epoch:%03d, iter:%03d] Loss: %.03f' % (epoch + 1, i, loss.item())
            with open('runs/log.txt', 'a', encoding='utf-8') as f:
                f.write(data)
                f.write('\n')

            if step % 10 == 0:
                print(data)
            step += 1
        with torch.no_grad():
            auc1, acc1 = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT1, VAL_REAL_ROOT=VAL_REAL_ROOT1, ACC=True)
            auc2, acc2 = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT2, VAL_REAL_ROOT=VAL_REAL_ROOT2, ACC=True)
            auc3, acc3 = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT3, VAL_REAL_ROOT=VAL_REAL_ROOT3, ACC=True)
            auc4, acc4 = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT4, VAL_REAL_ROOT=VAL_REAL_ROOT4, ACC=True)

        # Deepfakes Face2Face FaceSwap NeuralTextures
        tag = 'epoch-%03d-loss-%.03f\n' \
              '---DFAUC-%.04f---DFACC-%.04f\n' \
              '--F2FAUC-%.04f--F2FACC-%.04f\n' \
              '---FSAUC-%.04f---FSACC-%.04f\n' \
              '---NTAUC-%.04f---NTACC-%.04f' % \
               (epoch + 1, loss.item(), auc1, acc1, auc2, acc2, auc3, acc3, auc4, acc4)

        # if acc1 == 0:
        #     tag = 'epoch-%03d-loss-%.03f' \
        #           '--NTAUC-%.04f' % \
        #           (epoch + 1, loss.item(), auc1)
        # else:
        #     tag = 'epoch-%03d-loss-%.03f' \
        #             '--NTAUC-%.04f--NTACC-%.04f' % \
        #             (epoch + 1, loss.item(), auc1, acc1)

        # if acc3 == 0:
        #     tag = 'epoch-%03d-loss-%.03f' \
        #           '--FSAUC-%.04f' % \
        #           (epoch + 1, loss.item(), auc3)
        # else:
        #     tag = 'epoch-%03d-loss-%.03f' \
        #             '--FSAUC-%.04f--FSACC-%.04f' % \
        #             (epoch + 1, loss.item(), auc3, acc3)

        # tag_cross = 'epoch-%03d-loss-%.03f' \
        #       '--DFAUC-%.04f' \
        #       '--FSFAUC-%.04f' \
        #       '--FSAUC-%.04f' \
        #       '--F2FAUC-%.04f' % \
        #       (epoch + 1, loss.item(), auc1, auc2, auc3, auc4)
        print(tag)
        # print('Average AUC:', round((auc1+auc2+auc3+auc4)/4, 4))
        print('-'*50)
        with open('runs/log.txt', 'a', encoding='utf-8') as f:
            f.write(tag)
            f.write('\n')
        # torch.save(net, r'models/' + tag + '.pkl')
